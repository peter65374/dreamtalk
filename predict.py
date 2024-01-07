# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import subprocess
import shutil
import numpy as np
import torch
import torchaudio
from scipy.io import loadmat
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model
from cog import BasePredictor, Input, Path

from configs.default import get_cfg_defaults
from core.networks.diffusion_net import DiffusionNet
from core.networks.diffusion_util import NoisePredictor, VarianceSchedule
from core.utils import (
    crop_src_image,
    get_pose_params,
    get_video_style_clip,
    get_wav2vec_audio_window,
)
from generators.utils import get_netG, render_video

STYLES_MAT = [
    os.path.join("data/style_clip/3DMM", file)
    for file in os.listdir("data/style_clip/3DMM")
]
POSE_MAT = [os.path.join("data/pose", file) for file in os.listdir("data/pose")]


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # pass

        cache_dir = "checkpoints"
        local_files_only = True  # set to True if model is cached locally
        self.device = torch.device("cuda:0")
        # get wav2vec feat from audio
        self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(
            "jonatasgrosman/wav2vec2-large-xlsr-53-english",
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.wav2vec_model = (
            Wav2Vec2Model.from_pretrained(
                "jonatasgrosman/wav2vec2-large-xlsr-53-english",
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            .eval()
            .to(self.device)
        )
        # get renderer
        self.renderer = get_netG("checkpoints/renderer.pt", self.device)

        self.cfg = get_cfg_defaults()
        self.diff_net = get_diff_net(self.cfg, self.device).to(self.device)

    def predict(
        self,
        image: Path = Input(
            description="Input image. This specifies the input portrait. The resolution should be larger than 256x256 and will be cropped to 256x256."
        ),
        audio: Path = Input(
            description="Input audio file. The input audio file extensions should be wav, mp3, m4a, and mp4 (video with sound) should all be compatible."
        ),
        style_clip: str = Input(
            description="Input style_clip_mat, optional. This specifies the reference speaking style.",
            choices=STYLES_MAT,
            default="data/style_clip/3DMM/M030_front_neutral_level1_001.mat",
        ),
        pose: str = Input(
            description="Input pose, specifies the head pose and should be a .mat file.",
            choices=POSE_MAT,
            default=POSE_MAT[0],
        ),
        max_gen_len: int = Input(
            description="The maximum length (seconds) limitation for generating videos.",
            default=1000,
        ),
        # cfg_scale: float = Input(
        #     description="The scale of classifier-free guidance. It can adjust the intensity of speaking styles.",
        #     default=1.0,
        # ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=10
        ),
        crop_image: bool = Input(
            description="Enable cropping the input image. If your portrait is already cropped to 256x256, set this to False.",
            default=True,
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        # self.cfg = get_cfg_defaults()
        # cache_dir = "checkpoints"
        # local_files_only = True  # set to True if model is cached locally
        # self.device = torch.device("cuda:0")
        # # get wav2vec feat from audio
        # self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(
        #     "jonatasgrosman/wav2vec2-large-xlsr-53-english",
        #     cache_dir=cache_dir,
        #     local_files_only=local_files_only,
        # )
        # self.wav2vec_model = (
        #     Wav2Vec2Model.from_pretrained(
        #         "jonatasgrosman/wav2vec2-large-xlsr-53-english",
        #         cache_dir=cache_dir,
        #         local_files_only=local_files_only,
        #     )
        #     .eval()
        #     .to(self.device)
        # )
        # # get renderer
        # self.renderer = get_netG("checkpoints/renderer.pt", self.device)

        tmp_dir = "cog_temp"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)

        # get audio in 16000Hz
        wav_16k_path = os.path.join(tmp_dir, "tmp_input_16K.wav")
        command = f"ffmpeg -y -i {str(audio)} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {wav_16k_path}"
        subprocess.run(command.split())

        speech_array, sampling_rate = torchaudio.load(wav_16k_path)
        audio_data = speech_array.squeeze().numpy()
        inputs = self.wav2vec_processor(
            audio_data, sampling_rate=16_000, return_tensors="pt", padding=True
        )

        with torch.no_grad():
            audio_embedding = self.wav2vec_model(
                inputs.input_values.to(self.device), return_dict=False
            )[0]

        audio_feat_path = os.path.join(tmp_dir, "tmp_wav2vec.npy")
        np.save(audio_feat_path, audio_embedding[0].cpu().numpy())

        # get src image
        src_img_path = os.path.join(tmp_dir, "src_img.png")
        if crop_image:
            crop_src_image(str(image), src_img_path, 0.4)
        else:
            shutil.copy(str(image), src_img_path)

        with torch.no_grad():
            # get diff model and load checkpoint
            # self.cfg.CF_GUIDANCE.SCALE = cfg_scale
            # diff_net = get_diff_net(self.cfg, self.device).to(self.device)

            # generate face motion
            face_motion_path = os.path.join(tmp_dir, "facemotion.npy")
            inference_one_video(
                self.cfg,
                audio_feat_path,
                str(style_clip),
                str(pose),
                face_motion_path,
                self.diff_net,
                self.device,
                max_audio_len=max_gen_len,
                ddim_num_step=num_inference_steps,
            )

            # render video
            no_watermark_video_path = os.path.join(tmp_dir, "no_watermark.mp4")

            render_video(
                self.renderer,
                src_img_path,
                face_motion_path,
                wav_16k_path,
                no_watermark_video_path,
                self.device,
                fps=25,
                no_move=False,
            )

            # add watermark
            output_video_path = "tmp/out.mp4"
            os.system(
                f'ffmpeg -y -i {no_watermark_video_path} -vf  "movie=media/watermark.png,scale= 120: 36[watermask]; [in] [watermask] overlay=140:220 [out]" {output_video_path}'
            )

        return Path(output_video_path)


@torch.no_grad()
def get_diff_net(cfg, device):
    diff_net = DiffusionNet(
        cfg=cfg,
        net=NoisePredictor(cfg),
        var_sched=VarianceSchedule(
            num_steps=cfg.DIFFUSION.SCHEDULE.NUM_STEPS,
            beta_1=cfg.DIFFUSION.SCHEDULE.BETA_1,
            beta_T=cfg.DIFFUSION.SCHEDULE.BETA_T,
            mode=cfg.DIFFUSION.SCHEDULE.MODE,
        ),
    )
    checkpoint = torch.load(cfg.INFERENCE.CHECKPOINT, map_location=device)
    model_state_dict = checkpoint["model_state_dict"]
    diff_net_dict = {
        k[9:]: v for k, v in model_state_dict.items() if k[:9] == "diff_net."
    }
    diff_net.load_state_dict(diff_net_dict, strict=True)
    diff_net.eval()

    return diff_net


@torch.no_grad()
def inference_one_video(
    cfg,
    audio_path,
    style_clip_path,
    pose_path,
    output_path,
    diff_net,
    device,
    max_audio_len=None,
    sample_method="ddim",
    ddim_num_step=10,
):
    audio_raw = np.load(audio_path)

    if max_audio_len is not None:
        audio_raw = audio_raw[: max_audio_len * 50]
    gen_num_frames = len(audio_raw) // 2

    audio_win_array = get_wav2vec_audio_window(
        audio_raw,
        start_idx=0,
        num_frames=gen_num_frames,
        win_size=cfg.WIN_SIZE,
    )

    audio_win = torch.tensor(audio_win_array).to(device)
    audio = audio_win.unsqueeze(0)

    # the second parameter is "" because of bad interface design...
    style_clip_raw, style_pad_mask_raw = get_video_style_clip(
        style_clip_path, "", style_max_len=256, start_idx=0
    )

    style_clip = style_clip_raw.unsqueeze(0).to(device)
    style_pad_mask = (
        style_pad_mask_raw.unsqueeze(0).to(device)
        if style_pad_mask_raw is not None
        else None
    )

    gen_exp_stack = diff_net.sample(
        audio,
        style_clip,
        style_pad_mask,
        output_dim=cfg.DATASET.FACE3D_DIM,
        use_cf_guidance=cfg.CF_GUIDANCE.INFERENCE,
        cfg_scale=cfg.CF_GUIDANCE.SCALE,
        sample_method=sample_method,
        ddim_num_step=ddim_num_step,
    )
    gen_exp = gen_exp_stack[0].cpu().numpy()

    pose = get_pose_params(pose_path)
    if len(pose) >= len(gen_exp):
        selected_pose = pose[: len(gen_exp)]
    else:
        selected_pose = pose[-1].unsqueeze(0).repeat(len(gen_exp), 1)
        selected_pose[: len(pose)] = pose

    gen_exp_pose = np.concatenate((gen_exp, selected_pose), axis=1)
    np.save(output_path, gen_exp_pose)
    return output_path
