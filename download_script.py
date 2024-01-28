# download checkpoints from google drive to ./checkpoints
# renderer.pt https://drive.google.com/file/d/1BOZNkAhJBTV8_nbk_eu51WPCkJoYuglg/view?usp=sharing
# denoising_network.pth https://drive.google.com/file/d/1JbX7UVfLvpVWHT5q4doLo6RQdissjB2_/view?usp=sharing
import gdown

# download renderer.pt
url1 = 'https://drive.google.com/uc?id=1BOZNkAhJBTV8_nbk_eu51WPCkJoYuglg'
output1 = './checkpoints/renderer.pt'
gdown.download(url1, output1, quiet=False)

# download denoising_network.pth
url2 = 'https://drive.google.com/uc?id=1JbX7UVfLvpVWHT5q4doLo6RQdissjB2_'
output2 = './checkpoints/denoising_network.pth'
gdown.download(url2, output2, quiet=False)
