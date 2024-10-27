import torch
from torchvision.io import read_image

# df = pd.read_csv('/workspace/Project/Multi_imagery_SFPred/Dataset/label/24image_GOES_classification_Fold1_val.csv')
# df['Timestamp'] = pd.to_datetime(df['Timestamp'], format = '%Y-%m-%d %H:%M:%S')

# print(df['Timestamp'][0].second)

dir1 = '/workspace/data/hetero_data/hmi/compressed/mag/2016/01/02/HMI-Mag.2016.01.02_04.00.00.jpg'
dir2 = '/workspace/data/hetero_data/hmi/compressed/mag/2016/01/02/HMI-Mag.2016.01.02_05.00.00.jpg'
dir3 = '/workspace/data/hetero_data/hmi/compressed/mag/2016/01/02/HMI-Mag.2016.01.02_06.00.00.jpg'
image1 = read_image(dir1)
image2 = read_image(dir2)
image3 = read_image(dir3)
check = read_image('/workspace/data/Multi.m2010.12.06_07.00.00.png')

image = torch.stack([image1, image2, image3], dim = 0)
print(image)
print(type(image))
print(type(check))
print(image.shape)
print(image.squeeze().shape)
print(check.shape)