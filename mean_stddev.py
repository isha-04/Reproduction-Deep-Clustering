"""
Finding the mean and standard deviation of the image set (for normalizing)
"""

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from torch.utils.data import Dataset
import os
from natsort import natsorted
from PIL import Image


class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, index):
        img_loc = os.path.join(self.main_dir, self.total_imgs[index])
        image = Image.open(img_loc)
        tensor_image = self.transform(image)
        return tensor_image


data_path = 'tiny-imagenet/train/'

transform_img = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

image_data = torchvision.datasets.ImageFolder(
    root=data_path, transform=transform_img
)


image_data_loader = DataLoader(
    image_data, 
    batch_size=len(image_data), 
    shuffle=False, 
    num_workers=0
)

batch_size = 2

loader = DataLoader(
    image_data, 
    batch_size = batch_size, 
    num_workers=0
)

def batch_mean_and_sd(loader):
    
    num = 0
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels
        num += 1

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)  
    print(num)
    return mean,std


mean, std = batch_mean_and_sd(loader)
print("mean and std: \n", mean, std)