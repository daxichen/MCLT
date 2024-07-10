# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from PIL import Image, ImageFilter, ImageOps
import math
import random
import torchvision.transforms.functional as tf
import torch
from torch import nn
from torchvision.transforms import transforms
import numpy as np


class TwoCropsTransform(torch.utils.data.Dataset):

    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2/1.0

    def __getitem__(self, index):
        index = index
        im1 = (self.base_transform1[index])
        im2 = (self.base_transform2[index])
        return torch.FloatTensor(im1), torch.FloatTensor(im2)

    def __len__(self):
        return len(self.base_transform1)

    


# class GaussianBlur(object):
#     """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""
#
#     def __init__(self, sigma=[.1, 2.]):
#         self.sigma = sigma
#
#     def __call__(self, x):
#         sigma = random.uniform(self.sigma[0], self.sigma[1])
#         x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
#         return x



class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = int(kernel_size // 2)
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(1, 1, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=1)
        self.blur_v = nn.Conv2d(1, 1, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=1)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r+1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1)

        self.blur_h.weight.data.copy_(x.view(1, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(1, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img




class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)