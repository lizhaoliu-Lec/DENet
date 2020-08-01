import torch

import numpy as np
from PIL import Image

from torchvision import transforms


class Normalize(object):
    def __init__(self, mean, std):
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, img, mask):
        return self.normalize(img), mask


class ToTensor(object):
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img, mask):
        return self.to_tensor(img), torch.from_numpy(np.array(mask, dtype=np.long))


class Resize(object):
    def __init__(self, size):
        self.resize_img = transforms.Resize(size, interpolation=Image.BILINEAR)
        self.resize_mask = transforms.Resize(size, interpolation=Image.NEAREST)

    def __call__(self, img, mask):
        return self.resize_img(img), self.resize_mask(mask)
