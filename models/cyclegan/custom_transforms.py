import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        imgA = sample['A']
        imgB = sample['B']
        mask = sample['label']

        imgA = np.array(imgA).astype(np.float32)
        imgB = np.array(imgB).astype(np.float32)
        mask = np.array(mask).astype(np.uint8)

        imgA /= 255.0
        imgA -= self.mean
        imgA /= self.std

        imgB /= 255.0
        imgB -= self.mean
        imgB /= self.std

        return {'A': imgA, 'B': imgB, 'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        imgA = sample['A']
        imgB = sample['B']
        mask = sample['label']

        # grayscale
        imgA = np.expand_dims(np.array(imgA).astype(np.float32), axis=0)
        imgB = np.expand_dims(np.array(imgB).astype(np.float32), axis=0)

        mask = np.array(mask).astype(np.uint8)

        imgA = torch.from_numpy(imgA).float()
        imgB = torch.from_numpy(imgB).float()
        mask = torch.from_numpy(mask).long()

        return {'A': imgA,'B': imgB, 'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        imgA = sample['A']
        imgB = sample['B']
        mask = sample['label']

        if random.random() < 0.5:
            imgA = imgA.transpose(Image.FLIP_LEFT_RIGHT)
            imgB = imgB.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'A': imgA, 'B': imgB, 'label': mask}

