import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import custom_transforms as tr


def transform(sample):
    composed_transforms = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.Normalize((0.5,), (0.5,)),
        tr.ToTensor()])

    return composed_transforms(sample)


class ImageDataset(Dataset):

    def __init__(self, root):

        self.files_A = sorted(glob.glob(os.path.join(root, 'Day/set00/V*/lwir') + '/I*.jpg'))
        self.files_A += sorted(glob.glob(os.path.join(root, 'Day/set01/V*/lwir') + '/I*.jpg'))

        self.files_RGB = sorted(glob.glob(os.path.join(root, 'Day/set00/V*/visible') + '/I*.jpg'))
        self.files_RGB += sorted(glob.glob(os.path.join(root, 'Day/set01/V*/visible') + '/I*.jpg'))

        self.files_Label = sorted(glob.glob(os.path.join(root, 'Day/set00/V*/labels') + '/I*.png'))
        self.files_Label += sorted(glob.glob(os.path.join(root, 'Day/set01/V*/labels') + '/I*.png'))

        self.files_B = sorted(glob.glob(os.path.join(root, 'Night/set04/V*/lwir') + '/I*.jpg'))
        self.files_B += sorted(glob.glob(os.path.join(root, 'Night/set05/V*/lwir') + '/I*.jpg'))
        self.files_B += sorted(glob.glob(os.path.join(root, 'Night/set09/V*/lwir') + '/I*.jpg'))
        self.files_B += sorted(glob.glob(os.path.join(root, 'Night/set10/V*/lwir') + '/I*.jpg'))
        self.files_B += sorted(glob.glob(os.path.join(root, 'Night/set11/V*/lwir') + '/I*.jpg'))


        print(len(self.files_A))
        print(len(self.files_RGB))
        print(len(self.files_Label))
        print(len(self.files_B))

    def __getitem__(self, index):

        pil_img_A = Image.open(self.files_A[index % len(self.files_A)])
        pil_img_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]) # random B
        pil_img_label = Image.open(self.files_Label[index % len(self.files_Label)])

        # convert to grayscale
        pil_img_A = pil_img_A.convert('L')
        pil_img_B = pil_img_B.convert('L')

        pil_img_A = pil_img_A.resize((256, 256), Image.BICUBIC)
        pil_img_B = pil_img_B.resize((256, 256), Image.BICUBIC)
        pil_img_label = pil_img_label.resize((256, 256), Image.NEAREST)


        sample = {'A': pil_img_A, 'B': pil_img_B, 'label': pil_img_label}

        transformed = transform(sample)

        return transformed

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))