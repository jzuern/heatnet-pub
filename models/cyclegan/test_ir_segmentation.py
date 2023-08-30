#!/usr/bin/python3

import argparse
import sys
import os

import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='/home/zuern/datasets/thermal/KAIST/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--segmentation_ir', type=str, default='netSeg.pth', help='Segmentation network checkpoint file')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

import matplotlib.pyplot as plt



class ColorCode():

    def get_coding_1(self):
        coding = {}
        coding[0] = [100, 50, 200]
        coding[1] = [200, 100, 50]
        coding[2] = [0, 0, 255]
        coding[3] = [255, 220, 200]
        coding[4] = [0, 255, 0]
        coding[5] = [0, 255, 255]
        coding[6] = [255, 255, 0]
        coding[7] = [20, 180, 150]
        coding[8] = [200, 50, 255]
        coding[9] = [80, 10, 100]
        coding[10] = [20, 150, 220]
        coding[11] = [230, 120, 10]
        return coding

    def color_code_labels(self, net_out, argmax=True):
        if argmax:
            labels, indices = net_out.max(1)
            labels_cv = indices.cpu().numpy().squeeze()
        else:
            labels_cv = net_out.cpu().numpy().squeeze()

        h = labels_cv.shape[0]
        w = labels_cv.shape[1]

        color_coded = np.zeros((h, w, 3), dtype=np.uint8)

        for x in range(w):
            for y in range(h):
                color_coded[y, x, :] = self.color_coding[labels_cv[y, x]]

        return color_coded / 255.

    def __init__(self, max_classes):
        super(ColorCode, self).__init__()
        self.color_coding = self.get_coding_1()

color_coder = ColorCode(256)



###### Definition of variables ######
# Networks
import torch.nn as nn
from models.cyclegan.models import ResNeXt

netSeg = ResNeXt(structure=[3, 4, 6, 3], in_channels=opt.input_nc)
netSeg = nn.DataParallel(netSeg)

if opt.cuda:
    netSeg.cuda()

# Load state dicts
netSeg.load_state_dict(torch.load('/home/zuern/thermal_seg/cyclegan/netSeg.pth'))

# Set model's test mode
netSeg.eval()

# Inputs & targets memory allocation
input_B = torch.cuda.FloatTensor(opt.batchSize, 1, opt.size, opt.size)



import glob
images = sorted(glob.glob('/home/zuern/datasets/thermal/KAIST/Night/set10/V000/lwir/*.jpg'))
images = images[::5]

###################################

###### Testing######

# Create output dirs if they don't exist
if not os.path.exists('output/ir'):
    os.makedirs('output/ir')

from PIL import Image

f, axarr = plt.subplots(1,2)

import custom_transforms as tr

def transform(sample):
    composed_transforms = transforms.Compose([
        tr.Normalize((0.5,), (0.5,)),
        tr.ToTensor()])

    return composed_transforms(sample)


for i, image in enumerate(images):


    # Set model input
    image_pil = Image.open(image).convert('L')
    image_pil = image_pil.resize((256, 256), Image.BICUBIC)
    image = np.asarray(image_pil)
    image = image / 255.
    image = image - 0.5
    image = image / 0.5


    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image).float().cuda()

    image = Variable(input_B.copy_(image))

    # Generate output
    prediction = netSeg(image)

    labels, indices = prediction.max(1)
    indices = indices[0, :, :]

    segmented_B_np = color_coder.color_code_labels(indices, argmax=False)



    axarr[0].imshow(image_pil, cmap='gray')
    axarr[1].imshow(segmented_B_np)

    plt.savefig('output/ir/%04d.png' % (i+1))


    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(images)))

sys.stdout.write('\n')
###################################