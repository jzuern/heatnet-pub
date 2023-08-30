#!/usr/bin/python3

import argparse
import itertools

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn


from models.cyclegan.models import Generator
from models.cyclegan.models import Discriminator
# from models import SemanticSegmentation

from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from datasets import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=2, help='batchSize')
parser.add_argument('--dataroot', type=str, default='/home/zuern/datasets/thermal/KAIST/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100,
                    help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--n_classes', type=int, default=12, help='number of classes for semantic segmentation')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

import numpy as np

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




###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)


from models.cyclegan.models import ResNeXt

netSeg = ResNeXt(structure=[3, 4, 6, 3],
                 in_channels=opt.input_nc)


if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  netG_A2B = nn.DataParallel(netG_A2B)
  netG_B2A = nn.DataParallel(netG_B2A)
  netD_A = nn.DataParallel(netD_A)
  netD_B = nn.DataParallel(netD_B)
  netSeg = nn.DataParallel(netSeg)


if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()
    netSeg.cuda()


from torchsummary import summary
summary(netSeg, input_size=(1, 256,256))


netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)
netSeg.apply(weights_init_normal)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
criterion_semseg = torch.nn.CrossEntropyLoss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters(), netSeg.parameters()),
                               lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))


lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                   lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
TensorF = torch.cuda.FloatTensor

target_real = Variable(torch.cuda.FloatTensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(torch.cuda.FloatTensor(opt.batchSize).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()


dataloader = DataLoader(ImageDataset(opt.dataroot), batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

# Loss plot
logger = Logger(opt.n_epochs, len(dataloader))
###################################

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        real_A = batch['A'].cuda()
        real_B = batch['B'].cuda()
        real_A_label = batch['label'].cuda()

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B) * 5.0

        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A) * 5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

        # segmentation loss
        segmented_A = netSeg(real_A.cuda())
        segmented_fake_B = netSeg(fake_B.cuda())

        loss_segmentation_A = criterion_semseg(segmented_A, real_A_label)
        loss_segmentation_fake_B = criterion_semseg(segmented_fake_B, real_A_label)


        # Total loss
        loss_G = loss_identity_A + loss_identity_B + \
                 loss_GAN_A2B + loss_GAN_B2A + \
                 loss_cycle_ABA + loss_cycle_BAB + \
                 loss_segmentation_A + loss_segmentation_fake_B



        loss_G.backward()

        optimizer_G.step()

        ###################################


        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################

        color_coder = ColorCode(256)

        labels, indices = segmented_A.max(1)
        indices = indices[0,:,:]
        segmented_A = color_coder.color_code_labels(indices, argmax=False)
        segmented_A = np.transpose(segmented_A, (2,0,1))
        segmented_A = np.expand_dims(segmented_A, axis=0)
        segmented_A = torch.from_numpy(segmented_A).float()
        #
        labels, indices = segmented_fake_B.max(1)
        indices = indices[0, :, :]
        segmented_fake_B = color_coder.color_code_labels(indices, argmax=False)
        segmented_fake_B = np.transpose(segmented_fake_B, (2,0,1))
        segmented_fake_B = np.expand_dims(segmented_fake_B, axis=0)
        segmented_fake_B = torch.from_numpy(segmented_fake_B).float()

        real_A_label = real_A_label[0, :, :]
        real_A_label = color_coder.color_code_labels(real_A_label, argmax=False)
        real_A_label = np.transpose(real_A_label, (2,0,1))
        real_A_label = np.expand_dims(real_A_label, axis=0)
        real_A_label = torch.from_numpy(real_A_label).float()

        segmented_B = netSeg(real_B)
        labels, indices = segmented_B.max(1)
        indices = indices[0, :, :]
        segmented_B = color_coder.color_code_labels(indices, argmax=False)
        segmented_B = np.transpose(segmented_B, (2, 0, 1))
        segmented_B = np.expand_dims(segmented_B, axis=0)
        segmented_B = torch.from_numpy(segmented_B).float()


        # Progress report (http://localhost:8097)
        logger.log({'loss_G': loss_G,
                    'loss_G_identity': (loss_identity_A + loss_identity_B),
                    'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                    'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB),
                    'loss_D': (loss_D_A + loss_D_B),
                    'loss_segmentation_A': loss_segmentation_A,
                    'loss_segmentation_fake_A': loss_segmentation_fake_B},

                   images={'real_A': real_A, 'real_B': real_B,
                           'fake_A': fake_A, 'fake_B': fake_B,
                           'recovered_A': recovered_A, 'recovered_B': recovered_B,
                           'segmented_A': segmented_A, 'segmented_fake_B': segmented_fake_B,
                           'real_A_label': real_A_label, 'segmented_B': segmented_B}
                   )

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), 'netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), 'netG_B2A.pth')
    torch.save(netD_A.state_dict(), 'netD_A.pth')
    torch.save(netD_B.state_dict(), 'netD_B.pth')
    torch.save(netSeg.state_dict(), 'netSeg.pth')
###################################