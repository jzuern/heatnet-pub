#!/usr/bin/python3

import argparse

import torch
import torch.nn as nn
import wandb

import discriminator_model
import thermal_loader
import utils
from models.confusion_maximization.models import trgb_segnet as models
from utils import LambdaLR
from utils import weights_init_normal

class AverageMeter(object):
    '''Computes and stores the average and current value'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


wandb.init(project="HotNetConf")

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=2, help='batchSize')
parser.add_argument('--dataroot', type=str, default='/mnt/shared/ir_rgb_data/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100,
                    help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--batch_size', type=int, default=10, help='batch size in training')

opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Networks
trgb_segnet = models.__dict__["net_resnext50"]()
trgb_conf_discriminator = discriminator_model.Discriminator(512 * 2)

gpus = [2,3]
torch.cuda.set_device(gpus[0])
if opt.cuda:
    trgb_segnet.cuda()
    trgb_conf_discriminator.cuda()

trgb_segnet = nn.DataParallel(trgb_segnet.cuda(), gpus)
trgb_conf_discriminator = nn.DataParallel(trgb_conf_discriminator, gpus)


trgb_segnet.apply(weights_init_normal)
trgb_conf_discriminator.apply(weights_init_normal)

utils.initModelPartial(trgb_segnet, '../segmentation/last_12.pth.tar')

# Losses
criterion_conf = torch.nn.L1Loss()
criterion_semseg = torch.nn.CrossEntropyLoss()

# Optimizers & LR schedulers
optimizer = torch.optim.Adam(trgb_segnet.parameters(),
                               lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                 lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

dataloader_train = thermal_loader.ThermalDataLoader(opt.dataroot, split='train')

train_loader = torch.utils.data.DataLoader(dataloader_train,
                                                   batch_size=opt.batch_size,
                                                   shuffle=True,
                                                   num_workers=opt.n_cpu,
                                                   pin_memory=True,
                                                   drop_last=True)
# Loss plot
loss_avgmeter = AverageMeter()
###################################

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(train_loader):
        rgb_day = batch['rgb_day'].cuda()
        ir_day = batch['ir_day'].cuda()
        rgb_night = batch['rgb_night'].cuda()
        ir_night = batch['ir_night'].cuda()
        label_day = batch['label_day'].cuda().long()

        optimizer.zero_grad()

        in_day = torch.cat([rgb_day, ir_day], dim=1)
        in_night = torch.cat([rgb_night, ir_night], dim=1)
        pred_label_day, bot_feat_day= trgb_segnet(in_day)

        seg_loss = criterion_semseg(pred_label_day, label_day)

        seg_loss.backward()
        optimizer.step()
        loss_avgmeter.update(seg_loss.item())
        print("Current loss: %f " % loss_avgmeter.avg)
        wandb.log({'epoch': epoch, 'loss': loss_avgmeter.avg})

        # vis_utils.visDepth(ir_day, 'ir_day')
        # vis_utils.visImage3Chan(rgb_day, 'rgb_day')
        # vis_utils.visDepth(pred_label_day.max(1)[1].unsqueeze(0).float(), 'pred_label_day')
        # vis_utils.visDepth(label_day.unsqueeze(0).float(), 'label_day')
        # cv2.waitKey(10)

    # Update learning rates
    lr_scheduler.step()

    # Save models checkpoints
    torch.save(trgb_segnet.state_dict(), 'trgb_segnet.pth')
###################################