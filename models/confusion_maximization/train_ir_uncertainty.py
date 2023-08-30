#!/usr/bin/python3

import argparse
import itertools
import numpy as np
import torch
import torch.nn as nn
import wandb
import thermal_loader
from utils import LambdaLR, weights_init_normal
from models.segnetsplit import ResNeXtEncoder, ResNeXtDecoder


class ColorCode():

    def get_coding_1(self):
        coding = {}
        coding[-1] = [0,0,0]
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


wandb.init(project="HotNet")

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='batchSize')
parser.add_argument('--dataroot', type=str, default='/home/zuern/mnt/johan_shared/ir_rgb_data/')
parser.add_argument('--test-dataset-path', type=str, default='/home/zuern/mnt/johan_shared/ir_rgb_data/test_data_final/')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
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

color_coder = ColorCode(256)


# Networks
structure = [2, 2, 2, 2]

ir_encoder1 = ResNeXtEncoder(structure=structure,
                             in_channels=1,
                             classes=12)

ir_encoder2 = ResNeXtEncoder(structure=structure,
                             in_channels=1,
                             classes=1)

segmentation_decoder = ResNeXtDecoder(out_classes=12)
uncertainty_decoder = ResNeXtDecoder(out_classes=1,
                                     last_activation=torch.nn.Sigmoid())


if opt.cuda:
    uncertainty_decoder.cuda()
    segmentation_decoder.cuda()
    ir_encoder1.cuda()
    ir_encoder2.cuda()

uncertainty_decoder = nn.DataParallel(uncertainty_decoder.cuda())
segmentation_decoder = nn.DataParallel(segmentation_decoder.cuda())
ir_enoder1 = nn.DataParallel(ir_encoder1.cuda())
ir_enoder2 = nn.DataParallel(ir_encoder2.cuda())


ir_enoder1.apply(weights_init_normal)
ir_enoder2.apply(weights_init_normal)
segmentation_decoder.apply(weights_init_normal)
uncertainty_decoder.apply(weights_init_normal)


# Losses
criterion_uncertainty = torch.nn.MSELoss()
criterion_conf = torch.nn.L1Loss()
criterion_semseg = torch.nn.CrossEntropyLoss()

# Optimizers & LR schedulers
optimizer = torch.optim.Adam(itertools.chain(ir_enoder1.parameters(), ir_enoder2.parameters(),
                                             segmentation_decoder.parameters(),
                                             uncertainty_decoder.parameters()),
                               lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

dataloader_train = thermal_loader.ThermalDataLoader(opt.dataroot, split='train')

train_loader = torch.utils.data.DataLoader(dataloader_train,
                                                   batch_size=opt.batch_size,
                                                   shuffle=True,
                                                   num_workers=opt.n_cpu,
                                                   pin_memory=True,
                                                   drop_last=True)
# Loss plot
seg_loss_avgmeter = AverageMeter()
uncertainty_loss_avgmeter = AverageMeter()
###################################

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(train_loader):

        ir_day = batch['ir_day'].cuda()
        label_day = batch['label_day'].cuda().long()

        ir_day = ir_day[:, :, ::4, ::4]
        label_day = label_day[:, ::4, ::4]

        optimizer.zero_grad()

        encoded_day1, out_4_1 = ir_enoder1(ir_day)
        encoded_day2, out_4_2 = ir_enoder2(ir_day)

        predicted_label_day = segmentation_decoder(encoded_day1, out_4_1)
        predicted_uncertainty = uncertainty_decoder(encoded_day2, out_4_2)

        seg_loss = criterion_semseg(predicted_label_day, label_day)

        uncertainty_gt = torch.nn.CrossEntropyLoss(reduce=False)(predicted_label_day, label_day)

        print(uncertainty_gt.shape)
        uncertainty_loss = criterion_uncertainty(predicted_uncertainty, uncertainty_gt)

        seg_loss.backward(retain_graph=True)
        uncertainty_loss.backward()


        optimizer.step()
        seg_loss_avgmeter.update(seg_loss.item())
        uncertainty_loss_avgmeter.update(uncertainty_loss.item())

        print("Segmentation loss: %f " % seg_loss_avgmeter.avg, ", Uncertainty Loss: %f" % uncertainty_loss_avgmeter.avg)

        wandb.log({'epoch': epoch,
                   'Uncertainty Loss': uncertainty_loss_avgmeter.avg,
                   'Segmentation Loss': seg_loss_avgmeter.avg})

        if i % 300 == 0:

            with torch.no_grad():
                labels, indices = predicted_label_day.max(1)

                indices = indices[0, :, :]
                segmented_day = color_coder.color_code_labels(indices, argmax=False)
                segmented_day = np.transpose(segmented_day, (2, 0, 1))
                segmented_day = np.expand_dims(segmented_day, axis=0)
                segmented_day = torch.from_numpy(segmented_day).float() / 255.

                day_label = label_day[0, :, :]
                day_label = color_coder.color_code_labels(day_label, argmax=False)
                day_label = np.transpose(day_label, (2, 0, 1))
                day_label = np.expand_dims(day_label, axis=0)
                day_label = torch.from_numpy(day_label).float() / 255.

                wandb.log({"examples": [wandb.Image(segmented_day, caption="Segmentation Prediction"),
                                        wandb.Image(day_label, caption="GT Label"),
                                        wandb.Image(predicted_uncertainty, caption="Uncertainty Prediction")]})


    # Update learning rates
    lr_scheduler.step()


###################################
