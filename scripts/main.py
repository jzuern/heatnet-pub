import argparse
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
from random import randint
from torch.optim import Adam
import wandb

import models
from data import vistas_dataset
from models import build_net, segnet
import iou_eval
from helper import utils, config


def initModelPartial(model, weights_path):

    '''
    Initializes the model partially with the weights from the weights file
    :param model: model to be initialized
    :param weights_path: path to the weights file
    '''

    model_dict = model.state_dict()
    pretrained_dict = torch.load(weights_path, map_location=lambda storage, loc: storage)['state_dict']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

class ColorCode():
    """
    Class for color coding of labels
    """

    def random_color_coding(self, max_label):
        coding = {}
        for i in range(max_label + 1):
            coding[i] = [randint(0, 255), randint(0, 255), randint(0, 255)]
        return coding

    def color_code_labels(self, net_out, argmax=True):
        if argmax:
            labels, indices = net_out.max(1)
            labels_cv = indices.cpu().data.numpy().squeeze()
        else:
            labels_cv = net_out.cpu().data.numpy().squeeze()

        h = labels_cv.shape[0]
        w = labels_cv.shape[1]

        color_coded = np.zeros((h, w, 3), dtype=np.uint8)

        for x in range(w):
            for y in range(h):
                color_coded[y, x, :] = self.color_coding[labels_cv[y, x]]

        return color_coded

    def __init__(self, max_classes):
        super(ColorCode, self).__init__()
        self.color_coding = self.random_color_coding(max_classes)

parser = argparse.ArgumentParser(description='HeatNet Training')
parser.add_argument('--data', default='/data/', type=str,  help='root directory of the training dataset')
parser.add_argument('--valdata', default='/data/', type=str,  help='root directory of the validation dataset')
parser.add_argument('--arch', default='custom', type=str,  help='type of architecture of the network')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')

best_iou = 0
args = None
conf = None

color_coder = ColorCode(23)

def main():
    global args, best_iou, conf
    args = parser.parse_args()
    wandb.init(project="segnet")
    wandb.config.update(args)

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # Create model
    conf = config.load_config("../experiments/heatnet_conf.json")
    if args.arch == 'custom':
        model_params = utils.get_model_params(conf["network"])
        model = models.segnet.__dict__["net_" + conf["network"]["arch"]](**model_params)
    elif args.arch == 'pspnet':
        model, starting_epoch = build_net.build_network(None, 'resnet50')

    torch.cuda.set_device(0)

    if not args.distributed:
        model = torch.nn.DataParallel(model.cuda(), [0, 1, 2, 3])
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    np.random.seed(1203412412)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)

            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}

            args.start_epoch = 0
            # args.start_epoch = checkpoint['epoch']
            # best_prec1 = checkpoint['best_prec1']
            # model.load_state_dict(checkpoint['state_dict'])
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # print("=> loaded checkpoint '{}' (epoch {})"
            #       .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        init_weights(model)
        args.start_epoch = 0

    cudnn.benchmark = True

    # Data loading code

    height = 384
    width = 768

    optimizer = Adam(model.parameters(), lr=conf["optimizer"]["learning_rate"])

    train_dataset = vistas_dataset.VistasBorderDataLoader(
        args.data, width, height, contrast_enhancement=False, background_id=0)

    val_dataset = vistas_dataset.VistasBorderDataLoader(
        args.valdata, width, height, contrast_enhancement=False, background_id=0, augment_data=False)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=conf["optimizer"]["batch_size"], shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    train_dataset_length = train_dataset.__len__()

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        sampler=None)

    if args.evaluate:
        validate(val_loader, model)
        return

    num_epochs = conf["optimizer"]["schedule"]["epochs"]
    for epoch in range(args.start_epoch, num_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, optimizer, epoch, train_dataset_length, num_epochs)

        # evaluate on validation set
        iou = validate(val_loader, model)

        #remember best iou and save checkpoint
        if (epoch % 2) ==0:
            is_best = iou > best_iou
            best_iou = max(iou, best_iou)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': conf["network"]["arch"],
                'state_dict': model.state_dict(),
                'best_iou': best_iou,
                'optimizer': optimizer.state_dict(),
            }, is_best)

def train(train_loader, model, optimizer, epoch, train_dataset_length, num_epochs):
    global conf
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=13)
    for i, (image, ids, org_img) in enumerate(train_loader):

        # Debugging
        # seg_color_gt = color_coder.color_code_labels(ids[0:1, ...], argmax=False)
        # cv2.imshow("Seg GT", seg_color_gt)
        # vis_utils.visImage3Chan(image[0:1, ...], 'image')
        # cv2.waitKey()

        new_lr = utils.poly_lr_scheduler(optimizer, conf["optimizer"]["learning_rate"],
                                         epoch * train_dataset_length + i, num_epochs * train_dataset_length, power=0.9)

        # measure data loading time
        data_time.update(time.time() - end)

        ids = ids.cuda()
        image = image.cuda()

        image_var = image
        ids_var = ids.squeeze(1).long()

        seg = model(image_var)

        #        loss = F.nll_loss(seg, ids_var, size_average=True, ignore_index=13)
        loss = criterion(seg, ids_var)
        print('Seg loss: %f', loss.item())

        # measure accuracy and record loss
        losses.update(loss.item(), image.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if conf["optimizer"]["clip"] != 0.:
            nn.utils.clip_grad_norm(model.parameters(), conf["optimizer"]["clip"])
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
        wandb.log(
            {'epoch': epoch, 'loss': loss, 'lr': new_lr})


def validate(val_loader, model):
    batch_time = AverageMeter()

    metric = iou_eval.IoU(14, False, [12, 13])

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (image, ids, org_img) in enumerate(val_loader):
        ids = ids.cuda()
        image = image.cuda()

        image_var = image
        ids_var = ids.squeeze(1).long()

        # compute output
        seg = model(image_var)
        metric.add(seg, ids_var)

        # seg_color = color_coder.color_code_labels(seg)
        # seg_color_gt = color_coder.color_code_labels(ids_var, argmax=False)
        # cv2.imshow("Seg", seg_color)
        # cv2.imshow("Seg GT", seg_color_gt)
        # cv2.waitKey()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    iou_scores, iou_mean = metric.value()

    wandb.log(
        {'iou_mean': iou_mean})

    print('IOU:')
    print(iou_mean)
    print(iou_scores)

    for i, val in enumerate(iou_scores):
        wandb.log(
            {'iou_class_' + str(i): val})

    return iou_mean


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

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


def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()


def init_weights(model):
    global conf
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            init_fn = getattr(nn.init, conf["network"]["weight_init"])
            if conf["network"]["weight_init"].startswith("xavier") or conf["network"]["weight_init"] == "orthogonal":
                gain = conf["network"]["weight_gain_multiplier"]
                if conf["network"]["activation"] == "relu" or conf["network"]["activation"] == "elu":
                    gain *= nn.init.calculate_gain("relu")
                elif conf["network"]["activation"] == "leaky_relu":
                    gain *= nn.init.calculate_gain("leaky_relu", conf["network"]["leaky_relu_slope"])
                init_fn(m.weight, gain)
            elif conf["network"]["weight_init"].startswith("kaiming"):
                if conf["network"]["activation"] == "relu" or conf["network"]["activation"] == "elu":
                    init_fn(m.weight, 0)
                else:
                    init_fn(m.weight, conf["network"]["leaky_relu_slope"])

            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant(m.bias, 0.)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight, .1)
            nn.init.constant(m.bias, 0.)
        elif isinstance(m, nn.ConvTranspose2d):
            c1, c2, h, w = m.weight.data.size()
            weight = get_upsample_filter(h).cuda()
            m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)

if __name__ == '__main__':
    main()
