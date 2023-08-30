import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
import torch.nn as nn
# from visdom import Visdom
import numpy as np


class robust_loss(nn.Module):
    def __init__(self, a = 0.5, c = 1.0):
        super(robust_loss, self).__init__()
        self.a = a
        self.c = c
        self.e = 1e-05

    def forward(self, loss):
        b = abs(2. - self.a) + self.e
        d = self.a + self.e if self.a >= 0. else self.a - self.e
        out = (b/d) * (torch.pow(torch.pow(loss / self.c, 2)/b + 1., 0.5 * d) - 1.0)

        # out = (abs(2.0 - self.a)/ self.a) * (torch.pow((torch.pow(loss / self.c, 2) / abs(2.0 - self.a)) + 1.0, (self.a / 2.0))-1.0)
        return out


def make_one_hot(labels, C=2):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)

    target = Variable(target)

    return target

def tensor2image(tensor):
    image = 127.5 * (tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    return image.astype(np.uint8)

def initModelRenamed( model, weights_path, to_rename, rename):
        saved_model = torch.load(weights_path, map_location=lambda storage, loc: storage)
        if 'state_dict' in saved_model.keys():
            saved_model = saved_model['state_dict']
        # print(saved_model.keys())
        # print('-----------------------------------------')
        model_dict = model.state_dict()
        # print(model_dict.keys())

        weights_changed = {}
        for k, v in saved_model.items():
            k = k.replace(to_rename, rename)
            weights_changed[k] = v

        weights_changed = {k: v for k, v in weights_changed.items() if k in model_dict}

        print("Loaded dict with %d entries..." % len(weights_changed))
        assert (len(weights_changed)>0)
        model_dict.update(weights_changed)
        model.load_state_dict(model_dict)

def initModelPartial(model, weights_path):
        model_dict = model.state_dict()
        pretrained_dict = torch.load(weights_path, map_location=lambda storage, loc: storage)['state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print('Updated : %d entries (initModelPartial)' % pretrained_dict.__len__())
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

def initModelFull(model, weights_path):
        pretrained_dict = torch.load(weights_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(pretrained_dict)

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

def calculate_ious(pred, target, n_classes=13):

    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    background_class_idx = 12
    ignore_class_idx = 13

    for cls in range(0, n_classes):

        if cls == background_class_idx:  # Ignore _background_ class
            continue
        if cls == ignore_class_idx:  # Ignore _ignore_ class
            continue

        pred_inds = pred == cls
        target_inds = target == cls

        intersection = (pred_inds[target_inds]).long().sum().data.cpu()
        union = pred_inds[target != ignore_class_idx].long().sum().data.cpu() + \
                target_inds[target != ignore_class_idx].long().sum().data.cpu() - \
                intersection

        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))

    return np.array(ious)
