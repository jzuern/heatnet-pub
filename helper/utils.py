from functools import partial

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from collections import OrderedDict

class ABN(nn.Sequential):
    """Activated Batch Normalization

    This gathers a `BatchNorm2d` and an activation function in a single module
    """

    def __init__(self, num_features, activation=nn.ReLU(inplace=True), **kwargs):
        """Creates an Activated Batch Normalization module

        Parameters
        ----------
        num_features : int
            Number of feature channels in the input and output.
        activation : nn.Module
            Module used as an activation function.
        kwargs
            All other arguments are forwarded to the `BatchNorm2d` constructor.
        """
        super(ABN, self).__init__(OrderedDict([
            ("bn", nn.BatchNorm2d(num_features, **kwargs)),
            ("act", activation)
        ]))


def _get_norm_act(network_config):
    if network_config["bn_mode"] == "standard":
        if network_config["activation"] == "relu":
            return partial(ABN, activation=nn.ReLU(inplace=True))
        elif network_config["activation"] == "leaky_relu":
            return partial(ABN, activation=nn.LeakyReLU(network_config["leaky_relu_slope"], inplace=True))
        elif network_config["activation"] == "elu":
            return partial(ABN, activation=nn.ELU(inplace=True))
        else:
            print("Standard batch normalization is only compatible with relu, leaky_relu and elu")
            exit(1)
    else:
        print("Unrecognized batch normalization mode", network_config["bn_mode"])
        exit(1)


def get_model_params(network_config):
    """Convert a configuration to actual model parameters

    Parameters
    ----------
    network_config : dict
        Dictionary containing the configuration options for the network.

    Returns
    -------
    model_params : dict
        Dictionary containing the actual parameters to be passed to the `net_*` functions in `models`.
    """
    model_params = {}
    if network_config["input_3x3"] and not network_config["arch"].startswith("wider"):
        model_params["input_3x3"] = True
    model_params["norm_act"] = _get_norm_act(network_config)
    model_params["classes"] = network_config["classes"]
    if not network_config["arch"].startswith("wider"):
        model_params["dilation"] = network_config["dilation"]
    return model_params


def poly_lr_scheduler(optimizer, init_lr, iter, max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    lr = init_lr*(1 - iter/max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def create_optimizer(optimizer_config, model):
    """Creates optimizer and schedule from configuration

    Parameters
    ----------
    optimizer_config : dict
        Dictionary containing the configuration options for the optimizer.
    model : Model
        The network model.

    Returns
    -------
    optimizer : Optimizer
        The optimizer.
    scheduler : LRScheduler
        The learning rate scheduler.
    """
    if optimizer_config["classifier_lr"] != -1:
        # Separate classifier parameters from all others
        net_params = []
        classifier_params = []
        for k, v in model.named_parameters():
            if k.find("fc") != -1:
                classifier_params.append(v)
            else:
                net_params.append(v)
        params = [
            {"params": net_params},
            {"params": classifier_params, "lr": optimizer_config["classifier_lr"]},
        ]
    else:
        params = model.parameters()

    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(params,
                              lr=optimizer_config["learning_rate"],
                              momentum=optimizer_config["momentum"],
                              weight_decay=optimizer_config["weight_decay"],
                              nesterov=optimizer_config["nesterov"])
    elif optimizer_config["type"] == "Adam":
        optimizer = optim.Adam(params,
                               lr=optimizer_config["learning_rate"],
                               weight_decay=optimizer_config["weight_decay"])
    else:
        raise KeyError("unrecognized optimizer {}".format(optimizer_config["type"]))

    if optimizer_config["schedule"]["type"] == "step":
        scheduler = lr_scheduler.StepLR(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "multistep":
        scheduler = lr_scheduler.MultiStepLR(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "exponential":
        scheduler = lr_scheduler.ExponentialLR(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "constant":
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    elif optimizer_config["schedule"]["type"] == "linear":
        def linear_lr(it):
            return it * optimizer_config["schedule"]["params"]["alpha"] + optimizer_config["schedule"]["params"]["beta"]

        scheduler = lr_scheduler.LambdaLR(optimizer, linear_lr)

    return optimizer, scheduler
