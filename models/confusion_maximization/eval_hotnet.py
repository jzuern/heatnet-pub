import torch
import wandb
from models.confusion_maximization.models import conf_segnet
import numpy as np
from validation_bdd_mf import validate_model
import thermal_loader
from glob import glob
import os
from yaml import load
from torch import nn


def getPaths(db_paths):
    ir_files = []
    rgb_files = []
    label_files = []
    for d in db_paths:
        ir_files.extend(list(sorted(glob(os.path.join(d, 'ImagesIR/*_ir.png')))))
        rgb_files.extend(list(sorted(glob(os.path.join(d, 'ImagesRGB/*_rgb.png')))))
        label_files.extend(list(sorted(glob(os.path.join(d, 'SegmentationClass/*.npy')))))

    return ir_files, rgb_files, label_files

# Name UUIDs of runs and their respective names:

run_names = [
    'experiment_1_new',
    'experiment_1_noirscale',
    'experiment_2_new',
    'experiment_3_new',
    'experiment_4_new',
    'experiment_5',
    'experiment_6',
    'experiment_7',
    'experiment_8',
    'experiment_9',
    'experiment_10',
    'experiment_10_b8',
    'experiment_13',
]


def createValloader(data_dirnames):
    dataloader_val = thermal_loader.ThermalTestDataLoader(*getPaths(data_dirnames))

    val_loader = torch.utils.data.DataLoader(dataloader_val,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=1,
                                               pin_memory=True,
                                               drop_last=False)
    return val_loader

def initModelPartial(model, weights_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(weights_path, map_location=lambda storage, loc: storage)['state_dict']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)



for run_name in run_names:

    print('Evaluating Run {}!'.format(run_name))

    # initialize wandb project
    wandb.init(project="hotnet-eval",
               entity='team-awesome',
               name=run_name,
               reinit=True)

    # Get wandb config yaml file and model file
    model_weights_file = 'past_runs/{}/model_best.pth.tar'.format(run_name)
    yaml_config = 'past_runs/{}/config.yaml'.format(run_name)

    yaml_config = load(open(yaml_config))
    opt = type('D', (object,), yaml_config)()

    conf_segnet_model = conf_segnet.conv_segnet(pretrained=opt.pretraining['value'],
                                                disc_arch=opt.discarch['value'],
                                                num_critics=opt.num_critics['value'],
                                                feedback_seg=opt.feedback_seg['value'],
                                                no_conf=opt.no_conf['value'],
                                                modalities=opt.modalities['value'],
                                                input_adapter=opt.train_input_adapter['value'],
                                                cert_branch=opt.cert_branch['value'],
                                                arch=opt.arch['value'],
                                                late_fusion=opt.late_fusion['value'])

    # load checkpoint
    conf_segnet_model.cuda()
    conf_segnet_model = nn.DataParallel(conf_segnet_model.cuda(), [0])
    checkpoint = torch.load(model_weights_file, map_location=lambda storage, loc: storage)
    conf_segnet_model.load_state_dict(checkpoint['state_dict'])

    # create validation loader
    testroot_day = '/data/zuern/datasets/thermal_seg/test_set_day/converted'
    testroot_night = '/data/zuern/datasets/thermal_seg/test_set_night/converted'

    val_loader_night = createValloader([testroot_night])
    val_loader_day = createValloader([testroot_day])

    ious_night = validate_model(conf_segnet_model.module.trgb_segnet, val_loader_night, opt.modalities['value'], mode="night")
    ious_day = validate_model(conf_segnet_model.module.trgb_segnet, val_loader_day, opt.modalities['value'], mode="day")

    ious_combined = (ious_day + ious_night) / 2
    iou_combined_mean = np.mean(ious_combined)

    # log combined ious
    wandb.log({
        "combined_Test mean IoU": iou_combined_mean,
        "combined_Test IoU road,parking": ious_combined[0],
        "combined_Test IoU ground,sidewalk": ious_combined[1],
        "combined_Test IoU building,": ious_combined[2],
        'combined_Test IoU curb': ious_combined[3],
        'combined_Test IoU fence': ious_combined[4],
        'combined_Test IoU pole,traffic light,traffic sign': ious_combined[5],
        'combined_Test IoU vegetation': ious_combined[6],
        'combined_Test IoU terrain': ious_combined[7],
        'combined_Test IoU sky': ious_combined[8],
        'combined_Test IoU person,rider': ious_combined[9],
        'combined_Test IoU car,truck,bus,train': ious_combined[10],
        'combined_Test IoU motorcycle,bicycle': ious_combined[11],
    })
