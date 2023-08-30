import cv2
import torch
import numpy as np
from helper import config
from models import build_net
from models.confusion_maximization import vis_utils, thermal_loader
import os
from glob import glob
from torch.autograd import Variable
from models.confusion_maximization.utils import calculate_ious

vistas_stats = {'mean': [0.28389175985075144, 0.32513300997108185, 0.28689552631651594],
                     'std': [0.1777223070810445, 0.18099167120139084, 0.17613640748441522]}

def getPaths(db_paths):
    ir_files = []
    rgb_files = []
    label_files = []
    for d in db_paths:
        ir_files.extend(list(sorted(glob(os.path.join(d, 'ImagesIR/*_ir.png')))))
        rgb_files.extend(list(sorted(glob(os.path.join(d, 'ImagesRGB/*_rgb.png')))))
        label_files.extend(list(sorted(glob(os.path.join(d, 'SegmentationClass/*.npy')))))
    return ir_files, rgb_files, label_files


def createValloader(data_dirnames):
    dataloader_val = thermal_loader.ThermalTestDataLoader(*getPaths(data_dirnames), normalize=True, db_stats=vistas_stats)

    val_loader = torch.utils.data.DataLoader(dataloader_val,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=2,
                                               pin_memory=True,
                                               drop_last=False)
    return val_loader

def validate_model_freiburg(model, val_loader, save_dir=""):

    print('Evaluating...')

    model.eval()

    preds = Variable(torch.zeros(len(val_loader), 320, 704))
    gts = Variable(torch.zeros(len(val_loader), 320, 704))
    color_coder = vis_utils.ColorCode(13)

    for i, batch in enumerate(val_loader):
        # if i < 26:
        #     continue
        print('Validating ... %d of %d ...' % (i, len(val_loader)))
        rgb_im = batch['rgb'].cuda()
        ir_im = batch['ir'].cuda()
        label = batch['label'].cuda()
        label = label.to(torch.long)

        with torch.no_grad():
            segmented = model(rgb_im)

        segmented_argmax = torch.argmax(segmented.cpu(), 1).squeeze()

        # vis_utils.visImage3Chan(batch['rgb_org'][0:1, ...], 'rgb_image')
        # vis_utils.visDepth(ir_im[0:1, ...], 'ir_image')
        # color_gt = color_coder.color_code_labels(label[0:1, ...], False)
        # color_pred = color_coder.color_code_labels(segmented_argmax, False)
        # cv2.imshow('GT', color_gt)
        # cv2.imshow('PRED', color_pred)
        # cv2.waitKey()

        if save_dir is not "":
            pred_color = color_coder.color_code_labels(segmented)
            gt_color = color_coder.color_code_labels(label, argmax=False)
            rgb_image = np.transpose(batch['rgb_org'].cpu().data.numpy().squeeze(), (1, 2, 0))

            ir_cv = batch['ir_org'].clamp(0.2, 0.7).cpu().data.numpy().squeeze()
            cv2.normalize(ir_cv, ir_cv, 0, 255, cv2.NORM_MINMAX)
            ir_cv = cv2.applyColorMap(ir_cv.astype(np.uint8), cv2.COLORMAP_JET)

            cv2.imwrite(save_dir + '/pred_' + str(i) + ".png", (pred_color*255).astype(np.uint8))
            cv2.imwrite(save_dir + '/gt_' + str(i) + ".png", (gt_color*255).astype(np.uint8))
            cv2.imwrite(save_dir + '/rgb_' + str(i) + ".png",
                        cv2.cvtColor((rgb_image * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
            cv2.imwrite(save_dir + '/ir_' + str(i) + ".png", ir_cv)

        # account for offset in GT labels due to _background_ classes
        gts[i, :, :] = label.squeeze()
        preds[i, :, :] = segmented_argmax.squeeze()

    acc = calculate_ious(preds, gts)

    mean_iou = np.nanmean(acc)

    res = {
       "_Test mean IoU": np.nanmean(acc),
       "_Test IoU road,parking":                    acc[0],
       "_Test IoU ground,sidewalk":                 acc[1],
       "_Test IoU building,":                       acc[2],
       '_Test IoU curb':                            acc[3],
       '_Test IoU fence':                           acc[4],
       '_Test IoU pole,traffic light,traffic sign': acc[5],
       '_Test IoU vegetation':                      acc[6],
       '_Test IoU terrain':                         acc[7],
       '_Test IoU sky':                             acc[8],
       '_Test IoU person,rider':                    acc[9],
       '_Test IoU car,truck,bus,train':             acc[10],
       '_Test IoU motorcycle,bicycle':              acc[11]
    }
    print(res)
    # model.train()
    return mean_iou


def validate_on_freiburg(model):
    testroot_night = '/mnt/hpc.shared/label_data/test_set_night/converted/'
    testroot_day = '/mnt/hpc.shared/label_data/test_set_day/converted/'
    testroot_night_fence = '/mnt/hpc.shared/label_data/fence_data/converted/'

    val_loader_night = createValloader([testroot_night, testroot_night_fence])
    val_loader_day = createValloader([testroot_day])
    val_loader_combined = createValloader([testroot_night, testroot_night_fence, testroot_day])

    # Evaluate day images
    save_dir = "/home/vertensj/Documents/papers/iros_hotnet/qual_comp/vistas_model/day/"
    iou_day = validate_model_freiburg(model, val_loader_day, save_dir=save_dir)

    # Evaluate night images
    save_dir = "/home/vertensj/Documents/papers/iros_hotnet/qual_comp/vistas_model/night/"
    iou_night = validate_model_freiburg(model, val_loader_night, save_dir=save_dir)

    # Evaluate combined images
    iou = validate_model_freiburg(model, val_loader_combined)

    print('Total mean IoU night: %f , day: %f, combined %f' % (iou_night, iou_day, iou))

scale_factor = 4
color_coder = vis_utils.ColorCode(256)

conf = config.load_config("../experiments/heatnet_conf.json")

# model_params = utils.get_model_params(conf["network"])
# model = models.segnet.__dict__["net_" + conf["network"]["arch"]](**model_params)
model, starting_epoch = build_net.build_network(None, 'resnet50')

model = torch.nn.DataParallel(model, [0]).cuda()

pretrained_dict = torch.load("/mnt/hpc.vertensj/software/thermal_seg/segmentation/model_best.pth.tar", map_location=lambda storage, loc: storage)['state_dict']
model.load_state_dict(pretrained_dict)

validate_on_freiburg(model)

model.eval()
np.random.seed(1203412412)

