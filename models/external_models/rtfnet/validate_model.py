import torch
import numpy as np
from torch.autograd import Variable
from models.confusion_maximization.utils import calculate_ious
from glob import glob
import os
from models.confusion_maximization import vis_utils, thermal_loader

class_mfnet_to_ours= {
    4: 3,  # curb
    2: 9,  # person
    1: 10,  # car,truck,bus,train
    3: 11,  # bicycle
}

def validate_model_mfnet(model, val_loader):

    print('Evaluating MFNet dataset')

    # Containers for results
    preds = Variable(torch.zeros(len(val_loader), 480, 640))
    gts = Variable(torch.zeros(len(val_loader), 480, 640))

    for i, (images, labels, names) in enumerate(val_loader):

        print('Validating ... %d of %d ...' % (i, len(val_loader)))

        images = images.cuda()
        label = labels.cuda()

        with torch.no_grad():
            segmented= model(images)

        segmented = segmented[0:1, ...]

        segmented_argmax = torch.argmax(segmented.cpu(), 1).squeeze()

        gts[i, :, :] = label.squeeze()
        preds[i, :, :] = segmented_argmax

    acc = calculate_ious(preds, gts)


    result = {
        "_MFNET_Test mean IoU": np.nanmean(acc),
        '_MFNET_Test IoU curb': acc[4],
        '_MFNET_Test IoU person,rider': acc[2],
        '_MFNET_Test IoU car,truck,bus,train': acc[1],
        '_MFNET_Test IoU bicycle': acc[3],
    }

    print(result)

    return np.nanmean(acc)

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
    dataloader_val = thermal_loader.ThermalTestDataLoader(*getPaths(data_dirnames), normalize=False)

    val_loader = torch.utils.data.DataLoader(dataloader_val,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=2,
                                               pin_memory=True,
                                               drop_last=False)
    return val_loader



def validate_model_freiburg(model, val_loader, modalities):

    print('Evaluating...')

    # model.eval()
    # Containers for results

    preds = Variable(torch.zeros(len(val_loader), 320, 704))
    gts = Variable(torch.zeros(len(val_loader), 320, 704))
    color_coder = vis_utils.ColorCode(13)

    for i, batch in enumerate(val_loader):
        print('Validating ... %d of %d ...' % (i, len(val_loader)))
        rgb_im = batch['rgb'].cuda()
        ir_im = batch['ir'].cuda()
        label = batch['label'].cuda()
        label = label.to(torch.long)

        with torch.no_grad():
            segmented = model(torch.cat([rgb_im, ir_im], dim=1))

        segmented_argmax = torch.argmax(segmented.cpu(), 1).squeeze()
        segmented = torch.full_like(segmented_argmax, 13)

        for our_label, mf_label in class_mfnet_to_ours.items():
            segmented[segmented_argmax == our_label] = mf_label

        # vis_utils.visImage3Chan(rgb_im[0:1, 0:3, ...], 'rgb_image')
        # vis_utils.visDepth(ir_im[0:1, ...], 'ir_image')
        # color_gt = color_coder.color_code_labels(label[0:1, ...], False)
        # color_pred = color_coder.color_code_labels(segmented_argmax, False)
        # cv2.imshow('GT', color_gt)
        # cv2.imshow('PRED', color_pred)
        # cv2.waitKey()

        # account for offset in GT labels due to _background_ classes
        gts[i, :, :] = label.squeeze()
        preds[i, :, :] = segmented.squeeze()


    acc = calculate_ious(preds, gts)

    mean_iou = np.nanmean([acc[9],acc[10], acc[11]])

    res = {
        "_Test mean IoU": mean_iou,
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
    iou_day = validate_model_freiburg(model, val_loader_day, 'rgb_ir')

    # Evaluate night images
    iou_night = validate_model_freiburg(model, val_loader_night, 'rgb_ir')

    # Evaluate combined images
    iou = validate_model_freiburg(model, val_loader_combined, 'rgb_ir')

    print('Total mean IoU night: %f , day: %f, combined %f' % (iou_night, iou_day, iou))