import torch
import numpy as np
from torch.autograd import Variable
import wandb
from utils import calculate_ious
from vis_utils import ColorCode, visImage3Chan, visDepth
import cv2

class_mapping_ours_to_mfnet = {
    3: 4,  # curb
    9: 2,  # person
    10: 1,  # car,truck,bus,train
    11: 3,  # bicycle
}

class_mapping_BDD_to_ours = {
    0: 0,       # road
    1: 1,       # sidewalk
    2: 2,       # building
    3: 2,       # wall
    4: 4,       # fence
    5: 5,       # pole
    6: 5,       # traffic light
    7: 5,       # traffic sign
    8: 6,       # vegetation
    9: 7,       # terrain
    10: 8,      # sky
    11: 9,      # person
    12: 9,      # rider
    13: 10,     # car
    14: 10,     # truck
    15: 10,     # bus
    16: 10,     # train
    17: 11,     # motorcycle
    18: 11,     # bicycle
    255: 13     # ignore
}

def validate_model_bdd(model, val_loader, vis=False, save_dir=""):

    print('Evaluating BDD Night dataset {}.')

    # Containers for results
    # width = 1280
    # height = 720
    width = 704
    height = 320

    preds = Variable(torch.zeros(len(val_loader), height, width))
    gts = Variable(torch.zeros(len(val_loader), height, width))

    preds_rgb = Variable(torch.zeros(len(val_loader), 3, height, width))
    gts_rgb = Variable(torch.zeros(len(val_loader), 3, height, width))
    imgs_rgb = Variable(torch.zeros(len(val_loader), 3, height, width))

    color_coder = ColorCode(256)


    for i, batch in enumerate(val_loader):

        print('Validating ... %d of %d ...' % (i, len(val_loader)))

        rgb_im = batch['rgb'].cuda()
        label = batch['label'].cuda()
        label = label.to(torch.long)

        in_night = [rgb_im]

        in_night_d = []
        for t in in_night:
            in_night_d.append(torch.cat([t, t], dim=0))

        with torch.no_grad():
            segmented, _, _ = model(*in_night_d)
        segmented = segmented[0:1, ...]

        segmented_argmax = torch.argmax(segmented.cpu(), 1).squeeze()

        # apply label mapping
        label_ = torch.full_like(label, 12)
        for bdd_label, our_label in class_mapping_BDD_to_ours.items():
            label_[label == bdd_label] = our_label

        if vis:
            pred_color = color_coder.color_code_labels(segmented)
            gt_color = color_coder.color_code_labels(label_, argmax=False)
            cv2.imshow('Pred Seg', pred_color)
            cv2.imshow('GT Seg', gt_color)
            visImage3Chan(batch['rgb_org'], 'RGB')
            cv2.waitKey()

        if save_dir is not "":
            pred_color = color_coder.color_code_labels(segmented)
            gt_color = color_coder.color_code_labels(label_, argmax=False)
            rgb_image = np.transpose(batch['rgb_org'].cpu().data.numpy().squeeze(), (1, 2, 0))

            cv2.imwrite(save_dir + '/pred_' + str(i) + ".png", (pred_color * 255).astype(np.uint8))
            cv2.imwrite(save_dir + '/gt_' + str(i) + ".png", (gt_color * 255).astype(np.uint8))
            cv2.imwrite(save_dir + '/rgb_' + str(i) + ".png", cv2.cvtColor((rgb_image * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))

        gts[i, :, :] = label_.squeeze()
        preds[i, :, :] = segmented_argmax

        segmentation_gt = gts[i, :, :]
        segmentation_gt = color_coder.color_code_labels(segmentation_gt, argmax=False)
        segmentation_gt = np.transpose(segmentation_gt, (2, 0, 1))
        segmentation_gt = np.expand_dims(segmentation_gt, axis=0)
        segmentation_gt = torch.from_numpy(segmentation_gt).float()
        gts_rgb[i, :, :, :] = segmentation_gt

        segmentation_pred = preds[i, :, :]
        segmentation_pred = color_coder.color_code_labels(segmentation_pred, argmax=False)
        segmentation_pred = np.transpose(segmentation_pred, (2, 0, 1))
        segmentation_pred = np.expand_dims(segmentation_pred, axis=0)
        segmentation_pred = torch.from_numpy(segmentation_pred).float()

        preds_rgb[i, :, :, :] = segmentation_pred
        imgs_rgb[i, :, :, :] = rgb_im.squeeze()

        wandb.log({"BDD Night Test Images": [wandb.Image(imgs_rgb[0:10], caption="RGB"),
                                             wandb.Image(gts_rgb[0:10], caption="Segmentation Ground Truth"),
                                             wandb.Image(preds_rgb[0:10], caption="Segmentation Prediction")],
                   })

    acc = calculate_ious(preds, gts)

    wandb.log({
        "BDD Night Test mean IoU": np.nanmean(acc),
        "BDD Night Test IoU road,parking": acc[0],
        "BDD Night Test IoU ground,sidewalk": acc[1],
        "BDD Night Test IoU building,": acc[2],
        'BDD NightTest IoU curb': acc[3],
        'BDD NightTest IoU fence': acc[4],
        'BDD NightTest IoU pole,traffic light,traffic sign': acc[5],
        'BDD Night Test IoU vegetation': acc[6],
        'BDD Night Test IoU terrain': acc[7],
        'BDD Night Test IoU sky': acc[8],
        'BDD NightTest IoU person,rider': acc[9],
        'BDD NightTest IoU car,truck,bus,train': acc[10],
        'BDD Night Test IoU motorcycle,bicycle': acc[11]
    })
    return acc


def validate_model_mfnet(model, val_loader, modalities, mode="day", vis=False, save_dir=""):

    print('Evaluating MFNet dataset {}.'.format(mode))

    # Containers for results
    preds = Variable(torch.zeros(len(val_loader), 480, 640))
    gts = Variable(torch.zeros(len(val_loader), 480, 640))

    preds_rgb = Variable(torch.zeros(len(val_loader), 3, 480, 640))
    gts_rgb = Variable(torch.zeros(len(val_loader), 3, 480, 640))
    imgs_rgb = Variable(torch.zeros(len(val_loader), 3, 480, 640))
    imgs_ir = Variable(torch.zeros(len(val_loader), 480, 640))

    color_coder = ColorCode(256)

    for i, batch in enumerate(val_loader):

        print('Validating ... %d of %d ...' % (i, len(val_loader)))
        rgb_im = batch['rgb'].cuda()
        ir_im = batch['ir'].cuda()
        label = batch['label'].cuda()
        label = label.to(torch.long)

        # encoder
        if 'rgb' in modalities and 'ir' in modalities:
            in_night = [rgb_im, ir_im]
        elif 'rgb' in modalities:
            in_night = [rgb_im]
        elif 'ir' in modalities:
            in_night = [ir_im]
        else:
            print('No known modality selected....')
            exit()

        in_night_d = []
        for t in in_night:
            in_night_d.append(torch.cat([t, t], dim=0))

        with torch.no_grad():
            segmented, _, _ = model(*in_night_d)
        segmented = segmented[0:1, ...]

        if vis:
            pred_color = color_coder.color_code_labels(segmented)
            gt_color = color_coder.color_code_labels(label, argmax=False)
            cv2.imshow('Pred Seg', pred_color)
            cv2.imshow('GT Seg', gt_color)
            visImage3Chan(batch['rgb_org'], 'RGB')
            # visDepth(batch['ir_org'].clamp(0.3, 1.0), 'IR')
            visDepth(batch['ir_org'], 'IR')
            cv2.waitKey()

        if save_dir is not "":
            pred_color = color_coder.color_code_labels(segmented)
            gt_color = color_coder.color_code_labels(label, argmax=False)
            rgb_image = np.transpose(batch['rgb_org'].cpu().data.numpy().squeeze(), (1, 2, 0))

            # ir_cv = batch['ir_org'].clamp(0.2, 0.7).cpu().data.numpy().squeeze()
            ir_cv = batch['ir_org'].cpu().data.numpy().squeeze()
            cv2.normalize(ir_cv, ir_cv, 0, 255, cv2.NORM_MINMAX)
            ir_cv = cv2.applyColorMap(ir_cv.astype(np.uint8), cv2.COLORMAP_JET)

            cv2.imwrite(save_dir + '/pred_' + str(i) + ".png", (pred_color * 255).astype(np.uint8))
            cv2.imwrite(save_dir + '/gt_' + str(i) + ".png", (gt_color * 255).astype(np.uint8))
            cv2.imwrite(save_dir + '/rgb_' + str(i) + ".png",
                        cv2.cvtColor((rgb_image * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
            cv2.imwrite(save_dir + '/ir_' + str(i) + ".png", ir_cv)

        segmented_argmax = torch.argmax(segmented.cpu(), 1).squeeze()

        # apply label mapping
        segmented_ = Variable(torch.zeros(480, 640))
        for our_label, mf_label in class_mapping_ours_to_mfnet.items():
            segmented_[segmented_argmax == our_label] = mf_label

        gts[i, :, :] = label.squeeze()
        preds[i, :, :] = segmented_

        segmentation_gt = gts[i, :, :]
        segmentation_gt = color_coder.color_code_labels(segmentation_gt, argmax=False)
        segmentation_gt = np.transpose(segmentation_gt, (2, 0, 1))
        segmentation_gt = np.expand_dims(segmentation_gt, axis=0)
        segmentation_gt = torch.from_numpy(segmentation_gt).float()
        gts_rgb[i, :, :, :] = segmentation_gt

        segmentation_pred = preds[i, :, :]
        segmentation_pred = color_coder.color_code_labels(segmentation_pred, argmax=False)
        segmentation_pred = np.transpose(segmentation_pred, (2, 0, 1))
        segmentation_pred = np.expand_dims(segmentation_pred, axis=0)
        segmentation_pred = torch.from_numpy(segmentation_pred).float()

        preds_rgb[i, :, :, :] = segmentation_pred
        imgs_rgb[i, :, :, :] = rgb_im.squeeze()
        imgs_ir[i, :, :] = ir_im.squeeze()

    if mode == "day" or mode == "night":
        wandb.log({mode + "_MFNET_Test Images": [wandb.Image(imgs_rgb[0:10], caption="RGB"),
                                               wandb.Image(gts_rgb[0:10], caption="Segmentation Ground Truth"),
                                               wandb.Image(preds_rgb[0:10], caption="Segmentation Prediction")],
                   })

    acc = calculate_ious(preds, gts)

    wandb.log({
        mode + "_MFNET_Test mean IoU": np.nanmean(acc),
        mode + '_MFNET_Test IoU curb': acc[4],
        mode + '_MFNET_Test IoU person,rider': acc[2],
        mode + '_MFNET_Test IoU car,truck,bus,train': acc[1],
        mode + '_MFNET_Test IoU bicycle': acc[3],
    })

    return acc


def validate_model(model, val_loader, modalities, mode="day", vis=False, save_dir=""):

    print('Evaluating {}.'.format(mode))

    # model.eval()
    # Containers for results

    preds = Variable(torch.zeros(len(val_loader), 320, 704))
    gts = Variable(torch.zeros(len(val_loader), 320, 704))

    preds_rgb = Variable(torch.zeros(len(val_loader), 3, 320, 704))
    gts_rgb = Variable(torch.zeros(len(val_loader), 3, 320, 704))
    imgs_rgb = Variable(torch.zeros(len(val_loader), 3, 320, 704))
    imgs_ir = Variable(torch.zeros(len(val_loader), 320, 704))

    color_coder = ColorCode(256)

    for i, batch in enumerate(val_loader):
        print('Validating ... %d of %d ...' % (i, len(val_loader)))
        rgb_im = batch['rgb'].cuda()
        ir_im = batch['ir'].cuda()
        label = batch['label'].cuda()
        label = label.to(torch.long)

        # encoder

        if 'rgb' in modalities and 'ir' in modalities:
            in_night = [rgb_im, ir_im]
        elif 'rgb' in modalities:
            in_night = [rgb_im]
        elif 'ir' in modalities:
            in_night = [ir_im]
        else:
            print('No known modality selected....')
            exit()

#        in_night = torch.cat([rgb_im, ir_im], dim=1)
#         in_night = torch.cat([in_night, in_night], dim=0)
        in_night_d = []
        for t in in_night:
            in_night_d.append(torch.cat([t, t], dim=0))

        with torch.no_grad():
            segmented, _, _ = model(*in_night_d)
        segmented = segmented[0:1, ...]

        if vis:
            pred_color = color_coder.color_code_labels(segmented)
            gt_color = color_coder.color_code_labels(label, argmax=False)
            visImage3Chan(batch['rgb_org'], 'RGB')
            cv2.imshow('Pred Seg', pred_color)
            cv2.imshow('GT Seg', gt_color)
            # visDepth(batch['ir_org'].clamp(0.3, 1.0), 'IR')
            visDepth(batch['ir_org'], 'IR')
            cv2.waitKey()

        if save_dir is not "":
            pred_color = color_coder.color_code_labels(segmented)
            gt_color = color_coder.color_code_labels(label, argmax=False)
            rgb_image = np.transpose(batch['rgb_org'].cpu().data.numpy().squeeze(), (1, 2, 0))

            # ir_cv = batch['ir_org'].clamp(0.2, 0.7).cpu().data.numpy().squeeze()
            ir_cv = batch['ir_org'].cpu().data.numpy().squeeze()
            cv2.normalize(ir_cv, ir_cv, 0, 255, cv2.NORM_MINMAX)
            ir_cv = cv2.applyColorMap(ir_cv.astype(np.uint8), cv2.COLORMAP_JET)

            cv2.imwrite(save_dir + '/pred_' + str(i) + ".png", (pred_color*255).astype(np.uint8))
            cv2.imwrite(save_dir + '/gt_' + str(i) + ".png", (gt_color*255).astype(np.uint8))
            cv2.imwrite(save_dir + '/rgb_' + str(i) + ".png",
                        cv2.cvtColor((rgb_image * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
            cv2.imwrite(save_dir + '/ir_' + str(i) + ".png", ir_cv)

        segmented_argmax = torch.argmax(segmented.cpu(), 1)

        # account for offset in GT labels due to _background_ classes
        gts[i, :, :] = label.squeeze()
        preds[i, :, :] = segmented_argmax.squeeze()

        segmentation_gt = gts[i, :, :]
        segmentation_gt = color_coder.color_code_labels(segmentation_gt, argmax=False)
        segmentation_gt = np.transpose(segmentation_gt, (2, 0, 1))
        segmentation_gt = np.expand_dims(segmentation_gt, axis=0)
        segmentation_gt = torch.from_numpy(segmentation_gt).float()
        gts_rgb[i, :, :, :] = segmentation_gt

        segmentation_pred = preds[i, :, :]
        segmentation_pred = color_coder.color_code_labels(segmentation_pred, argmax=False)
        segmentation_pred = np.transpose(segmentation_pred, (2, 0, 1))
        segmentation_pred = np.expand_dims(segmentation_pred, axis=0)
        segmentation_pred = torch.from_numpy(segmentation_pred).float()

        preds_rgb[i, :, :, :] = segmentation_pred
        imgs_rgb[i, :, :, :] = rgb_im.squeeze()
        imgs_ir[i, :, :] = ir_im.squeeze()

    if mode == "day" or mode == "night":
        wandb.log({ mode + "_Test Images": [wandb.Image(imgs_rgb, caption="RGB"),
                                   # wandb.Image(imgs_ir, caption="IR"),
                                   wandb.Image(gts_rgb, caption="Segmentation Ground Truth"),
                                   wandb.Image(preds_rgb, caption="Segmentation Prediction")],
                   })

    ious = calculate_ious(preds, gts)

    wandb.log({
        mode + "_Test mean IoU": np.nanmean(ious),
        mode + "_Test IoU road,parking":                    ious[0],
        mode + "_Test IoU ground,sidewalk":                 ious[1],
        mode + "_Test IoU building,":                       ious[2],
        mode + '_Test IoU curb':                            ious[3],
        mode + '_Test IoU fence':                           ious[4],
        mode + '_Test IoU pole,traffic light,traffic sign': ious[5],
        mode + '_Test IoU vegetation':                      ious[6],
        mode + '_Test IoU terrain':                         ious[7],
        mode + '_Test IoU sky':                             ious[8],
        mode + '_Test IoU person,rider':                    ious[9],
        mode + '_Test IoU car,truck,bus,train':             ious[10],
        mode + '_Test IoU motorcycle,bicycle':              ious[11],
    })
    # model.train()
    return ious

def inference(model, val_loader, modalities, vis=False, save_dir=""):
    color_coder = ColorCode(256)

    for i, batch in enumerate(val_loader):
        print('Inference ... %d of %d ...' % (i, len(val_loader)))
        rgb_im = batch['rgb'].cuda()
        ir_im = batch['ir'].cuda()

        # encoder
        if 'rgb' in modalities and 'ir' in modalities:
            in_night = [rgb_im, ir_im]
        elif 'rgb' in modalities:
            in_night = [rgb_im]
        elif 'ir' in modalities:
            in_night = [ir_im]
        else:
            print('No known modality selected....')
            exit()

        in_night_d = []
        for t in in_night:
            in_night_d.append(torch.cat([t, t], dim=0))

        with torch.no_grad():
            segmented, _, _ = model(*in_night_d)
        segmented = segmented[0:1, ...]

        if vis:
            pred_color = color_coder.color_code_labels(segmented)
            visImage3Chan(batch['rgb_org'], 'RGB')
            cv2.imshow('Pred Seg', pred_color)
            visDepth(batch['ir_org'], 'IR')
            cv2.waitKey()

        if save_dir is not "":
            pred_color = color_coder.color_code_labels(segmented)
            rgb_image = np.transpose(batch['rgb_org'].cpu().data.numpy().squeeze(), (1, 2, 0))

            # ir_cv = batch['ir_org'].clamp(0.2, 0.7).cpu().data.numpy().squeeze()
            ir_cv = batch['ir_org'].cpu().data.numpy().squeeze() * 255
            # cv2.normalize(ir_cv, ir_cv, 0, 255, cv2.NORM_MINMAX)
            ir_cv = cv2.applyColorMap(ir_cv.astype(np.uint8), cv2.COLORMAP_JET)

            cv2.imwrite(save_dir + '/pred_' + str(i) + ".png", (pred_color*255).astype(np.uint8))
            cv2.imwrite(save_dir + '/rgb_' + str(i) + ".png",
                        cv2.cvtColor((rgb_image * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
            cv2.imwrite(save_dir + '/ir_' + str(i) + ".png", ir_cv)

    return

