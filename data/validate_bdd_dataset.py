import cv2
import torch
import numpy as np
from helper import config
from models import build_net
from models.confusion_maximization import vis_utils, thermal_loader
from torch.autograd import Variable
from models.confusion_maximization.utils import calculate_ious

vistas_stats = {'mean': [0.28389175985075144, 0.32513300997108185, 0.28689552631651594],
                     'std': [0.1777223070810445, 0.18099167120139084, 0.17613640748441522]}

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

def createValloader(data_dirnames):
    dataloader_val = thermal_loader.BDDValDataset(db_path=data_dirnames, db_stats=vistas_stats)

    val_loader = torch.utils.data.DataLoader(dataloader_val,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=2,
                                               pin_memory=True,
                                               drop_last=False)
    return val_loader

def validate_model_bdd(model, val_loader):

    print('Evaluating...')

    preds = Variable(torch.zeros(len(val_loader), 320, 704))
    gts = Variable(torch.zeros(len(val_loader), 320, 704))

    color_coder = vis_utils.ColorCode(14)

    for i, batch in enumerate(val_loader):
        print('Validating ... %d of %d ...' % (i, len(val_loader)))
        rgb_im = batch['rgb'].cuda()
        label = batch['label'].cuda()
        label = label.to(torch.long)

        with torch.no_grad():
            segmented = model(rgb_im)

        segmented_argmax = torch.argmax(segmented.cpu(), 1).squeeze()

        label_ = torch.full_like(label, 12)
        for bdd_label, our_label in class_mapping_BDD_to_ours.items():
            label_[label == bdd_label] = our_label

#        color_gt = color_coder.color_code_labels(label_[0:1, ...], False)
#        color_pred = color_coder.color_code_labels(segmented_argmax, False)
#        cv2.imshow('GT', color_gt)
#        cv2.imshow('PRED', color_pred)
#        vis_utils.visImage3Chan(batch['rgb_org'], 'RGB')
#        cv2.waitKey()
        save_dir = "/home/vertensj/Documents/papers/iros_hotnet/qual_comp/bdd_rgb_only_model/night/"
        if save_dir is not "":
            pred_color = color_coder.color_code_labels(segmented)
            gt_color = color_coder.color_code_labels(label_, argmax=False)
            rgb_image = np.transpose(batch['rgb_org'].cpu().data.numpy().squeeze(), (1, 2, 0))

            cv2.imwrite(save_dir + '/pred_' + str(i) + ".png", (pred_color*255).astype(np.uint8))
            cv2.imwrite(save_dir + '/gt_' + str(i) + ".png", (gt_color*255).astype(np.uint8))
            cv2.imwrite(save_dir + '/rgb_' + str(i) + ".png",
                        cv2.cvtColor((rgb_image * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))

        # account for offset in GT labels due to _background_ classes
        gts[i, :, :] = label_.squeeze()
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


def validate_on_bdd(model):
    testroot_night = '/mnt/hpc.shared/bdd100k_seg/bdd100k/seg/'

    val_loader_night = createValloader(testroot_night)

    # Evaluate night images
    iou_night = validate_model_bdd(model, val_loader_night)

    print('Total mean IoU night: %f' % (iou_night))

color_coder = vis_utils.ColorCode(256)

conf = config.load_config("../experiments/heatnet_conf.json")

# model_params = utils.get_model_params(conf["network"])
# model = models.segnet.__dict__["net_" + conf["network"]["arch"]](**model_params)
model, starting_epoch = build_net.build_network(None, 'resnet50')

model = torch.nn.DataParallel(model, [0]).cuda()

pretrained_dict = torch.load("/mnt/hpc.vertensj/software/thermal_seg/segmentation/model_best.pth.tar", map_location=lambda storage, loc: storage)['state_dict']
model.load_state_dict(pretrained_dict)

validate_on_bdd(model)

model.eval()
np.random.seed(1203412412)
