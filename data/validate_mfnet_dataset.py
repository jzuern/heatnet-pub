import cv2
import torch
import numpy as np
from helper import config
from models import build_net
from models.confusion_maximization import vis_utils
from torch.autograd import Variable
from models.confusion_maximization.utils import calculate_ious
import torchvision.transforms.functional as F
import os
import torch.utils.data as data

class_mapping_ours_to_mfnet = {
    3: 4,  # curb
    9: 2,  # person
    10: 1,  # car,truck,bus,train
    11: 3,  # bicycle
}

vistas_stats = {'mean': [0.28389175985075144, 0.32513300997108185, 0.28689552631651594],
                     'std': [0.1777223070810445, 0.18099167120139084, 0.17613640748441522]}

def createMFNetValloader(root_dir, split):

    dataloader_val = MFDataset(root_dir, split=split)

    val_loader = torch.utils.data.DataLoader(dataloader_val,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=2,
                                               pin_memory=True,
                                               drop_last=False)

    return val_loader

class MFDataset(data.Dataset):

    def __init__(self, data_dir, split):
        super(MFDataset, self).__init__()

        if split == 'day':
            with open(os.path.join(data_dir, 'test_day.txt'), 'r') as f:
                self.names = [name.strip() for name in f.readlines()]
        elif split == 'night':
            with open(os.path.join(data_dir, 'test_night.txt'), 'r') as f:
                self.names = [name.strip() for name in f.readlines()]
        elif split == 'combined':
            with open(os.path.join(data_dir, 'test.txt'), 'r') as f:
                self.names = [name.strip() for name in f.readlines()]
        else:
            print('Unknown split. Exiting')
            exit()


        self.data_dir = data_dir
        self.split = split
        self.n_data = len(self.names)
        self.width = 640
        self.height = 480

    def __len__(self):
        return self.n_data

    def __getitem__(self, index):

        name = self.names[index]

        img_file = self.data_dir + '/images/' + name + '.png'
        label_file = self.data_dir + '/labels/' + name + '.png'

        im = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)

        rgb_im = im[:, :, :3]
        ir_im = im[:, :, 3:]

        rgb_im = cv2.cvtColor(rgb_im, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)

        rgb_im = cv2.resize(rgb_im, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        ir_im = cv2.resize(ir_im, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        rgb_im = F.to_tensor(rgb_im)
        ir_im = F.to_tensor(ir_im)
        label = torch.from_numpy(label)

        rgb_im = F.normalize(rgb_im, mean=[0.28389175985075144, 0.32513300997108185, 0.28689552631651594],
                                std=[0.1777223070810445, 0.18099167120139084, 0.17613640748441522])
        ir_im = F.normalize(ir_im, mean=[0.5], std=[0.5])

        out_dict = {}

        out_dict['rgb'] = rgb_im
        out_dict['label'] = label
        out_dict['ir'] = ir_im

        return out_dict

def validate_model_mfnet(model, val_loader):

    print('Evaluating MFNet dataset')

    # Containers for results
    preds = Variable(torch.zeros(len(val_loader), 480, 640))
    gts = Variable(torch.zeros(len(val_loader), 480, 640))

    for i, (batch) in enumerate(val_loader):

        rgb_im = batch['rgb'].cuda()
        ir_im = batch['ir'].cuda()
        label = batch['label'].cuda()
        label = label.to(torch.long)

        print('Validating ... %d of %d ...' % (i, len(val_loader)))

        with torch.no_grad():
            segmented = model(rgb_im)

        segmented = segmented[0:1, ...]

        segmented_argmax = torch.argmax(segmented.cpu(), 1).squeeze()
        segmented = torch.zeros_like(segmented_argmax)
        for our_label, mf_label in class_mapping_ours_to_mfnet.items():
            segmented[segmented_argmax == our_label] = mf_label

        # vis_utils.visImage3Chan(rgb_im[0:1, 0:3, ...], 'rgb_image')
        # vis_utils.visDepth(ir_im[0:1, ...], 'ir_image')
        # color_gt = color_coder.color_code_labels(label[0:1, ...], False)
        # color_pred = color_coder.color_code_labels(segmented, False)
        # cv2.imshow('GT', color_gt)
        # cv2.imshow('PRED', color_pred)
        # cv2.waitKey()

        gts[i, :, :] = label.squeeze()
        preds[i, :, :] = segmented

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


scale_factor = 4
color_coder = vis_utils.ColorCode(256)

conf = config.load_config("../experiments/heatnet_conf.json")

# model_params = utils.get_model_params(conf["network"])
# model = models.segnet.__dict__["net_" + conf["network"]["arch"]](**model_params)
model, starting_epoch = build_net.build_network(None, 'resnet50')

model = torch.nn.DataParallel(model, [0]).cuda()

pretrained_dict = torch.load("/mnt/hpc.vertensj/software/thermal_seg/segmentation/model_best.pth.tar", map_location=lambda storage, loc: storage)['state_dict']
model.load_state_dict(pretrained_dict)

# initModelPartial(model, "/mnt/hpc.shared/final_model_with_background.pth.tar")
# initModelPartial(model, "last_12.pth.tar")

model.eval()
np.random.seed(1203412412)
test_dataset_day = createMFNetValloader("/mnt/hpc.shared/MFNet_dataset", 'day')
test_dataset_night = createMFNetValloader("/mnt/hpc.shared/MFNet_dataset", 'night')
test_dataset_combined = createMFNetValloader("/mnt/hpc.shared/MFNet_dataset", 'combined')

miou_day = validate_model_mfnet(model, test_dataset_day)
miou_night = validate_model_mfnet(model, test_dataset_night)
miou_combined = validate_model_mfnet(model, test_dataset_combined)

print('Mean IOU Day: %f, Night: %f , Combined %f' % (miou_day, miou_night, miou_combined))

