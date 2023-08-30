import cv2
import torch
from data import vistas_borders_dataset
import numpy as np
from random import randint
from helper import utils, config
import models

import time

def visImage1Chan(data, name, size=(1280, 384)):
    cv = data.cpu().data.numpy().squeeze()
    cv2.normalize(cv, cv, 0, 255, cv2.NORM_MINMAX)
    cv = cv.astype(np.uint8)
    cv = cv2.resize(cv, size).astype(np.uint8)
    cv2.imshow(name, cv)
    return cv

def visDepth(data, name, size=(640, 320)):
    """
    Visualize depth map
    :param data: torch tensor
    :param name: name of the window
    :param size: size of the image
    """
    disp_cv = data.cpu().data.numpy().squeeze()
    # disp_cv = np.clip(disp_cv, 0, 30)
    cv2.normalize(disp_cv, disp_cv, 0, 255, cv2.NORM_MINMAX)
    disp_cv_color = cv2.applyColorMap(disp_cv.astype(np.uint8), cv2.COLORMAP_JET)
    # disp_cv_color = cv2.resize(disp_cv_color, size)
    cv2.imshow(name, disp_cv_color)
    return disp_cv_color

class ColorCode():
    """
    Class for color coding segmentation masks
    """

    def get_coding_1(self):
        coding = {}
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

    def random_color_coding(self, max_label):
        coding = {}
        for i in range(max_label + 1):
            coding[i] = [randint(0, 255), randint(0, 255), randint(0, 255)]
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
                # print(labels_cv[y, x])
                color_coded[y, x, :] = self.color_coding[labels_cv[y, x]]

        return color_coded

    def __init__(self, max_classes):
        super(ColorCode, self).__init__()
        self.color_coding = self.random_color_coding(max_classes)

def initModelPartial(model, weights_path):
        model_dict = model.state_dict()
        pretrained_dict = torch.load(weights_path, map_location=lambda storage, loc: storage)['state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

height = 384
width = 768
color_coder = ColorCode(256)

print("Open dataset...")
images_db = vistas_borders_dataset.VistasBorderDataLoader("/mnt/rzc.vertensj/training_data/mapillary-vistas-dataset_public_v1.0/borders_augmented4/paths.txt", width, height, contrast_enhancement=False, augment_data=False, sub_mean=True)

data_len = images_db.length

images_loader = torch.utils.data.DataLoader(images_db, batch_size=1,
                                            shuffle=True, num_workers=3)
print("Opened vistas dataset successfully...")

conf = config.load_config("../experiments/heatnet_conf.json")
model_params = utils.get_model_params(conf["network"])
model = models.__dict__["net_" + conf["network"]["arch"]](**model_params)
model = torch.nn.DataParallel(model, [0]).cuda()

initModelPartial(model, "last_12.pth.tar")

model.eval()
np.random.seed(1203412412)

for c_i, (borders, ids, inst, image, sub_ids, org, dt, dt_inner_gt) in enumerate(images_loader):

    print("Iteration: %d " % (c_i))

    gt_seg = ids_cv = color_coder.color_code_labels(ids, argmax=False)

    ids = ids.cuda()

    borders_var = torch.autograd.Variable(borders.squeeze(1).long())
    ids_var = torch.autograd.Variable(ids.squeeze(1).long())

    torch.cuda.synchronize()
    start_t = time.time()
    with torch.no_grad():
        img_seg = model(image)
    torch.cuda.synchronize()
    end_t = time.time()
    print('Network took %f seconds...' % (end_t-start_t))
    labels, indices = img_seg.max(1)

    output_seg = color_coder.color_code_labels(indices, argmax=False)

    kernel = np.ones((3, 3), np.uint8)

    image_cv = np.transpose(image.numpy().squeeze(), (1, 2, 0))
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    cv2.imshow("rgb", image_cv)
    cv2.imshow("output_seg", output_seg)

    cv2.waitKey()
