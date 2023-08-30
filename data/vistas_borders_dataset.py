import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import random
import cv2
import torch
import numpy as np
import copy

def resizeAndPad(img, size, padColor=11, interp = cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    sh, sw = size

    # aspect ratio of image
    aspect = w/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img

class VistasBorderDataLoader(data.Dataset):
    def __init__(self, db_path, width, height, transform=None, mean=None, contrast_enhancement=True, augment_data=True, sub_mean=True):
        self.width = width
        self.height = height
        self.transform = transform
        self.mean = mean
        self.augment_data = augment_data

        self.sub_mean = sub_mean

        self.contrast_enhancement = contrast_enhancement
        gridsize = 8
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize, gridsize))

        matches = []
        matches.append(db_path)

        self.list_dataset_paths = []  # Stores pairs of consecutive image paths

        for f in matches:
            with open(f) as f:
                content = f.readlines()
            content = [x.strip() for x in content]

            for line in content:
                frames = line.split(" ")  # Current first, then previous
                #frames = [f.replace('/home/vertensj/', '/mnt/rzc.vertensj/') for f in frames]
                # frames = [f.replace('/mnt/rzc.vertensj/', '/home/vertensj/') for f in frames]
                self.list_dataset_paths.append((frames[0], frames[1], frames[2], frames[3]))  # Borders, TrainIds, Image, DT

        self.length = len(self.list_dataset_paths)
        print("Current number of image pair in vistas dataset: %d " % (self.length))

    def __getitem__(self, index):
        path_tupel = self.list_dataset_paths[index]
        borders_cv = cv2.imread(path_tupel[0], 0)
        dt_cv = cv2.imread(path_tupel[-1], 0)
        ids_cv = cv2.imread(path_tupel[1], -1)
        inst_cv = np.array(ids_cv % 256, dtype=np.uint8)
        ids_cv = np.array(ids_cv / 256, dtype=np.uint8) # Convert to class labels
        image_cv = cv2.imread(path_tupel[2])

        image_size = image_cv.shape
        if not(image_size[0] >= self.height and image_size[1] >= self.width):
            borders_cv = resizeAndPad(borders_cv, (self.height, borders_cv.shape[1]), interp=cv2.INTER_NEAREST, padColor = 2)
            dt_cv = resizeAndPad(dt_cv, (self.height, borders_cv.shape[1]), interp=cv2.INTER_NEAREST, padColor = 2)
            ids_cv = resizeAndPad(ids_cv, (self.height, ids_cv.shape[1]), interp=cv2.INTER_NEAREST, padColor = 11)
            inst_cv = resizeAndPad(inst_cv, (self.height, inst_cv.shape[1]), interp=cv2.INTER_NEAREST, padColor = 11)
            image_cv = resizeAndPad(image_cv, (self.height, image_cv.shape[1]), padColor = 0)

        # Contrast enhancement##########################################################################################

        if self.contrast_enhancement:
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2LAB)

            image_lab_planes = cv2.split(image_cv)

            image_lab_planes[0] = self.clahe.apply(image_lab_planes[0])

            image_cv = cv2.merge(image_lab_planes)

            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_LAB2RGB)
        else:
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

            if not self.augment_data:
                if image_cv.shape[0] > 1200:
                    height = int(image_cv.shape[0]/4)
                    width = int(image_cv.shape[1]/4)
                    image_cv = cv2.resize(image_cv, (width, height), interpolation=cv2.INTER_LINEAR)

            # Joint augmentations#######################################################################################

        # To PIL image
        image = Image.fromarray(image_cv)
        borders = Image.fromarray(borders_cv)
        dt = Image.fromarray(dt_cv)
        ids = Image.fromarray(ids_cv)
        inst = Image.fromarray(inst_cv)

        if self.augment_data:
            # Cropping
            i, j, h, w = transforms.RandomCrop.get_params(image, (self.height, self.width))
            image = F.crop(image, i, j, h, w)
            borders = F.crop(borders, i, j, h, w)
            dt = F.crop(dt, i, j, h, w)
            ids = F.crop(ids, i, j, h, w)
            inst = F.crop(inst, i, j, h, w)

            # Flipping
            if random.random() > 0.5:
                image = F.hflip(image)
                borders = F.hflip(borders)
                dt = F.hflip(dt)
                ids = F.hflip(ids)
                inst = F.hflip(inst)

            if random.random() > 0.4:
                # Color jitter
                jitter_transform = transforms.ColorJitter.get_params(0.2, 0.2, 0.2, 0.0)
                image = jitter_transform(image)

        # To torch tensor
        image = F.to_tensor(image)
        ids_expand = np.expand_dims(np.array(ids), 0)
        ids = torch.from_numpy(ids_expand)

        inst_expand = np.expand_dims(np.array(inst), 0)
        inst = torch.from_numpy(inst_expand)

        dt_borders = Image.fromarray(cv2.distanceTransform(1-np.array(borders), cv2.DIST_L2, 5))

        borders = torch.from_numpy(np.expand_dims(np.array(borders), 0))
        dt = torch.from_numpy(np.expand_dims(np.array(dt), 0))
        dt_borders = torch.from_numpy(np.expand_dims(np.array(dt_borders), 0))

        image_org = copy.deepcopy(image)
        if self.sub_mean:
            image = F.normalize(image, mean=[0.28389175985075144, 0.32513300997108185, 0.28689552631651594],
                                std=[0.1777223070810445, 0.18099167120139084, 0.17613640748441522])

        # Subtracted image with gaps between semantic regions
        borders_np = borders.numpy()
        borders_clip = np.expand_dims(np.clip(borders_np, 0, 1), 0)

        # sub_id = torch.from_numpy(ids_np)
        borders_clip = torch.from_numpy(borders_clip)

        return borders, ids, inst, image, borders_clip, image_org, dt_borders, dt

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'
