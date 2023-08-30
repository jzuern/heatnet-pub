import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import random
import cv2
import torch
import numpy as np
import copy
from glob import glob
import os

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
    def __init__(self, db_path, width, height, transform=None, mean=None, contrast_enhancement=False, augment_data=True, sub_mean=True, background_id=12):
        self.width = width
        self.height = height
        self.transform = transform
        self.mean = mean
        self.augment_data = augment_data

        self.sub_mean = sub_mean
        self.background_id = background_id

        self.contrast_enhancement = contrast_enhancement
        gridsize = 8
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(gridsize, gridsize))

        matches = []
        matches.append(db_path)

        self.list_dataset_paths = []  # Stores pairs of consecutive image paths

        label_files = list(sorted(glob(os.path.join(db_path, 'labels/*.png'))))

        for l in label_files:
            rgb_file = l.replace('labels', 'images')

            if os.path.isfile(rgb_file):
                self.list_dataset_paths.append((rgb_file, l))

        self.length = len(self.list_dataset_paths)
        print("Current number of image pair in vistas dataset: %d " % (self.length))

    def __getitem__(self, index):
        path_tupel = self.list_dataset_paths[index]
        image_cv = cv2.imread(path_tupel[0])
        ids_cv = cv2.imread(path_tupel[1], -1)
        ids_cv = np.array(ids_cv / 256, dtype=np.uint8) # Convert to class labels

        if self.augment_data:
            if random.random() > 0.0:
                height = image_cv.shape[0]
                width = image_cv.shape[1]
                scale = random.choice([0.75,  1.0,  1.25])
                aspect_ratio = float(height / width)
                new_size = (1024, int(aspect_ratio * 1024))
                new_size = (int(new_size[0] * scale), int(new_size[1] * scale))

                image_cv = cv2.resize(image_cv, new_size, interpolation=cv2.INTER_LINEAR)
                ids_cv = cv2.resize(ids_cv, new_size, interpolation=cv2.INTER_NEAREST)


        image_size = image_cv.shape
        if not(image_size[0] >= self.height and image_size[1] >= self.width):
            ids_cv = resizeAndPad(ids_cv, (self.height, ids_cv.shape[1]), interp=cv2.INTER_NEAREST, padColor = self.background_id)
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
                    image_cv = cv2.resize(image_cv, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                    ids_cv = cv2.resize(ids_cv, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        # Joint augmentations#######################################################################################
        # To PIL image

        image = Image.fromarray(image_cv)
        ids = Image.fromarray(ids_cv)

        if self.augment_data:
            # Cropping
            i, j, h, w = transforms.RandomCrop.get_params(image, (self.height, self.width))
            image = F.crop(image, i, j, h, w)
            ids = F.crop(ids, i, j, h, w)

            # Flipping
            if random.random() > 0.5:
                image = F.hflip(image)
                ids = F.hflip(ids)

            if random.random() > 0.4:
                # Color jitter
                self.brightness = (0.8, 1.2)
                self.contrast = (0.8, 1.2)
                self.saturation = (0.8, 1.2)
                self.hue = (-0.1, 0.1)
                jitter_transform =  transforms.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
                image = jitter_transform(image)

            if random.random() > 0.5:
                angle = (random.random() - 0.5) * 40  # random angle in [-20, 20]
                image = F.rotate(image, angle, resample=Image.BILINEAR)
                ids = F.rotate(ids, angle, resample=Image.NEAREST)

        # To torch tensor
        image = F.to_tensor(image)
        ids_expand = np.expand_dims(np.array(ids), 0)
        ids = torch.from_numpy(ids_expand)

        image_org = copy.deepcopy(image)
        if self.sub_mean:
            image = F.normalize(image, mean=[0.28389175985075144, 0.32513300997108185, 0.28689552631651594],
                                std=[0.1777223070810445, 0.18099167120139084, 0.17613640748441522])

        return image, ids, image_org

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'
