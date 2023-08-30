import torch.utils.data as data
import os
import os.path
import fnmatch
import cv2
import torch
import numpy as np

image_stats = {'mean': [0.35675976, 0.37380189, 0.3764753],
               'std': [0.32064945, 0.32098866, 0.32325324]}

def applyClaheCV(clahe, images):
    for idx in range(len(images)):
        image_cv = cv2.cvtColor(images[idx], cv2.COLOR_RGB2LAB)
        lab_planes = cv2.split(image_cv)
        lab_planes[0] = clahe.apply(lab_planes[0])
        image_cv = cv2.merge(lab_planes)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_LAB2RGB)
        images[idx] = image_cv
    return images

def normalizeRGBBatch(images, stats=image_stats):
    mean = torch.Tensor(stats['mean']).cuda()
    std = torch.Tensor(stats['std']).cuda()
    for image in images:
        for c in range(image.size(1)):
            image[:, c, ...] = ((image[:, c, ...] - mean[c]) / std[c])
    return images

def readFiles(path):
    with open(path) as file:
        content = file.readlines()
        content = [x.strip() for x in content]
    return content

def searchForFiles(name, path):
    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, name):
            matches.append(os.path.join(root, filename))
    return matches

def getImageStats():
    return image_stats

class ThermalDataLoader(data.Dataset):
    def __init__(self, db_path, contrast_enhancement=False, load_aligned_ir = False, load_right=True):

        self.list_dataset_paths = []

        rgb_fl_files = searchForFiles("fl_rgb_drive_*.txt", db_path)
        rgb_fr_files = searchForFiles("fr_rgb_drive_*.txt", db_path)
        ir_fl_files = searchForFiles("fl_ir_drive_*.txt", db_path)
        ir_fr_files = searchForFiles("fr_ir_drive_*.txt", db_path)

        rgb_fl_files.sort()
        rgb_fr_files.sort()
        ir_fl_files.sort()
        ir_fr_files.sort()

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.contrast_enhancement = contrast_enhancement
        self.load_aligned_ir = load_aligned_ir
        self.load_right = load_right

        print('Images Found: %d' % (len(rgb_fl_files)))
        assert (len(rgb_fl_files) == len(rgb_fr_files) == len(ir_fl_files) == len(ir_fr_files))

        for i in range(len(rgb_fl_files)):
            rgb_fl_paths = readFiles(rgb_fl_files[i])
            rgb_fr_paths = readFiles(rgb_fr_files[i])
            ir_fl_paths = readFiles(ir_fl_files[i])
            ir_fr_paths = readFiles(ir_fr_files[i])

            for line_left, line_right, lines_left_ir, lines_right_ir in zip(rgb_fl_paths, rgb_fr_paths, ir_fl_paths,
                                                                            ir_fr_paths):
                frames_left = line_left.split(" ")
                frames_right = line_right.split(" ")
                frames_left_ir = lines_left_ir.split(" ")
                frames_right_ir = lines_right_ir.split(" ")

                self.list_dataset_paths.append([frames_left, frames_right, frames_left_ir, frames_right_ir])

        self.length = len(self.list_dataset_paths)
        print("Current number of image pair in thermal dataset: %d " % (self.length))

    def __getitem__(self, index):
        paths = self.list_dataset_paths[index]

        rgb_fr = None
        ir_fr = None

        rgb_fl = [cv2.imread(p) for p in paths[0]]

        if self.load_right:
            rgb_fr = [cv2.imread(p) for p in paths[1]]

        if self.contrast_enhancement:
            applyClaheCV(self.clahe, rgb_fl)
            if self.load_right:
                applyClaheCV(self.clahe, rgb_fr)
        org_left = rgb_fl[0]

        if self.load_aligned_ir:
            for i, p in enumerate(paths[2]):
                file_name = os.path.split(p)[1].replace('fl_ir', 'fl_ir_aligned')
                remapped_path = os.path.split(os.path.split(p)[0])[0] + '/fl_ir_aligned/' + file_name
                paths[2][i] = remapped_path

        ir_fl = []
        ir_fr = []

        for i in range(len(paths[2])):
            l_path = paths[2][i]
            r_path = paths[3][i]

            if os.path.isfile(l_path):
                ir_fl.append(cv2.imread(l_path, cv2.IMREAD_ANYDEPTH) )
            if os.path.isfile(r_path):
                ir_fr.append(cv2.imread(r_path, cv2.IMREAD_ANYDEPTH) )

        # ir_fl = [cv2.imread(p, cv2.IMREAD_ANYDEPTH) for p in paths[2]]
        # if self.load_right:
        #     ir_fr = [cv2.imread(p, cv2.IMREAD_ANYDEPTH) for p in paths[3]]

        ir_fl = [torch.from_numpy(im.astype(np.float32)).unsqueeze(0) for im in ir_fl]
        if self.load_right:
            ir_fr = [torch.from_numpy(im.astype(np.float32)).unsqueeze(0) for im in ir_fr]

        rgb_fl = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in rgb_fl]
        if self.load_right:
            rgb_fr = [cv2.cvtColor(r, cv2.COLOR_BGR2RGB) for r in rgb_fr]

        rgb_fl = [torch.from_numpy(im).permute(2, 0, 1).float()/255 for im in rgb_fl]
        if self.load_right:
            rgb_fr = [torch.from_numpy(im).permute(2, 0, 1).float()/255 for im in rgb_fr]

        out_dict = {}
        out_dict['rgb_fl'] = rgb_fl
        out_dict['rgb_fr'] = rgb_fr
        out_dict['ir_fl'] = ir_fl
        out_dict['ir_fr'] = ir_fr
        out_dict['paths_left'] = paths[0]
        out_dict['org_left'] = org_left

        return out_dict

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'
