import cv2
import numpy as np
from c_relabeller.relabeller import relabel_vistas_image
import os, glob, sys

vistasPath = "/mnt/rzc.vertensj/training_data/mapillary-vistas-dataset_public_v1.0/"
split = "validation"  # "training" or "validation"
save_dir = '/mnt/hpc.shared/vistas_data' + "/iros_2020_data_val/"

searchLabels = os.path.join(vistasPath, split, "instances", "*.png")
searchImages = os.path.join(vistasPath, split, "images", "*.jpg")

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(save_dir + "images/"):
    os.makedirs(save_dir + "images/")
if not os.path.exists(save_dir + "labels/"):
    os.makedirs(save_dir + "labels/")

filesLabels = glob.glob(searchLabels)
filesLabels.sort()
filesImages = glob.glob(searchImages)
filesImages.sort()

if os.path.isfile(save_dir + "paths.txt"):
    print("Already parsed...")
    exit()

print("Found %d instance train ids files..." % (len(filesLabels)))
print("Found %d image files..." % (len(filesImages)))

for nr, (label_path, image_path) in enumerate(zip(filesLabels, filesImages)):
    print("Processing image: %d" % (nr))
    image_filename = os.path.splitext(os.path.basename(image_path))[0]
    label_filename = os.path.splitext(os.path.basename(label_path))[0]

    label = cv2.imread(label_path, -1)
    image = cv2.imread(image_path)

    # scale_augment = [0.75,  1.0,  1.25,  1.5]
    scale_augment = [1.0]

    for aug_i in scale_augment:

        height = label.shape[0]
        width = label.shape[1]

        aspect_ratio = float(height / width)
        new_size = (1024, int(aspect_ratio * 1024))
        new_size = (int(new_size[0]*aug_i), int(new_size[1]*aug_i))

        image_resized = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
        label_resized = cv2.resize(label, new_size, interpolation=cv2.INTER_NEAREST)

        relabelled = np.asarray(relabel_vistas_image(label_resized))

        save_path_label = save_dir + "labels/" + label_filename + str(aug_i) + ".png"
        save_path_image = save_dir + "images/" + image_filename + str(aug_i) + ".png"

        cv2.imwrite(save_path_label, relabelled)
        cv2.imwrite(save_path_image, image_resized)

        with open(save_dir + "paths.txt", "a") as myfile:
           myfile.write(save_path_label + " " + save_path_image + "\n")

        # cv2.imshow("image_resized", image_resized)
        # cv2.imshow("label_resized", instance_label_array * 100)
        # cv2.imshow("borders_resized", borders * 100)
        # cv2.waitKey()