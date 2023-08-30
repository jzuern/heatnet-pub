import numpy as np
from glob import glob
import cv2


class_names_dict = {
    0: 'road,parking',
    1: 'ground_sidewalk',
    2: 'building',
    3: 'curb',
    4: 'fence',
    5: 'pole,signs',
    6: 'vegetation',
    7: 'terrain',
    8: 'sky',
    9: 'person',
    10: 'car,truck,bus,train',
    11: 'bicycle',
    12: '_background_',
    13: '_ignore_'
}


def get_training_data_stats():

    fl_training_label_files = sorted(glob('/mnt/ir_rgb_data/*/*/fl_rgb_labels/*.png'))

    label_counter_dict = {}

    for k in class_names_dict.keys():
        label_counter_dict[k] = 0

    for labelfile in fl_training_label_files:

        label_day = cv2.imread(labelfile, cv2.IMREAD_GRAYSCALE)

        for k in label_counter_dict.keys():
            counter = np.count_nonzero(label_day == k)
            label_counter_dict[k] += counter


    print('\n\n\nTRAINING DATASET STATS\n\n\n')
    for k in label_counter_dict.keys():
        print(k, ' - ', class_names_dict[k], ' - ', label_counter_dict[k])






def get_testing_data_stats():

    class_counter_day = np.zeros(len(class_names_dict.keys()), dtype=np.int32)
    class_counter_night = np.zeros(len(class_names_dict.keys()), dtype=np.int32)

    fl_testing_label_files_day = sorted(glob('/mnt/label_data/test_set_day/converted/SegmentationClass/*.npy'))
    fl_testing_label_files_night = sorted(glob('/mnt/label_data/test_set_night/converted/SegmentationClass/*.npy'))

    for label_day_file in fl_testing_label_files_day:
        label_day = np.load(label_day_file)

        for key in class_names_dict.keys():
            class_counter_day[key] += np.count_nonzero(label_day == key)


    for label__file in fl_testing_label_files_night:
        label_night = np.load(label__file)

        for key in class_names_dict.keys():
            class_counter_night[key] += np.count_nonzero(label_night == key)

    print('\n\n\nTESTING DATASET STATS\n\n\n')


    for key in class_names_dict.keys():
        n_pixels = class_counter_day[key]
        percentage = 100*float(class_counter_day[key])/np.sum(class_counter_day)
        print('DAY    {:20} has {:10d} pixels or {:05.2f}% of all DAY pixels.'.format(class_names_dict[key], n_pixels, percentage))


    for key in class_names_dict.keys():
        n_pixels = class_counter_night[key]
        percentage = 100*float(class_counter_night[key])/np.sum(class_counter_night)
        print('NIGHT  {:20} has {:10d} pixels or {:05.2f}% of all NIGHT pixels.'.format(class_names_dict[key], n_pixels, percentage))

    for key in class_names_dict.keys():
        n_pixels = class_counter_night[key] + class_counter_day[key]
        percentage = 100*float(class_counter_night[key] + class_counter_day[key]) / (np.sum(class_counter_night) + np.sum(class_counter_day))
        print('BOTH  {:20} has {:10d} pixels or {:05.2f}% of all pixels.'.format(class_names_dict[key], n_pixels, percentage))


if __name__ == '__main__':

    # get_testing_data_stats()
    get_training_data_stats()