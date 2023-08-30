import sys

import rosbag
from tf_bag import BagTfTransformer
import rospy
import tf
from sensor_msgs.msg import Image
import lidar_to_numpy
import numpy as np
import cv2
from cv2 import fisheye
import Queue
from enum import Enum
import yaml
import gmplot
import utm
import datetime
import os.path
import glob, os


# Requirement: https://github.com/IFL-CAMP/tf_bag
# This program dumps sequences of images and motion from rosbags.c
# The program will also generate txt files with odometry and paths to 3-image sequence (first path is earliest frame)

save_dir = "/home/vertensj/Documents/robocar_bags/dumped/"      # Directory to save images and motion information
date = "test2"                                  # Name of the folder which gets saved
base_path = "/home/vertensj/Documents/robocar_bags/data_collection_new/lidar2/" # Path to folder which holds the rosbag of a specific ride
front_stereo_calib_path = "/home/vertensj/software/thermal_seg/data/calibrations/front_stereo_05_08_19/front_stereo_calibration.yaml"
back_stereo_calib_path = "/home/vertensj/software/thermal_seg/data/calibrations/back_stereo_05_08_19/back_stereo_calibration.yaml"
thermal_stereo_calib_path = "/home/vertensj/software/thermal_seg/data/calibrations/thermal_29_07_19/thermal_stereo_calib.yaml"
queue_size = 5
fps = 10
dump_lidar = True  # Note that the lidar pointclouds are recorded with different frequencies, a pointcloud will never
                   # match the timestamp of a image exactly TODO: Transform the pointclouds according to the odometry

class CameraSelect(Enum):
    LEFT = 1
    RIGHT = 2


class Undistorter:
    def __init__(self, cal_file):
        self.c_m_0 = np.zeros((3, 3))
        self.c_m_1 = np.zeros((3, 3))
        self.dist_0 = np.zeros((4, 1))
        self.dist_1 = np.zeros((4, 1))
        self.R = np.zeros((3, 3))
        self.T = np.zeros((3, 1))
        self.R_1 = np.zeros((3, 3))
        self.R_2 = np.zeros((3, 3))
        self.P_1 = np.zeros((3, 4))
        self.P_2 = np.zeros((3, 4))

        stream = file(cal_file, 'r')
        self.c_file = yaml.load(stream)

    def prepare(self):

        lp = self.c_file["left"]
        rp = self.c_file["right"]

        self.c_m_0 = np.matrix(
            [[lp["intrinsics"][0], 0, lp["intrinsics"][2]], [0, lp["intrinsics"][1], lp["intrinsics"][3]], [0, 0, 1]])
        self.dist_0 = np.array([lp["distortion_coeffs"][0], lp["distortion_coeffs"][1], lp["distortion_coeffs"][2],
                                lp["distortion_coeffs"][3]])

        self.c_m_1 = np.matrix(
            [[rp["intrinsics"][0], 0, rp["intrinsics"][2]], [0, rp["intrinsics"][1], rp["intrinsics"][3]], [0, 0, 1]])
        self.dist_1 = np.array([rp["distortion_coeffs"][0], rp["distortion_coeffs"][1], rp["distortion_coeffs"][2],
                                rp["distortion_coeffs"][3]])

        self.R = np.matrix(rp["T_cn_cnm1"])[0:3, 0:3]
        self.T = np.array(np.matrix(rp["T_cn_cnm1"])[0:3, 3])
        self.resolution = tuple(lp["resolution"])

        self.R1, self.R2, self.P1, self.P2, Q, _, _ = cv2.stereoRectify(self.c_m_0, self.dist_0, self.c_m_1, self.dist_1,
                                                                      self.resolution, self.R, self.T,
                                                                      flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)


    def undistort_rectify(self, img_left, select, interpolation=cv2.INTER_LINEAR):
        if (select == CameraSelect.LEFT):
            mapx, mapy = cv2.initUndistortRectifyMap(self.c_m_0, self.dist_0, self.R1, self.P1, self.resolution,
                                                         cv2.CV_32F)
        elif (select == CameraSelect.RIGHT):
            mapx, mapy = cv2.initUndistortRectifyMap(self.c_m_1, self.dist_1, self.R2, self.P2, self.resolution,
                                                         cv2.CV_32F)
        img_rect = cv2.remap(img_left, mapx, mapy, interpolation)

        return img_rect


def msg_to_mat(msg, debayer=False, thermal=False):
    if thermal:
        image_decode = np.fromstring(msg.data, dtype=np.uint16).reshape((512, 640))

    else:
        np_img = np.fromstring(msg.data, dtype=np.uint8).reshape((650, 1920))

        # For compressed images:
        # np_img = np.fromstring(msg.data, dtype=np.uint8)
        # image_decode = cv2.imdecode(np_img, cv2.IMREAD_UNCHANGED)

    if debayer and not thermal:
        bgr = np.zeros((np_img.shape[0], np_img.shape[1], 3), dtype=np.uint8)
        cv2.cvtColor(np_img, cv2.COLOR_BAYER_BG2BGR, bgr, 3)
        return bgr
    else:
        return image_decode


class DataQueue:
    def __init__(self, topic_lists, queue_size=5):
        self.queue_size = queue_size
        self.topics = topic_lists
        self.data = {key: Queue.Queue(queue_size) for (key) in topic_lists}

    def add_to_queue(self, topic, msg):
        if self.data[topic].qsize() == self.queue_size:
            self.data[topic].get()
        self.data[topic].put(msg)

    def getPreceeding(self, topic, index=0):
        return self.data[topic].queue[index]

    def getLength(self, topic):
        return self.data[topic].qsize()

    def isFull(self, topic):
        return self.data[topic].qsize() == self.queue_size


class Synchronizer:
    def __init__(self, topic_lists, queue_size=11):
        self.queue_size = queue_size
        self.topics = topic_lists
        self.data = {key: Queue.Queue(queue_size) for (key) in topic_lists}
        self.times = {key: Queue.Queue(queue_size) for (key) in topic_lists}

    def add_to_queue(self, topic, msg, stamp):
        if self.data[topic].qsize() == self.queue_size:
            self.data[topic].get()
        if self.times[topic].qsize() == self.queue_size:
            self.times[topic].get()
        self.data[topic].put(msg)
        self.times[topic].put(stamp)

        # print "Added msg to queue"

    # Searches topics in the queue which are nearest to the reference topic
    # Output: Nearest Data per topic, Timestamp corresponding to the data
    def check_synced(self):
        ref_topic = self.topics[0]
        if self.data[ref_topic].qsize() == self.queue_size:
            ref_time = self.times[ref_topic].queue[self.queue_size / 2]
            best_indexes = -np.ones((len(self.topics) - 1))
            # Search for nearest time in other queues
            for topic_index, i_t in enumerate(self.topics):
                # if not self.data[topic_index].qsize() == self.queue_size:
                #     continue
                if ref_topic == i_t:
                    continue
                best_time = -1
                best_index = -1
                for index, i_time in enumerate(self.times[i_t].queue):
                    dt = abs(ref_time - i_time).nsecs / 1000000.0
                    # DT is in milliseconds!
                    if best_time == -1:
                        best_time = dt

                    if (dt < best_time):
                        best_time = dt
                        best_index = index

                if best_index != -1:
                    best_indexes[topic_index - 1] = best_index
                else:
                    print('Warning: No near index found for time synchronization')
                    return None, None

            out = []
            times = []
            out.append(self.data[ref_topic].queue[self.queue_size / 2])
            times.append(ref_time)
            for topic_index, i_t in enumerate(self.topics):
                if ref_topic == i_t:
                    continue
                out.append(self.data[i_t].queue[int(best_indexes[topic_index - 1])])
                times.append(self.times[i_t].queue[int(best_indexes[topic_index - 1])])

            return out, times
        else:
            return None, None


def checkFramerate(queue, topic, threshold_ns=43e06):
    max_td = rospy.Duration()
    for i_idx in range(0, queue.getLength(topic) - 1):
        image = queue.getPreceeding(topic, i_idx)
        image_p1 = queue.getPreceeding(topic, i_idx + 1)
        td_i = abs(image_p1.header.stamp - image.header.stamp)
        if td_i > max_td:
            max_td = td_i
    if max_td < rospy.Duration(nsecs=threshold_ns):
        return True
    else:
        print("Time difference is: %f seconds" % max_td.to_sec())
        return False


def generateName(prefix, stamp, seq_nr):
        return prefix + "_"  + str(stamp.secs) + "_" + str(stamp.nsecs) + "" + str(seq_nr) + ".png"


def createFolder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print('Created %s' % dir)


def checkTransforms(left_images, transformer):
    consistent = True
    translations = []
    quaternions = []
    time_stamps = []
    for im_id in range(len(left_images)):
        time_stamp_pair = left_images[im_id].header.stamp
        time_stamps.append(time_stamp_pair)
        translation_pair, quaternion_pair = transformer.lookupTransform('odom_combined', 'base_link',
                                                                        time_stamp_pair)
        if translation_pair is None or quaternion_pair is None:
            consistent = False
        else:
            translations.append(translation_pair)
            quaternions.append(quaternion_pair)

    if consistent:
        return translations, quaternions, time_stamps
    else:
        return tuple((None, None, None))


class PC_Parser:
    def __init__(self, bag_file, tf_bag, topics, prefixes, core_dir, cal_files=None, debayer=False, thermal=False, dump_lidar=False):
        self.origin = np.zeros((3))
        self.init_time = -1
        self.init_origin = False
        self.heat_lat = []
        self.heat_long = []
        self.pre_translation = np.ones((3)) * 100
        self.pre_time = rospy.Time()
        self.dir_init = False
        self.save_dir = ""
        self.center_of_grid = utm.from_latlon(48.013551, 7.833116)
        self.cal_files = cal_files
        self.debayer = debayer
        self.thermal = thermal
        self.dump_lidar = dump_lidar

        self.prefixes = prefixes
        self.core_dir = core_dir

        self.bag = rosbag.Bag(bag_file)
        self.tf_bag = rosbag.Bag(tf_bag)
        self.bag_transformer = BagTfTransformer(self.tf_bag)
        # print(self.bag_transformer.getTransformGraphInfo())

        self.sync1 = Synchronizer(topics)
        self.queue = DataQueue(topics, queue_size=queue_size)

        self.topics = topics

        self.origin_dir = None
        self.time_str = None

        # self.gmap = gmplot.GoogleMapPlotter.from_geocode("Freiburg im Breisgau")
        self.gmap = gmplot.GoogleMapPlotter(48.013551, 7.833116, 16)

        createFolder(core_dir)

        self.undistorters =[]
        if self.cal_files is not None:
            for cf in self.cal_files:
                undistorter = Undistorter(cf)
                undistorter.prepare()
                self.undistorters.append(undistorter)

    def writeOrigin(self):
        if self.origin_dir is not None and self.time_str is not None:
            with open(self.origin_dir + "drive_" + self.time_str + "_origin.txt", "a") as p_file:
                p_file.write(str(self.origin[0]) + "/" + str(self.origin[1]) + "/" + str(self.origin[2]) + "\n")

    def init_tf(self):
        for topic, msg, t in self.tf_bag.read_messages(topics=['/applanix/origin']):
            if topic == '/applanix/origin' and not self.init_origin:
                # Initialize the origin transformation which is the starting utm coordinate of the applanix system
                self.origin[0] = msg.position.x
                self.origin[1] = msg.position.y
                self.origin[2] = msg.position.z
                self.init_origin = True

                break

    def topic_is_image(self, topic_name):
        image_topic_names = ['/rgb_fl_burst', '/rgb_fr_burst', '/ir_left_burst', '/ir_right_burst']
        if any(x in topic_name for x in image_topic_names):
            return True
        else:
            return False

    def topic_is_lidar(self, topic_name):
        lidar_topic_names = ['/top_velodyne', '/left_puck', '/right_puck']
        if any(x in topic_name for x in lidar_topic_names):
            return True
        else:
            return False

    def start(self):
        print(self.topics)
        for topic, msg, t in self.bag.read_messages(
                topics=self.topics):
            print(topic)
            if topic == self.topics[0]:

                # Initialization phase: Build folders, infer daytime, check if parsed already
                if not self.dir_init:

                    self.time_str = datetime.datetime.fromtimestamp(msg.header.stamp.secs).strftime('%Y_%m_%d_%H_%M_%S')

                    # Differentiate between day and night drives
                    hour = datetime.datetime.fromtimestamp(msg.header.stamp.secs).hour

                    if hour >= 22 or hour < 6:
                        daytime = 'night'
                    else:
                        daytime = 'day'

                    save_dir = self.core_dir + "drive_" + daytime + "_" + self.time_str + "/"

                    self.save_dirs = []
                    for p in self.prefixes:
                        self.save_dirs.append(save_dir + "/" + p  + "/")

                    if os.path.isdir(self.save_dirs[-1]):
                        print("File is already parsed")
                        return

                    for p in self.save_dirs:
                        createFolder(p)

                    vehicle_dir = save_dir + "/vehicle/"
                    paths_dir = save_dir + "/paths/"
                    self.origin_dir = save_dir + "/origin/"

                    createFolder(vehicle_dir)
                    createFolder(paths_dir)
                    createFolder(self.origin_dir)

                    self.dir_init = True

                self.sync1.add_to_queue(topic, msg, msg.header.stamp)
                synced_image, synced_times = self.sync1.check_synced()
                if synced_image is not None and True:

                    time_stamp = synced_times[0] if synced_times[0] < synced_times[1] else synced_times[1]
                    td = time_stamp - self.pre_time

                    for i in range(len(self.topics)):
                        self.queue.add_to_queue(self.topics[i], synced_image[i])

                    # At least 0.2 seconds
                    if (td.nsecs > 0):

                        if self.init_time == -1:
                            self.init_time = time_stamp.secs
                        else:
                            passed_time = time_stamp.secs - self.init_time
                            if passed_time % 30 == 0:
                                print("Time passed: %f minutes" % ((time_stamp.secs - self.init_time)/60))
                        self.pre_time = time_stamp

                        # translation, quaternion = self.bag_transformer.lookupTransform('odom_combined', 'base_link',
                        #                                                                time_stamp)
                        # if translation is None or quaternion is None:
                        #     print("Failed to get transformation!")
                        #     continue
                        #
                        # translation = np.array(translation)
                        # diff = translation - self.pre_translation
                        #
                        # translation_utm = translation + self.origin
                        # lat_long = utm.to_latlon(translation_utm[0], translation_utm[1], 32, 'U')

                        #  Check if preceding frames are available and if timestamps and transforms are consistent
                        if self.queue.isFull(self.topics[0]):
                            # Check time consistency for first topic
                            if checkFramerate(self.queue, self.topics[0]):
                                # All stamps are valid

                                # Calculate section
                                # section_x = int((translation_utm[0] - self.center_of_grid[0]) / 500)
                                # section_y = int((translation_utm[1] - self.center_of_grid[1]) / 500)
                                section_x = 0
                                section_y = 0

                                # Get transformation string and timestamps for first topic
                                topic_data = []
                                for i_i in range(queue_size):
                                    topic_data.append(self.queue.getPreceeding(self.topics[0], i_i))

                                # transformation_string = ""
                                # # Check transformations for consistency and get time stamps
                                # image_translation, image_quaternions, image_timestamps = checkTransforms(topic_data,
                                #                                                                          self.bag_transformer)
                                #
                                # # Write out tf-transformation
                                # if image_translation is not None:
                                #     for im_id in range(len(topic_data)):
                                #         time_stamp_pair = image_timestamps[im_id]
                                #         translation_pair = image_translation[im_id]
                                #         quaternion_pair = image_quaternions[im_id]
                                #         transformation_string += str(translation_pair[0]) + " " + str(
                                #             translation_pair[1]) + " " + \
                                #                                  str(translation_pair[2]) + " " + str(
                                #             quaternion_pair[0]) + " " + \
                                #                                  str(quaternion_pair[1]) + " " + str(
                                #             quaternion_pair[2]) + " " + \
                                #                                  str(quaternion_pair[3]) + " / "
                                #
                                #     with open(vehicle_dir + self.prefixes[0] + "_drive_" + self.time_str + "_" + str(
                                #             section_x) + "-" + str(section_y) + "_vehicle.txt", "a") as p_file:
                                #         p_file.write(transformation_string + "\n")

                                # Save all topics

                                for t_i in range(len(self.topics)):
                                    topic_data = []
                                    for i_i in range(queue_size):
                                        topic_data.append(self.queue.getPreceeding(self.topics[t_i], i_i))

                                    paths = []

                                    # Save only if transformation is available
                                    # if image_translation is not None:
                                    if True:
                                        for im_id in range(len(topic_data)):
                                            image_filename = self.save_dirs[t_i] + generateName(self.prefixes[t_i],
                                                                                                rospy.Time(0), im_id)

                                            if self.topic_is_image(self.topics[t_i]):

                                                lr_switch = CameraSelect.LEFT if ("fl" in self.topics[t_i] or "bl" in self.topics[t_i] or "left" in self.topics[t_i]) else CameraSelect.RIGHT

                                                paths.append(image_filename)
                                                if not os.path.isfile(image_filename):
                                                    if self.cal_files[t_i] is not None:
                                                        rect = self.undistorters[t_i].undistort_rectify(
                                                            msg_to_mat(topic_data[im_id], self.debayer, "ir" in self.topics[t_i]),
                                                            lr_switch)
                                                        cv2.imwrite(image_filename, rect)
                                                    else:
                                                        cv2.imwrite(image_filename,
                                                                    msg_to_mat(topic_data[im_id], self.debayer, "ir" in self.topics[t_i]))
                                            elif self.topic_is_lidar(self.topics[t_i]):
                                                np_lidar = lidar_to_numpy.msg_to_arr(topic_data[im_id])
                                                np.save(image_filename, np_lidar)

                                    # Generate path sequences with section division.
                                    # A section divides the world into 500*500m tiles
                                    with open(paths_dir + self.prefixes[t_i] + "_drive_" + self.time_str + "_" + str(
                                            section_x) + "-" + str(section_y) + ".txt", "a") as p_file:
                                        str_paths = ""
                                        for p in paths:
                                            str_paths = str_paths + p + " "
                                        p_file.write(str_paths + "\n")

                                else:
                                    print("Transforms are inconsistent")

                            else:
                                print("Time difference too high")

                        # Add point to google maps every 10 meters
                        # mag = np.sqrt(diff.dot(diff))
                        # if mag > 10.0:
                        #     self.heat_lat.append(lat_long[0])
                        #     self.heat_long.append(lat_long[1])
                        #     self.pre_translation = translation

            elif topic in self.topics:
                self.sync1.add_to_queue(topic, msg, msg.header.stamp)

        print("Generating gmaps heatmap with: %d entries" % len(self.heat_lat))

        # Write heatmap of data occurences in google maps
        self.gmap.heatmap(self.heat_lat, self.heat_long)
        self.gmap.draw("/home/vertensj/heatmaps.html")
        self.bag.close()

        return self.heat_lat, self.heat_long


def saveHeatStats(lat, long, filename):
    with open(filename, "a") as p_file:
        for i in range(len(lat)):
            p_file.write(str(lat[i]) + " " + str(long[i]) + "\n")


def parseListofFiles(base_path, save_path, date):
    # Parse rosbags#####################################################################################################
    cameras_lasers = []
    os.chdir(base_path + "camera_lasers/")
    for file in glob.glob("*.bag"):
        cameras_lasers.append(base_path + "camera_lasers/" + file)
    cameras_lasers.sort()

    core_dir = save_path + date + "/"

    for bag_id in range(len(cameras_lasers)):
        print("Processing bag id nr : %d of %d" % (bag_id, len(cameras_lasers)))

        tf_bag = cameras_lasers[bag_id]

        # Front Stereo Pair
        # topics = ['/rgb_fl_burst', '/rgb_fr_burst', '/ir_left_burst', '/ir_right_burst']
        topics = ['/rgb_fl_burst', '/rgb_fr_burst']

        cal_files = [front_stereo_calib_path, front_stereo_calib_path, thermal_stereo_calib_path, thermal_stereo_calib_path]
        # prefixes = ['fl_rgb', 'fr_rgb', 'fl_ir', 'fr_ir']
        prefixes = ['fl_rgb', 'fr_rgb']

        if dump_lidar:
            topics.extend(['/top_velodyne/points'])
            prefixes.extend(['v64'])

        image_bag = cameras_lasers[bag_id]
        parser = PC_Parser(image_bag, tf_bag, topics, prefixes, core_dir, cal_files, True, dump_lidar=dump_lidar)
        parser.init_tf()
        parser.start()
        parser.writeOrigin()


# Start parsing folder with rosbags
parseListofFiles(base_path, save_dir, date)
