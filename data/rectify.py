import rosbag
from sensor_msgs.msg import Image
import numpy as np
import cv2
from cv2 import fisheye
import tf
from tf_bag import BagTfTransformer
import Queue
from enum import Enum

class CameraSelect(Enum):
    LEFT = 1
    RIGHT = 2

class Undistorter:
    def __init__(self):
        self.c_m_0 = np.zeros((3,3))
        self.c_m_1 = np.zeros((3, 3))
        self.dist_0 = np.zeros((4,1))
        self.dist_1 = np.zeros((4,1))
        self.R = np.zeros((3,3))
        self.T = np.zeros((3,1))
        self.R_1 = np.zeros((3,3))
        self.R_2 = np.zeros((3, 3))
        self.P_1 = np.zeros((3, 4))
        self.P_2 = np.zeros((3, 4))

    def prepare(self):
        self.c_m_0 = np.matrix(
            [[1010.7234313264403, 0, 1002.9629088985278], [0, 1009.1829722092845, 358.2714637354781], [0, 0, 1]])
        self.dist_0 = np.array([0.09591790561780944, 0.12373563338566723, -0.0975544797256392, 0.11613597508133179])

        self.c_m_1 = np.matrix(
            [[1011.2044423890844, 0, 1001.6375234436252], [0, 1009.7482878445264, 427.7572918140741], [0, 0, 1]])
        self.dist_1 = np.array([0.09155160663130679, 0.13854249300660224, -0.12766992970542612, 0.13681343069404775])

        self.R = np.matrix([[0.9998805919777298, -0.014374203502396918, 0.005673099676157448],
                       [0.014425596660187882, 0.9998543112732692, -0.009124603512131731],
                       [-0.005541114261726946, 0.009205351809013641, 0.9999422771094391]])
        self.T = np.array([-0.5011739875266582, -0.00853806222837021, 0.0024935075413970646])

        # self.R = np.eye(3,3)
        # self.T = np.zeros((3,1))

        self.R1, self.R2, self.P1, self.P2, Q = fisheye.stereoRectify(self.c_m_0, self.dist_0, self.c_m_1, self.dist_1, (2048, 1136), self.R, self.T, flags=cv2.CALIB_ZERO_DISPARITY)

    def undistort_rectify(self, img_left, select):
        if(select == CameraSelect.LEFT):
            mapx, mapy = fisheye.initUndistortRectifyMap(self.c_m_0, self.dist_0, self.R1, self.P1, (2048, 1136), cv2.CV_32F)
        elif(select == CameraSelect.RIGHT):
            mapx, mapy = fisheye.initUndistortRectifyMap(self.c_m_1, self.dist_1, self.R2, self.P2, (2048, 1136), cv2.CV_32F)
        img_rect = cv2.remap(img_left, mapx, mapy, cv2.INTER_LINEAR)

        return img_rect
