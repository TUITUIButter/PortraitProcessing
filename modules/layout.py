import math

import numpy
from .base_module import BaseModule
from .utils import Utils
import cv2


class Layout(BaseModule):
    def __init__(self, path='modules/graph_opt.pb'):
        # 加载模型
        self.net = cv2.dnn.readNetFromTensorflow(path)

    def cal_score(self, img) -> float:
        self.get_pose_single(img)
        return 0

    def opt_img(self, img) -> numpy.ndarray:
        res = [1, 2, 3]
        return numpy.array(res)
        pass


    def get_pose_single(self, img):
        points = Utils.get_pose_point(img)

        # 考虑将哪两个点(多种选择)之间的连线作为人体的连线
        nose_point = points[Utils.BODY_PARTS["Nose"]]
        lHip_point = points[Utils.BODY_PARTS["LHip"]]
        rHip_point = points[Utils.BODY_PARTS["RHip"]]
        if not nose_point or (not lHip_point and not rHip_point):  # 没有鼻子或者没有左右胯都没有
            print("无法计算人体倾斜的角度")
            return 7

        x_nose = nose_point[0]
        y_nose = nose_point[1]
        if lHip_point and rHip_point:
            x_hip = (lHip_point[0] + rHip_point[0]) / 2
            y_hip = (lHip_point[1] + rHip_point[1]) / 2
        else:
            x_hip = lHip_point[0] if lHip_point else rHip_point[0]
            y_hip = lHip_point[1] if lHip_point else rHip_point[1]

        # 返回上面角的度数
        angle = math.atan2(abs(x_nose - x_hip), abs(y_nose - y_hip)) * 180 / math.pi
        print("angle: {:.2f}°".format(angle))
