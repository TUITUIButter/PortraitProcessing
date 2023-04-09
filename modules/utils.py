import cv2
import math
import numpy
import torch
import os
import numpy as np
from matplotlib import pyplot as plt
from portraitNet.pred_img import portraitSeg


class Utils:

    # pose_net = cv2.dnn.readNetFromTensorflow('modules/graph_opt.pb')
    points = []
    BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                  "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                  "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                  "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}
    POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                  ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                  ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                  ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                  ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

    """
    获取19个关键点
    """
    @classmethod
    def get_pose_point(cls, img):
        # 获取图像的高度和宽度
        (h, w) = img.shape[:2]

        # 创建一个 blob 对象
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        # 将 blob 对象传递给模型
        cls.pose_net.setInput(blob)

        # 运行模型
        output = cls.pose_net.forward()

        points = []
        for i in range(len(cls.BODY_PARTS)):
            # Slice heatmap of corresponding body's part.
            heatMap = output[0, i, :, :]
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (w * point[0]) / output.shape[3]
            y = (h * point[1]) / output.shape[2]
            # Add a point if it's confidence is higher than threshold.
            points.append((int(x), int(y)) if conf > 0.2 else None)

        # 划线
        for pair in cls.POSE_PAIRS:
            part_from = pair[0]
            part_to = pair[1]
            assert (part_from in cls.BODY_PARTS)
            assert (part_to in cls.BODY_PARTS)

            id_from = cls.BODY_PARTS[part_from]
            id_to = cls.BODY_PARTS[part_to]

            if points[id_from] and points[id_to]:
                cv2.line(img, points[id_from], points[id_to], (0, 255, 0), 3)
                cv2.ellipse(img, points[id_from], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                cv2.ellipse(img, points[id_to], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

        height, width = img.shape[:2]
        # 设置窗口大小可调
        cv2.namedWindow("image", cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('image', width, height)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return points

    """
        使用Real-time的模型完成人像分离
    """
    @classmethod
    def separate_character(cls, imgFile):
        alphargb, human, background, human_blur_background = portraitSeg(imgFile)
        # 显示分割出的人像和背景
        cv2.imshow("human", human)
        cv2.imshow("background", background)
        cv2.imshow("human_blur_background", human_blur_background)
        cv2.waitKey(0)

if __name__ == '__main__':
    Utils.separate_character(imgFile="../imgs/middle-tilt.jpg")
