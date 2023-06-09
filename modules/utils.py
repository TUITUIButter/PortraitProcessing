import cv2
import math
import numpy
import torch
import os
import numpy as np
from matplotlib import pyplot as plt

abspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Utils:
    pose_net = cv2.dnn.readNetFromTensorflow(os.path.join(abspath, 'modules', 'graph_opt.pb'))
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

        # # 划线
        # for pair in cls.POSE_PAIRS:
        #     part_from = pair[0]
        #     part_to = pair[1]
        #     assert (part_from in cls.BODY_PARTS)
        #     assert (part_to in cls.BODY_PARTS)
        #
        #     id_from = cls.BODY_PARTS[part_from]
        #     id_to = cls.BODY_PARTS[part_to]
        #
        #     if points[id_from] and points[id_to]:
        #         cv2.line(img, points[id_from], points[id_to], (0, 255, 0), 3)
        #         cv2.ellipse(img, points[id_from], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
        #         cv2.ellipse(img, points[id_to], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

        # height, width = img.shape[:2]
        # # 设置窗口大小可调
        # cv2.namedWindow("image", cv2.WINDOW_KEEPRATIO)
        # cv2.resizeWindow('image', width, height)
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return points

    """
        亮度检测
    """
    @classmethod
    def brightness_detection(cls, img):

        # 把图片转换为灰度图
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 获取灰度图矩阵的行数和列数
        r, c = gray_img.shape[:2]

        piex_sum = r * c  # 整个图的像素个数为r*c

        bright_sum_n = gray_img > 200
        dark_sum_n = gray_img < 50

        bright_sum = np.sum(bright_sum_n)
        dark_sum = np.sum(dark_sum_n)

        dark_prop = dark_sum / piex_sum
        bright_prop = bright_sum / piex_sum
        print("dark_sum: {:.3f}".format(dark_sum))
        print("bright_sum: {:.3f}".format(bright_sum))
        print("piex_sum: {:.3f}".format(piex_sum))
        print("dark_prop=dark_sum/piex_sum: {:.3f}".format(dark_prop))
        print("bright_prop=bright_sum/piex_sum: {:.3f}".format(bright_prop))
        if dark_prop >= 0.6:  # 人为设置的超参数:表示若偏暗像素所占比例超过0.6,则这张图被认为整体环境黑暗的图片
            print("dark!")
        elif bright_prop >= 0.12:
            print("bright!")
        else:
            print("fine!")

        return dark_prop, bright_prop

    # 用于显示图片的灰度直方图
    # @classmethod
    # def hist(cls, img):
    #     # img = cv2.imread(pic_path, 0)
    #     # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    #     plt.subplot(121)
    #     plt.imshow(img, 'gray')
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.title("Original")
    #     plt.subplot(122)
    #     plt.hist(img.ravel(), 256, [0, 256])
    #     plt.show()


if __name__ == '__main__':
    # Utils.separate_character(imgFile="../imgs/middle-tilt.jpg")
    # 可以测试01，01dark，02，02dark，03，03bright
    Utils.brightness_detection(cv2.imread("../imgs/03.jpg"))
    # 使用NIMA评估图像的质量
    mean, std = Utils.eval_pic("../imgs/03.jpg")
    print(' mean: %.3f | std: %.3f' % (mean, std))
