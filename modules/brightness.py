import numpy
from .utils import Utils
from .base_module import BaseModule
import numpy as np
import cv2


class Brightness(BaseModule):
    def __init__(self):
        self.dark_prop, self.bright_prop = 0, 0
        self.dark_threshold = 0.35
        self.bright_threshold = 0.1

    def cal_score(self, img) -> float:
        self.dark_prop, self.bright_prop = Utils.brightness_detection(img)
        score = 10
        if self.dark_prop >= self.dark_threshold:  # 整体环境黑暗的图片
            score -= (self.dark_prop - self.dark_threshold) / (1 - self.dark_threshold) * 10.0
        elif self.bright_prop >= self.bright_threshold:
            score -= (self.bright_prop - self.bright_threshold) / (1 - self.bright_threshold) * 10.0

        print("brightness score: {:.2f}".format(score))
        return score

    def opt_img(self, img) -> numpy.ndarray:
        if self.dark_prop >= self.dark_threshold:  # 整体环境黑暗的图片
            gamma = self.dark_prop - self.dark_threshold + 1.4
        elif self.bright_prop >= self.bright_threshold:
            gamma = 0.8 - (self.bright_prop - self.bright_threshold)
        else:
            gamma = 1
        print("adjust gamma:" + str(gamma))
        img = adjust_gamma(img, gamma)
        return img


def adjust_gamma(image, gamma=1.0):
    # 构建一个0到255的灰度值映射表
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # 应用映射表，返回调整后的图像
    return cv2.LUT(image, table)
