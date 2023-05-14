import numpy
from .utils import Utils
from .base_module import BaseModule
import numpy as np
import cv2


class Saturation(BaseModule):

    def __init__(self):
        self.low_threshold = 0.4
        self.high_threshold = 0.6
        self.contrast = 0.5

    def cal_score(self, img) -> float:

        # 直方图均衡化
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_equalized = cv2.equalizeHist(gray)

        # 计算对比度
        contrast = np.std(img_equalized) / np.mean(img_equalized)

        self.contrast = contrast

        print('contrast: ', contrast)

        # 判断对比度高低
        if contrast < self.low_threshold:
            print('图像对比度较低')
            return 10 - (self.low_threshold - contrast) * 10
        elif contrast > self.high_threshold:
            print('图像对比度较高')
            return 10 - (contrast - self.high_threshold) * 10
        else:
            print('图像对比度适中')
            return 10

    # 修改图像的饱和度,saturation_scale>0, <1降低对比度,>1提升对比度 建议0-2
    def opt_img(self, img) -> numpy.ndarray:
        if self.contrast < self.low_threshold:
            saturation_scale = 1 + (self.low_threshold - self.contrast)
        elif self.contrast > self.high_threshold:
            saturation_scale = 1 - (self.contrast - self.high_threshold)
        else:
            return img

        img = change_contrast(img, saturation_scale)

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation_mask = np.ones_like(hsv_img[:, :, 1]) * saturation_scale
        saturation_mask = saturation_mask.astype(np.float32)

        hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1] * saturation_mask, 0, 255)

        result_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        return result_img


# 修改图像的对比度,coefficent>0, <1降低对比度,>1提升对比度 建议0-2
def change_contrast(img, coefficent):
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = cv2.mean(img)[0]
    graynew = m + coefficent * (imggray - m)
    img1 = np.zeros(img.shape, np.float32)
    k = np.divide(graynew, imggray, out=np.zeros_like(graynew), where=imggray != 0)
    img1[:, :, 0] = img[:, :, 0] * k
    img1[:, :, 1] = img[:, :, 1] * k
    img1[:, :, 2] = img[:, :, 2] * k
    img1[img1 > 255] = 255
    img1[img1 < 0] = 0
    return img1.astype(np.uint8)
