import cv2
import numpy as np


def adjust_gamma(image, gamma=1.0):
    # 构建一个0到255的灰度值映射表
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # 应用映射表，返回调整后的图像
    return cv2.LUT(image, table)


# 加载图像
image = cv2.imread('./imgs/01dark.png')

# 调整亮度
adjusted = adjust_gamma(image, gamma=0.8)

# 显示原始图像和调整后的图像
cv2.imshow("Original", image)
cv2.imshow("Adjusted", adjusted)
cv2.waitKey(0)
