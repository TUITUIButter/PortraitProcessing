from modules.layout import Layout
from modules.brightness import Brightness
from modules.saturation import Saturation
import cv2

# image = cv2.imread('./imgs/test.png')  # angle: 5.57°
# image = cv2.imread("./imgs/middle-tilt.jpg")  # 倾斜的图片 angle: 25.91°
# image = cv2.imread("./imgs/02dark.png")  # angle: 5.38°
image = cv2.imread("./imgs/cloudy.jpg")  # angle: 5.38°
# image = cv2.imread("./imgs/good-001.JPG")  # 找不到LHip这个点
# image = cv2.imread("./imgs/good-002.JPG")  # 背影无法定位
# image = cv2.imread("./imgs/good-010.JPEG")  # angle: 10.78°
# image = cv2.imread("imgs/bad-001.JPG")  # angle: 3.59°
assert image is not None

# 布局
layout = Layout('./modules/graph_opt.pb')
layout_score = layout.cal_score(image)  # 布局得分

# 亮度
brightness = Brightness()
brightness_score = brightness.cal_score(image)  # 亮度得分
adjusted = brightness.opt_img(image)  # 调整亮度

# 饱和度
saturation = Saturation()
saturation.cal_score(adjusted)  # 饱和度得分
adjusted = saturation.opt_img(adjusted)  # 调整饱和度


cv2.namedWindow('Original', cv2.WINDOW_KEEPRATIO)    # 窗口大小可以改变
cv2.imshow("Original", image)
cv2.namedWindow('Adjusted', cv2.WINDOW_KEEPRATIO)    # 窗口大小可以改变
cv2.imshow("Adjusted", adjusted)
cv2.waitKey(0)
