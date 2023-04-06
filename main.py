from modules.layout import Layout
import cv2

# image = cv2.imread('./imgs/test.png')  # angle: 5.57°
image = cv2.imread("./imgs/middle-tilt.jpg")  # 倾斜的图片 angle: 25.91°
# image = cv2.imread("./imgs/02.jpg")  # angle: 5.38°
# image = cv2.imread("./imgs/good-001.JPG")  # 找不到LHip这个点
# image = cv2.imread("./imgs/good-002.JPG")  # 背影无法定位
# image = cv2.imread("./imgs/good-010.JPEG")  # angle: 10.78°
# image = cv2.imread("imgs/bad-001.JPG")  # angle: 3.59°
assert image is not None
layout = Layout('./modules/graph_opt.pb')

layout.cal_score(image)
