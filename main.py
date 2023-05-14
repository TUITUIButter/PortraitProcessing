from modules.layout import Layout
from modules.brightness import Brightness
from modules.saturation import Saturation
from modules.utils import Utils
import cv2

path = "./imgs/middle-tilt.jpg"  # 倾斜的图片 angle: 25.91°
# path = './imgs/test.png'  # angle: 5.57°
# path = "./imgs/02dark.png"  # angle: 5.38°
# path = "./imgs/cloudy.jpg"  # angle: 5.38°
# path = "./imgs/good-001.JPG"  # 找不到LHip这个点
# path = "./imgs/good-002.JPG"  # 背影无法定位
# path = "./imgs/good-010.JPEG"  # angle: 10.78°
# path = "imgs/bad-001.JPG"  # angle: 3.59°

image = cv2.imread(path)  # 倾斜的图片 angle: 25.91°
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
saturation_score = saturation.cal_score(image)  # 饱和度得分
adjusted = saturation.opt_img(adjusted)  # 调整饱和度

# TODO 添加抠图与虚化，虚化效果小一点
# 人物分离与背景虚化
alphargb, human, background, human_blur_background = Utils.separate_character(imgFile=None, imgOri=adjusted)
adjusted = human_blur_background

total_score = layout_score + brightness_score + saturation_score

# TODO 加上AI打分
ai_score = float(Utils.eval_pic(path)[0][0])  # 获取第一个值并转成float类型


print('total_score', total_score)
print('AI Score', ai_score)

cv2.namedWindow('Original', cv2.WINDOW_KEEPRATIO)    # 窗口大小可以改变
cv2.imshow("Original", image)
cv2.namedWindow('Adjusted', cv2.WINDOW_KEEPRATIO)    # 窗口大小可以改变
cv2.imshow("Adjusted", adjusted)
cv2.waitKey(0)
