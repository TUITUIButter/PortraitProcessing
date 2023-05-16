from modules.layout import Layout
from modules.brightness import Brightness
from modules.saturation import Saturation
from modules.utils import Utils
from SCUNet.scunet import Denoising
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

# 初始化各个模块，加载模型
layout = Layout('./modules/graph_opt.pb')
brightness = Brightness()
saturation = Saturation()
denoising = Denoising()


layout_score = layout.cal_score(image)  # 布局得分
brightness_score = brightness.cal_score(image)  # 亮度得分
saturation_score = saturation.cal_score(image)  # 饱和度得分
ai_score = float(Utils.eval_pic(path)[0][0])  # AI打分，获取第一个值并转成float类型
total_score = layout_score + brightness_score + saturation_score

print('total_score', total_score)
print('AI Score', ai_score)

# 亮度
adjusted = brightness.opt_img(image)  # 调整亮度
adjusted = saturation.opt_img(adjusted)  # 调整饱和度

# 人物分离与背景虚化
alphargb, human, background, human_blur_background = Utils.separate_character(imgFile=None, imgOri=adjusted)
adjusted = human_blur_background

denoising.run(adjusted)  # 最后一步去噪，回保存到res.jpg


# cv2.namedWindow('Original', cv2.WINDOW_KEEPRATIO)    # 窗口大小可以改变
# cv2.imshow("Original", image)
# cv2.namedWindow('Adjusted', cv2.WINDOW_KEEPRATIO)    # 窗口大小可以改变
# cv2.imshow("Adjusted", adjusted)
# cv2.waitKey(0)
