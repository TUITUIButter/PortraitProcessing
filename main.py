from modules.layout import Layout
from modules.brightness import Brightness
from modules.saturation import Saturation
from modules.utils import Utils
from SCUNet.scunet import Denoising
from Neural_IMage_Assessment.NIMA import NIMA_predict
from portraitNet.pred_img import PortraitSeg
import cv2
import time

# path = "./imgs/middle-tilt.jpg"  # 倾斜的图片 angle: 25.91°
# path = './imgs/test.png'  # angle: 5.57°
# path = "./imgs/02.jpg"  # angle: 5.38°
# path = "./imgs/cloudy.jpg"  # angle: 5.38°
# path = "./imgs/good-001.JPG"  # 找不到LHip这个点
# path = "./imgs/good-002.JPG"  # 背影无法定位
# path = "./imgs/good-010.JPEG"  # angle: 10.78°
# path = "imgs/bad-001.JPG"  # angle: 3.59°
path = "imgs/09.jpg"

image = cv2.imread(path)  # 倾斜的图片 angle: 25.91°
assert image is not None

# 初始化各个模块，加载模型
layout = Layout('./modules/graph_opt.pb')
brightness = Brightness()
saturation = Saturation()
denoising = Denoising()
NIMA_predict = NIMA_predict()
portraitSeg = PortraitSeg()


begin = time.perf_counter()
layout_score = layout.cal_score(image)  # 布局得分
brightness_score = brightness.cal_score(image)  # 亮度得分
saturation_score = saturation.cal_score(image)  # 饱和度得分
# ai_score = float(NIMA_predict.eval_pic_by_NIMA(path)[0][0])  # AI打分，获取第一个值并转成float类型
ai_score = NIMA_predict.eval_pic_by_NIMA(path)
total_score = layout_score + brightness_score + saturation_score
end = time.perf_counter()
print('cal_score time: {}s'.format(end-begin))

print('total_score', total_score)
print('AI Score', ai_score)

begin = time.perf_counter()
# 亮度
adjusted = brightness.opt_img(image)  # 调整亮度
adjusted = saturation.opt_img(adjusted)  # 调整饱和度

# 人物分离与背景虚化
alphargb, human, background, human_blur_background = portraitSeg.portraitSeg(imgFile=None, imgOri=adjusted)
adjusted = human_blur_background

denoising.run(adjusted)  # 最后一步去噪，回保存到res.jpg
end = time.perf_counter()
print('adjust time: {}s'.format(end-begin))

# cv2.namedWindow('Original', cv2.WINDOW_KEEPRATIO)    # 窗口大小可以改变
# cv2.imshow("Original", image)
# cv2.namedWindow('Adjusted', cv2.WINDOW_KEEPRATIO)    # 窗口大小可以改变
# cv2.imshow("Adjusted", adjusted)
# cv2.waitKey(0)
