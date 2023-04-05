from modules.layout import Layout
import cv2

image = cv2.imread('./imgs/04.jpg')
assert image is not None
layout = Layout('./modules/graph_opt.pb')

layout.cal_score(image)
