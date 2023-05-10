import math

import numpy
from .base_module import BaseModule
from .utils import Utils
import cv2


def get_pose_single(points):
    # 考虑将哪两个点(多种选择)之间的连线作为人体的连线
    nose_point = points[Utils.BODY_PARTS["Nose"]]
    lHip_point = points[Utils.BODY_PARTS["LHip"]]
    rHip_point = points[Utils.BODY_PARTS["RHip"]]
    if not nose_point or (not lHip_point and not rHip_point):  # 没有鼻子或者没有左右胯都没有
        print("无法计算人体倾斜的角度")
        return 7

    x_nose = nose_point[0]
    y_nose = nose_point[1]
    if lHip_point and rHip_point:
        x_hip = (lHip_point[0] + rHip_point[0]) / 2
        y_hip = (lHip_point[1] + rHip_point[1]) / 2
    else:
        x_hip = lHip_point[0] if lHip_point else rHip_point[0]
        y_hip = lHip_point[1] if lHip_point else rHip_point[1]

    # 返回上面角的度数
    angle = math.atan2(abs(x_nose - x_hip), abs(y_nose - y_hip)) * 180 / math.pi

    # 根据度数获取分数，线性赋分
    min_angle = 10
    max_angle = 20
    if angle <= min_angle:  # 满分
        print("angle: {:.2f}°\tangle score: {:.2f}".format(angle, 10))
        return 10
    if angle >= max_angle:  # 0分
        print("angle: {:.2f}°\tangle score: {:.2f}".format(angle, 0))
        return 0

    percent = (angle - min_angle) / (max_angle - min_angle)
    score = 10 - percent * 10
    print("angle: {:.2f}°\tangle score: {:.2f}".format(angle, score))
    return score


def get_position_single(img, points):
    (h, w) = img.shape[:2]
    # 查看人的头部是否过于偏上或者偏下
    nose_point = points[Utils.BODY_PARTS["Nose"]]
    if not nose_point:  # 找不到鼻子，那这个肯定不好
        head_score = 2
    else:
        y_nose = nose_point[1]

        if y_nose / h < 0.15:  # 头过于靠上，比值小于0.15
            head_score = 3
        elif y_nose / h > 0.7:  # 整个身子都下图片的下半方
            head_score = 2
        else:
            head_score = 10

    # 计算所有点的中心当作身体重心，查看身体的位置
    av_x = 0
    av_y = 0
    cnt = 0
    for point in points:
        if point is None:
            continue
        cnt += 1
        av_x += point[0]
        av_y += point[1]
    av_x /= cnt
    av_y /= cnt

    # 转化为百分比
    av_x /= w
    av_y /= h

    # 计算黄金分割点，水平方向有左右两个分割点，竖直方向只有下面的，人像图不能过于靠上
    golden_w_l = 1 - 0.618
    golden_w_r = 0.618
    golden_h = 0.618

    golden_score = 0
    ep = 0.05  # 宽容度，百分比
    if golden_w_l - ep <= av_x <= golden_w_r + ep:  # 左边右边或者中间，一个区间
        golden_score += 4
    if golden_h - ep <= av_y <= golden_h + ep:
        golden_score += 4

    if golden_score == 8:  # 如果xy都在黄金分割点，满分
        golden_score += 2

    print("position score: {:.2f}".format((golden_score + head_score) / 2))
    return (golden_score + head_score) / 2


class Layout(BaseModule):
    def __init__(self, path='modules/graph_opt.pb'):
        # 加载模型
        self.net = cv2.dnn.readNetFromTensorflow(path)

    def cal_score(self, img) -> float:
        points = Utils.get_pose_point(img)
        angle_score = get_pose_single(points)
        position_score = get_position_single(img, points)
        return (angle_score + position_score) / 2

    def opt_img(self, img) -> numpy.ndarray:
        res = [1, 2, 3]
        return numpy.array(res)
        pass
