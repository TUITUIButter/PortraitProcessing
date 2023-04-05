# 抽象类加抽象方法就等于面向对象编程中的接口
from abc import ABCMeta, abstractmethod
import numpy


class BaseModule:
    __metaclass__ = ABCMeta  # 指定这是一个抽象类

    """
    根据照片返回分数，分数应缩放至[0-10]
    """
    @abstractmethod
    def cal_score(self, img) -> float:
        pass

    """
    处理图片，返回新图。通道统一为RGB
    """
    @abstractmethod
    def opt_img(self, img) -> numpy.ndarray:
        pass
