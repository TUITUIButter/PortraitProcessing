import numpy
from .utils import Utils
from .base_module import BaseModule


class Brightness(BaseModule):
    def cal_score(self, img) -> float:
        dark_threshold = 0.6
        bright_threshold = 0.3

        dark_prop, bright_prop = Utils.brightness_detection(img)
        score = 10
        if dark_prop >= dark_threshold:  # 整体环境黑暗的图片
            score -= (dark_prop - dark_threshold) / (1 - dark_threshold) * 10.0
        elif bright_prop >= 0.3:
            score -= (bright_prop - bright_threshold) / (1 - bright_threshold) * 10.0

        print("brightness score: {:.2f}".format(score))
        return score

    def opt_img(self, img) -> numpy.ndarray:
        pass
