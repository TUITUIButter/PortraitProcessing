import cv2
import numpy as np
from SCUNet.scunet import Denoising
import time

# 加载图像
image = cv2.imread('./imgs/03.jpg')
de = Denoising()

begin = time.perf_counter()
de.run(image)
end = time.perf_counter()

print(end-begin)
