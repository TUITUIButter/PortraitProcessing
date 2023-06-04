import os

import cv2
import torch

from .utils import utils_image as util
from .models.network_scunet import SCUNet as net

'''
% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com; github: https://github.com/cszn)
by Kai Zhang (2021/05-2021/11)
'''
abspath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

class Denoising:
    def __init__(self) -> None:
        n_channels = 3
        model_path = os.path.join(abspath, 'SCUNet/model_zoo/scunet_color_real_psnr.pth')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # ----------------------------------------
        # load model
        # ----------------------------------------
        self.model = net(in_nc=n_channels, config=[4, 4, 4, 4, 4, 4, 4], dim=64)

        self.model.load_state_dict(torch.load(model_path), strict=True)
        self.model.eval()
        for k, v in self.model.named_parameters():
            v.requires_grad = False
        self.model = self.model.to(self.device)

    def run(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # ------------------------------------
        # (1) img_L
        # ------------------------------------

        img_L = util.uint2tensor4(img)
        img_L = img_L.to(self.device)

        img_E = self.model(img_L)
        img_E = util.tensor2uint(img_E)

        # ------------------------------------
        # save results
        # ------------------------------------
        util.imsave(img_E, 'res.jpg')

        return img_E
