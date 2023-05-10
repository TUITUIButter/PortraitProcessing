import argparse
import os
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms

from Neural_IMage_Assessment.model.model import *

seed = 42
torch.manual_seed(seed)

# transform
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
    ])

abspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
checkpoint_path = os.path.join(abspath, 'checkpoints', 'NIMA_EPOCH_82.pth')
def eval_pic_by_NIMA(img_path: str, checkpoint_path=checkpoint_path):
    '''
    读取图片并对图片进行质量评估
    返回一个(-1, 10)的值 值越大代表质量越好
    中文解读blog https://www.leiphone.com/category/ai/4Jqh8c9VEymN3Bfw.html
    Github: https://github.com/yunxiaoshi/Neural-IMage-Assessment

    Args:
        img_path: 输入图片的路径
        checkpoint_path: 预训练模型的地址
    Returns:
        mean: 平均值 (代表图像的评分)
        std: 方差
    '''
    base_model = models.vgg16(pretrained=True)
    model = NIMA(base_model)
    try:
        model.load_state_dict(torch.load(checkpoint_path))  # args.model是预训练模型的地址
        print('successfully loaded model')
    except:
        raise

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    im = Image.open(os.path.join(img_path))
    im = im.convert('RGB')
    imt = test_transform(im)
    imt = imt.unsqueeze(dim=0)
    imt = imt.to(device)
    with torch.no_grad():
        out = model(imt)
    out = out.view(10, 1)
    mean, std = 0.0, 0.0
    for j, e in enumerate(out, 1):
        mean += j * e
    for k, e in enumerate(out, 1):
        std += e * (k - mean) ** 2
    std = std ** 0.5
    # print(' mean: %.3f | std: %.3f' % (mean, std))
    return mean, std


if __name__ == '__main__':
    mean, std = eval_pic_by_NIMA(os.path.join(abspath, 'imgs', '01.PNG'))
    print(mean, std)
