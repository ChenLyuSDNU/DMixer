import cv2
from numpy import dtype
import time

import torch

# from network import MobileNet
from torchvision.utils import save_image

image_path = '/home/s2020317058/datasets/dehaze/4KDehazing/low/33_33000066.jpg'
checkpoint = './ouruhd_deblur10.pth'

###### 读取图片################
image = cv2.imread(image_path)[:, :, ::-1]
# image = cv2.resize(image, (256, 256))
image = image.transpose((2, 0, 1))/255.0
image = image.astype('float32')
image = torch.from_numpy(image).unsqueeze(0).cuda()
print(image.shape)
print(image)

# model = MobileNet()
# model.load_state_dict(torch.load(checkpoint))
model = torch.load(checkpoint).cuda()
print(model)

with torch.no_grad():
    start = time.time()
    pre = model(image)
    end = time.time()
    print(end - start)
    print(pre)
    save_image(pre, 'bb.jpg')
    save_image(image, 'aa.jpg')
