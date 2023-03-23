from posixpath import join
import cv2
from numpy import dtype

import torch
import os
# from network_unet import MobileNet
from torchvision.utils import save_image

###########4KID###########
# image_path = '/home/s2020317058/datasets/dehaze/4KDehazing/low/'
# output_path = '/home/s2020317058/datasets/dehaze/4KDehazing/output/'


# ###########I-HAZE###########
image_path = '/home/s2020317058/datasets/dehaze/O-HAZE/test/hazy/'
output_path = '/home/s2020317058/datasets/dehaze/O-HAZE/test/output/'

###########O-HAZE###########
# image_path = '/home/s2020317058/datasets/dehaze/O-HAZE/hazy/'
# output_path = '/home/s2020317058/datasets/dehaze/O-HAZE/output/'

if not os.path.exists:
    os.mkdir(output_path)

##### 模型加载 ############
checkpoint = './model/ouruhd_OHAZE700.pth'
model = torch.load(checkpoint)
model = model.cuda()
print(model)

filenames = os.listdir(image_path)
for filename in filenames:
    image_p = os.path.join(image_path, filename)
    save_path = os.path.join(output_path, filename)
    print(image_p)
    ###### 读取图片 ################
    image = cv2.imread(image_p)[:, :, ::-1]
    # image = cv2.resize(image, (256, 256))
    image = image.transpose((2, 0, 1))/255.0
    image = image.astype('float32')
    image = torch.from_numpy(image).unsqueeze(0).cuda()
    print(image.shape)

    # model = MobileNet()
    # model.load_state_dict(torch.load(checkpoint))

    with torch.no_grad():
        pre = model(image)
        save_image(pre, save_path)


######################test########################
# list_dir(image_path)
