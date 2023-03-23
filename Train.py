import argparse
import os
import dataset

import numpy as np
import torch
from kornia.filters import laplacian
from torch import nn, optim
from torchvision.utils import save_image
from tqdm import tqdm

import network_unet


def train(args):

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = network_unet.MobileNet()
    #model = nn.DataParallel(model, device_ids=[0, 1, 2])  
    model = model.to(args.gpu_id)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    
       
    mse = nn.L1Loss().to(args.gpu_id)
    
    content_folder1 = '/home/s2020317058/datasets/dehaze/4KDehazing/low/'
    information_folder = '/home/s2020317058/datasets/dehaze/4KDehazing/high/'

    # content_folder1 = '/home/s2020317058/datasets/dehaze/O-HAZE/hazy'
    # information_folder = '/home/s2020317058/datasets/dehaze/O-HAZE/GT'
    
    train_loader = dataset.style_loader(content_folder1, information_folder, args.size, 1)
    
    num_batch = len(train_loader)
    for epoch in range(args.epoch):
      for idx, batch in tqdm(enumerate(train_loader), total=num_batch):
            total_iter = epoch*num_batch  + idx
               
            content = batch[0].float().to(args.gpu_id)
            information = batch[1].float().to(args.gpu_id)
            
            optimizer.zero_grad()
            output = model(content)
             
            total_loss =  mse(output , information) 
            total_loss.backward()
            
            optimizer.step()

            if np.mod(total_iter+1, 1) == 0:
                print('{}, Epoch:{} Iter:{} total loss: {}'.format(args.save_dir, epoch, total_iter, total_loss.item()))
                
            if not os.path.exists(args.save_dir+'/image'):
                os.mkdir(args.save_dir+'/image')

      if epoch % 2 ==0:
        #content = torch.log(content)
        #output = torch.log(output)
        out_image = torch.cat([content[0:3], output[0:3], information[0:3]], dim=0)
        save_image(out_image, args.save_dir+'/image/OHAZEiter{}_1.jpg'.format(total_iter+1))
        torch.save(model, 'model' +'/ouruhd_OHAZE{}.pth'.format(epoch))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', default='cuda:0', type=str)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--size', default=512, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--save_dir', default='result', type=str)
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    
    train(args)
