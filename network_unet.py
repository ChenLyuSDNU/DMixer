import torch
from torch._C import Size
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import time

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class Trans(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Trans, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, 1, 1, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU()
    )

def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup*3, kernal_size, stride, 1, bias=False),
        nn.ReLU(),
        nn.Conv2d(oup*3, oup*3, kernal_size, stride, 1, bias=False),
        nn.ReLU(),
        nn.Conv2d(oup*3, oup, kernal_size, stride, 1, bias=False),
        nn.Sigmoid()
    )

class MobileMixerBlock(nn.Module):
    def __init__(self, dim, channel, kernel_size, size, dropout=0.):
        super().__init__()

        self.size = size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.mlp_1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

        self.mlp_2 = nn.Sequential(
            nn.LayerNorm(size[0]),
            nn.Linear(size[0], size[0]),
            nn.Sigmoid(),
            nn.LayerNorm(size[0]),
            nn.Linear(size[0], size[0]),
            nn.Sigmoid(),
            nn.LayerNorm(size[0]),
            nn.Linear(size[0], size[0]),
            nn.Sigmoid()
        )

        self.mlp_3 = nn.Sequential(
            nn.LayerNorm(size[1]),
            nn.Linear(size[1], size[1]),
            nn.Sigmoid(),
            nn.LayerNorm(size[1]),
            nn.Linear(size[1], size[1]),
            nn.Sigmoid(),
            nn.LayerNorm(size[1]),
            nn.Linear(size[1], size[1]),
            nn.Sigmoid()
        )

        self.mlp_4 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

        self.mlp_5 = nn.Sequential(
            nn.LayerNorm(size[0]),
            nn.Linear(size[0], size[0]),
            nn.Sigmoid(),
            nn.LayerNorm(size[0]),
            nn.Linear(size[0], size[0]),
            nn.Sigmoid(),
            nn.LayerNorm(size[0]),
            nn.Linear(size[0], size[0]),
            nn.Sigmoid()
        )

        self.mlp_6 = nn.Sequential(
            nn.LayerNorm(size[1]),
            nn.Linear(size[1], size[1]),
            nn.Sigmoid(),
            nn.LayerNorm(size[1]),
            nn.Linear(size[1], size[1]),
            nn.Sigmoid(),
            nn.LayerNorm(size[1]),
            nn.Linear(size[1], size[1]),
            nn.Sigmoid()
        )

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(channel * 2, channel, kernel_size)

        self.t = Trans(3, 3)

    
    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Global representations
        x = self.mlp_6(x).permute(0, 3, 1, 2)
        x = self.mlp_5(x).permute(0, 3, 1, 2)
        x = self.mlp_4(x).permute(0, 3, 1, 2)
        x = self.mlp_3(x).permute(0, 3, 1, 2)
        x = self.mlp_2(x).permute(0, 3, 1, 2)
        x = self.mlp_1(x).permute(0, 3, 1, 2)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return self.t(x) * y

class MobileNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.mixer_block1 = MobileMixerBlock(32, 3, 3, size=(768, 768))
        # self.mixer_block2 = MobileMixerBlock(32, 3, 3, size=(1024, 1024))
        self.mixer_block3 = MobileMixerBlock(64, 3, 3, size=(384, 384))
        self.u_net_tail = UNet(n_channels=3, n_classes=3)
        
        

    def forward(self, x):
        
        _, _, h, w = x.shape
        t = x
        x = F.interpolate(x, (768, 768), mode='bicubic', align_corners=True)
        z = F.interpolate(x, (768, 768), mode='bicubic', align_corners=True)
        # stage1
        ###########################################################
        x = self.mixer_block1(x) + z

        # stage 2
        ############################################################
        y = F.interpolate(x, (384, 384), mode='bicubic', align_corners=True)
        y = self.mixer_block3(y)
        y = F.interpolate(y, (768, 768), mode='bicubic', align_corners=True) + z
        #############################################################

        x = nn.Sigmoid()(x + y) + z
        x = self.u_net_tail(x)
        x = F.interpolate(x, (h, w), mode='bicubic', align_corners=True)
        return x * t


if __name__ == '__main__':
    for y in range(1000):
        input = torch.rand(1, 3, 3840, 2160).to('cuda:3')

        model = MobileNet().to('cuda:3')

        start = time.time()
        pre = model(input)
        end = time.time()
        print(end - start)
        # print(pre)
        # print(pre.shape)