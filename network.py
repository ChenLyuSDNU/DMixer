import torch
from torch._C import Size
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

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
        self.conv4 = conv_nxn_bn(channel, channel, kernel_size)

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
        # x = self.mlp_6(x).transpose(3, 2)
        # x = self.mlp_5(x).permute(0, 2, 3, 1)
        # x = self.mlp_4(x).permute(0, 3, 2, 1)
        # x = self.mlp_3(x).transpose(3, 2)
        # x = self.mlp_2(x).permute(0, 2, 3, 1)
        # x = self.mlp_1(x).permute(0, 3, 2, 1)

        # Fusion
        x = self.conv3(x)
        # x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return self.t(x) * y - x + 1  

class MobileNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.mixer_block1 = MobileMixerBlock(32, 3, 3, size=(256, 256))

    def forward(self, x):
        
        _, _, h, w = x.shape
        x = F.interpolate(x, (256, 256), mode='bicubic', align_corners=True)
        x = self.mixer_block1(x)
        x = F.interpolate(x, (h, w), mode='bicubic', align_corners=True)

        return x


if __name__ == '__main__':
    input = torch.rand(4, 3, 256, 256).cuda()

    model = MobileNet().cuda()

    pre = model(input)

    print(pre)
    print(pre.shape)