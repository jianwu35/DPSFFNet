import torch
from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np
from fft_conv_pytorch import FFTConv2d
from torch import nn
from torch.nn import functional as F

from 超声心动图.超声心动图.Echotest.echonet.utils.models.CBAM import CBAM
from typing_extensions import Sequence

from 超声心动图.超声心动图.ISIC.mymodels.myconv import DwConv
import torch
from typing import Sequence


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # print(in_planes)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x


class mynet(nn.Module):
    def __init__(self, in_planes, ratio, kernel_size=7):
        super(mynet, self).__init__()

        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x1, x2):
        _x1 = x1
        _x2 = x2
        x1 = self.ca(x1)
        x2 = self.ca(x2)
        x12 = x1 * x2
        x12 = F.softmax(x12, dim=1)
        x12_1 = x12 * _x1
        x12_2 = x12 * _x2
        x12_1 = self.sa(x12_1)
        x12_2 = self.sa(x12_2)
        x12_1 = F.softmax(x12_1, dim=1)
        x12_2 = F.softmax(x12_2, dim=1)
        x12_1 = x12_1 * _x1
        x12_2 = x12_2 * _x2
        x12_1 = x12_1 + _x1
        x12_2 = x12_2 + _x2
        out = torch.cat([x12_1, x12_2], dim=1)
        return out
