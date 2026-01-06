import math

import torch.nn as nn
from torch.nn.modules.utils import _triple


class DwConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation: int = 1, bias=True):
        super(DwConv, self).__init__()

        # 逐通道卷积（Depthwise Convolution）
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels, bias=bias)

        # 点卷积（Pointwise Convolution）
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)  # 1x1卷积核

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class R2plus1DConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(2, 2), padding=(3, 3), bias=True):
        super(R2plus1DConv, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_size[0], kernel_size[1]),
                                stride=stride[:2], padding=padding[:2], bias=bias)
        self.conv1d = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size[1], stride=1,
                                padding=padding[1], bias=bias)

    def forward(self, x, padding=(3, 3), kernel_size=(7, 7), stride=(2, 2)):
        batch_size, channels, time_steps, height, width = x.size()

        x = x.reshape(batch_size, channels * time_steps, height, width)

        x = self.conv2d(x)  # Move depth dimension to after channel dimension for 2D convolution

        new_height = (height + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
        new_width = (width + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
        # assert x.size(-1) * x.size(-2) == new_height * new_width, "Shape mismatch after 2D convolution"
        x = x.reshape(batch_size, channels * time_steps, new_height * new_width)
        x = self.conv1d(x)
        # Reshape back to original shape
        x = x.reshape(batch_size, channels * time_steps, new_height, new_width)

        return x


class SpatioTemporalConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(SpatioTemporalConv, self).__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        spatial_kernel_size = [1, kernel_size[1], kernel_size[2]]
        spatial_stride = [1, stride[1], stride[2]]
        spatial_padding = [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride = [stride[0], 1, 1]
        temporal_padding = [padding[0], 0, 0]

        intermed_channels = int(
            math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels) / \
                       (kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                      stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.bn = nn.BatchNorm3d(intermed_channels)
        self.relu = nn.ReLU()

        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size,
                                       stride=temporal_stride, padding=temporal_padding, bias=bias)

    def forward(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))
        x = self.temporal_conv(x)
        return x
