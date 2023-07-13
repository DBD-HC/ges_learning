import torch.nn as nn
import torch
from model.mask_batchnorm import *
import torch.nn.functional as F
import math


class MobileNetV1Block1D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MobileNetV1Block1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, in_channel, groups= in_channel, kernel_size=3, padding=1, bias=False)
        self.bn1 = MaskedBatchNorm1d(in_channel)
        self.conv2 = nn.Conv1d(in_channel, out_channel, kernel_size=1, bias=False)
        self.bn2 = MaskedBatchNorm1d(out_channel)

    def forward(self, x, mask):
        x = self.conv1(x)
        x = self.bn1(x, mask=mask)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x, mask=mask)
        x = F.relu(x, inplace=True)
        return x

class MobileNetV1Block2D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MobileNetV1Block2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel,groups=in_channel, kernel_size=3, padding=1, bias=False)
        self.bn1 = MaskedBatchNorm2d(in_channel)
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        self.bn2 = MaskedBatchNorm2d(out_channel)

    def forward(self, x, mask):
        x = self.conv1(x)
        x = self.bn1(x, mask=mask)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x, mask=mask)
        x = F.relu(x, inplace=True)
        return x

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Conv2dBnRelu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(3, 3), stride=1, group=1):
        super(Conv2dBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                               kernel_size=kernel_size, padding=1, stride=stride, bias=False, groups=group)
        self.bn = MaskedBatchNorm2d(out_channel)

    def forward(self, x, mask):
        x = self.conv(x)
        x = self.bn(x, mask=mask)
        x = F.leaky_relu(x, inplace=True)
        return x

class Conv1dBnRelu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, group=1, padding=1):
        super(Conv1dBnRelu, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel,
                               kernel_size=kernel_size, padding=padding, stride=stride, bias=False, groups=group)
        self.bn = MaskedBatchNorm1d(out_channel)

    def forward(self, x, mask):
        x = self.conv(x)
        x = self.bn(x, mask=mask)
        x = F.leaky_relu(x, inplace=True)
        return x

class SingleDimConvV2(nn.Module):
    def __init__(self, in_channel, dim, channel_num):
        super(SingleDimConvV2, self).__init__()
        conv_kernel = [3, 3]
        conv_kernel[dim] = 1
        conv_kernel = tuple(conv_kernel)
        pooling_kernel = [2, 2]
        pooling_kernel[dim] = 1
        pooling_kernel = tuple(pooling_kernel)
        self.conv1 = Conv2dBnRelu(in_channel, channel_num[0], kernel_size=conv_kernel)
        # self.dw1 = Conv2dBnRelu(channel_num[0], channel_num[0], kernel_size=(3, 1), stride=2, group=channel_num[0])
        self.conv2 = Conv2dBnRelu(channel_num[0], channel_num[1], kernel_size=conv_kernel)
        # self.dw2 = Conv1dBnRelu(channel_num[1], channel_num[1], stride=2, group=channel_num[1])
        self.conv3 = Conv2dBnRelu(channel_num[1], channel_num[2], kernel_size=conv_kernel)
        # self.dw3 = Conv1dBnRelu(channel_num[2], channel_num[2], stride=2, group=channel_num[2])
        if in_channel != channel_num[2]:
            self.down_sample = Conv1dBnRelu(in_channel, channel_num[2], kernel_size=1, padding=0)

        self.mp = nn.MaxPool2d(kernel_size=pooling_kernel, ceil_mode=True)
        self.ap = nn.AvgPool2d(kernel_size=pooling_kernel, ceil_mode=True)

    def forward(self, x, mask):
        res = x
        x = self.conv1(x, mask)
        # x = self.dw1(x, mask)
        x = self.mp(x)
        x = self.conv2(x, mask)
        x = self.ap(x)
        x = self.conv3(x, mask)
        x = self.ap(x)
        return x



class SingleDimConv(nn.Module):
    def __init__(self, in_channel, channel_num):
        super(SingleDimConv, self).__init__()
        self.conv1 = Conv1dBnRelu(in_channel, channel_num[0])
        # self.dw1 = Conv1dBnRelu(channel_num[0], channel_num[0], stride=2, group=channel_num[0])
        self.conv2 = Conv1dBnRelu(channel_num[0], channel_num[1])
        # self.dw2 = Conv1dBnRelu(channel_num[1], channel_num[1], stride=2, group=channel_num[1])
        self.conv3 = Conv1dBnRelu(channel_num[1], channel_num[2])
        # self.dw3 = Conv1dBnRelu(channel_num[2], channel_num[2], stride=2, group=channel_num[2])
        if in_channel != channel_num[2]:
            self.down_sample = Conv1dBnRelu(in_channel, channel_num[2], kernel_size=1, padding=0)

        self.mp = nn.MaxPool1d(kernel_size=2, ceil_mode=True)
        self.ap = nn.AvgPool1d(kernel_size=2, ceil_mode=True)

    def forward(self, x, mask):
        res = x
        x = self.conv1(x, mask)
        # x = self.dw1(x, mask)
        # x = self.mp(x)
        x = self.conv2(x, mask)
        # x = self.ap(x)
        x = self.conv3(x, mask)
        # x = self.ap(x)
        if self.down_sample is not None:
            res = self.down_sample(res, mask)
        x = res + x
        return x

class CNN2d3LayersV2(nn.Module):
    def __init__(self, in_channel, channel_num):
        super(CNN2d3LayersV2, self).__init__()
        self.conv1 = Conv2dBnRelu(in_channel, channel_num[0])
        # self.dw1 = Conv2dBnRelu(channel_num[0], channel_num[0], stride=2, group=channel_num[0])
        self.conv2 = Conv2dBnRelu(channel_num[0], channel_num[1])
        # self.dw2 = Conv2dBnRelu(channel_num[1], channel_num[1], stride=2, group=channel_num[1])
        self.conv3 = Conv2dBnRelu(channel_num[1], channel_num[2])
        # self.dw3 = Conv2dBnRelu(channel_num[2], channel_num[2], stride=2, group=channel_num[2])

        self.mp = nn.MaxPool2d(kernel_size=2, ceil_mode=True)
        self.ap = nn.AvgPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x, mask):
        x = self.conv1(x, mask)
        # x = self.dw1(x, mask)
        x = self.mp(x)
        x = self.conv2(x, mask)
        x = self.ap(x)
        x = self.conv3(x, mask)
        x = self.ap(x)
        return x

class CNN2d3Layers(nn.Module):
    def __init__(self, in_channel, channel_num):
        super(CNN2d3Layers, self).__init__()
        self.conv1 = Conv2dBnRelu(in_channel, channel_num[0])
        self.conv2 = Conv2dBnRelu(channel_num[0], channel_num[1])
        self.conv3 = Conv2dBnRelu(channel_num[1], channel_num[2])
        self.mp = nn.MaxPool2d(kernel_size=2, ceil_mode=True)
        self.ap = nn.AvgPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x, mask):
        x = self.conv1(x, mask)
        x = self.mp(x)
        x = self.conv2(x, mask)
        x = self.ap(x)
        x = self.conv3(x, mask)
        x = self.ap(x)
        return x

class MobileNet1DV1(nn.Module):
    def __init__(self, in_channel, channel_num=(4, 8, 16)):
        super(MobileNet1DV1, self).__init__()
        self.conv1 = MobileNetV1Block1D(in_channel, channel_num[0])
        self.mp1 = nn.MaxPool1d(kernel_size=2,ceil_mode=True)
        self.conv2 = MobileNetV1Block1D(channel_num[0], channel_num[1])
        self.ap1 = nn.MaxPool1d(kernel_size=2, ceil_mode=True)
        self.conv3 = MobileNetV1Block1D(channel_num[1], channel_num[2])

    def forward(self, x, mask):
        x = self.conv1(x, mask)
        x = self.mp1(x)
        x = self.conv2(x, mask)
        x = self.ap1(x)
        x = self.conv3(x, mask)
        x = self.ap1(x)
        return x



class MobileNetV12D(nn.Module):
    def __init__(self, in_channel, channel_num=(16, 24, 32)):
        super(MobileNetV12D, self).__init__()
        self.conv1 = MobileNetV1Block2D(in_channel, channel_num[0])
        self.dw = nn.MaxPool2d(kernel_size=2,ceil_mode=True)
        self.conv2 = MobileNetV1Block2D(channel_num[0], channel_num[1])
        self.ap1 = nn.MaxPool2d(kernel_size=2, ceil_mode=True)
        self.conv3 = MobileNetV1Block2D(channel_num[1], channel_num[2])

    def forward(self, x, mask):
        x = self.conv1(x, mask)
        x = self.mp1(x)
        x = self.conv2(x, mask)
        x = self.ap1(x)
        x = self.conv3(x, mask)
        x = self.ap1(x)
        return x

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)

class MobileNetV22D(nn.Module):
    def __init__(self, input_channel=2, last_channel=32, width_mult=1.):
        super(MobileNetV22D, self).__init__()
        block = InvertedResidual
        last_channel = last_channel
        interverted_residual_setting = [
            # t, c, n, s
            [1, 8, 1, 1],
            [6, 16, 3, 2],
            [6, 32, 4, 2],
            [6, 64, 3, 1],
            # [6, 128, 1, 1],
        ]
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(input_channel, 8, 2)]
        input_channel = 32
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(len(x), -1)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()