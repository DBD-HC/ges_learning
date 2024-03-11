import torch.nn as nn
import torch
from model.mask_batchnorm import *
import torch.nn.functional as F
import math


class MobileNetV1Block1D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MobileNetV1Block1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, in_channel, groups=in_channel, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(in_channel)
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
        self.conv1 = nn.Conv2d(in_channel, in_channel, groups=in_channel, kernel_size=3, padding=1, bias=False)
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
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, group=1, padding=1, dilation=1, bias=False,
                 bn=True, active=True):
        super(Conv2dBnRelu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, dilation=dilation,
                      kernel_size=kernel_size, padding=padding, stride=stride, bias=bias, groups=group)
        )
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channel))
        if active:
            self.conv.append(nn.ReLU())

    def forward(self, x):
        x = self.conv(x)
        return x


class DeepWiseConv1d(nn.Module):
    def __init__(self, in_channel, out_channel, expand_ratio=1, kernel_size=3, stride=1):
        super(DeepWiseConv1d, self).__init__()
        self.conv = nn.Sequential(
        )
        hidden_channel = in_channel * expand_ratio
        if expand_ratio != 1:
            self.conv.append(
                Conv1dBnRelu(in_channel, hidden_channel, kernel_size=1, bias=False, active=True, bn=True, padding=0))
        self.conv.append(
            Conv1dBnRelu(hidden_channel, hidden_channel, group=hidden_channel, kernel_size=kernel_size, bias=False,
                         stride=stride, active=True, bn=True, padding=1))
        self.conv.append(
            Conv1dBnRelu(hidden_channel, out_channel, kernel_size=1, bias=False, active=expand_ratio <= 1, bn=True,
                         padding=0))

    def forward(self, x):
        x = self.conv(x)
        return x


class DeepWiseConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, expand_ratio=1, kernel_size=3, stride=1):
        super(DeepWiseConv2d, self).__init__()
        self.conv = nn.Sequential(
        )
        hidden_channel = in_channel * expand_ratio
        if expand_ratio != 1:
            self.conv.append(
                Conv2dBnRelu(in_channel, hidden_channel, kernel_size=1, bias=False, active=True, bn=True, padding=0))
        self.conv.append(
            Conv2dBnRelu(hidden_channel, hidden_channel, group=hidden_channel, kernel_size=kernel_size, bias=False,
                         stride=stride, active=True, bn=True, padding=1))
        self.conv.append(
            Conv2dBnRelu(hidden_channel, out_channel, kernel_size=1, bias=False, active=expand_ratio <= 1, bn=True,
                         padding=0))

    def forward(self, x):
        x = self.conv(x)
        return x


class Conv1dBnRelu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, group=1, padding=1, dilation=1, bias=False,
                 bn=True, active=True):
        super(Conv1dBnRelu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, dilation=dilation,
                      kernel_size=kernel_size, padding=padding, stride=stride, bias=bias, groups=group)
        )
        if bn:
            self.conv.append(nn.BatchNorm1d(out_channel))
        if active:
            self.conv.append(nn.ReLU())

    def forward(self, x):
        x = self.conv(x)
        return x


class Rebu(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(Rebu, self).__init__()
        self.conv1 = None
        if in_channel != out_channel:
            self.conv1 = Conv1dBnRelu(in_channel, out_channel)
        self.conv2 = Conv1dBnRelu(out_channel, out_channel, stride=stride)
        self.cutshort = None
        if in_channel != out_channel and stride == 1:
            self.cutshort = Conv1dBnRelu(in_channel, out_channel, kernel_size=1, padding=0)

    def forward(self, x, mask):
        res = x
        if self.conv1 is not None:
            x = self.conv1(x, mask)
        x = self.conv2(x, mask)
        if self.cutshort is not None:
            x = x + self.cutshort(res, mask)
        return x


class SingleDimConvV2(nn.Module):
    def __init__(self, in_channel, channel_num):
        super(SingleDimConvV2, self).__init__()
        self.conv1 = Conv1dBnRelu(in_channel, channel_num[0], stride=2)
        self.dw1 = Rebu(channel_num[0], channel_num[1], stride=2)
        self.dw2 = Rebu(channel_num[1], channel_num[2], stride=2)
        self.dw3 = Rebu(channel_num[1], channel_num[1], stride=1)
        self.dw4 = Rebu(channel_num[1], channel_num[2], stride=1)
        self.dw5 = Rebu(channel_num[2], channel_num[2], stride=2)

    def forward(self, x, mask):
        x = self.conv1(x, mask)
        x = self.dw1(x, mask)
        x = self.dw2(x, mask)

        return x


class SingleDimConv(nn.Module):
    def __init__(self, in_channel, channel_num, input_size=32, dropout=0.):
        super(SingleDimConv, self).__init__()
        self.expand = Conv1dBnRelu(in_channel, channel_num[0], kernel_size=1, padding=0, bn=True, active=True)
        self.conv1 = Conv1dBnRelu(channel_num[0], channel_num[0], kernel_size=3, padding=1, bn=True, active=True)
        # self.ln1 = nn.LayerNorm(input_size)
        # self.dw1 = Conv1dBnRelu(channel_num[0], channel_num[0], stride=2, group=channel_num[0])
        self.conv2 = Conv1dBnRelu(channel_num[0], channel_num[1], kernel_size=3, padding=1, bn=True, active=True)
        # self.ln2 = nn.LayerNorm(input_size)
        # self.dw2 = Conv1dBnRelu(channel_num[1], channel_num[1], stride=2, group=channel_num[1])
        self.conv3 = Conv1dBnRelu(channel_num[1], channel_num[2], kernel_size=3, padding=1, bn=False, active=False)
        # self.ln3 = nn.LayerNorm(input_size)
        # self.ln4 = nn.LayerNorm(input_size)
        # self.dw3 = Conv1dBnRelu(channel_num[2], channel_num[2], stride=2, group=channel_num[2])
        self.down_sample = None
        if in_channel != channel_num[2]:
            self.down_sample = Conv1dBnRelu(in_channel, channel_num[2], kernel_size=1, padding=0, bn=False)
        self.bn = MaskedBatchNorm1d(channel_num[2])
        self.mp = nn.MaxPool1d(kernel_size=2, ceil_mode=True)
        # self.ap = nn.AvgPool1d(kernel_size=2, ceil_mode=True)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x = self.ln1(x)
        res = x
        x = self.expand(x, mask)
        x = self.conv1(x, mask)
        # x = self.ln2(x)
        x = self.dropout1(x)
        # x = self.dw1(x, mask)
        x = self.conv2(x, mask)
        # x = self.ln3(x)
        # x = self.ap(x)
        x = self.conv3(x, mask)
        # x = self.ap(x)
        if self.down_sample is not None:
            res = self.down_sample(res, mask)
        # x = self.bn(res + x, mask=mask)
        x = x + res
        x = F.relu(x)
        x = self.bn(x, mask=mask)
        # x = F.relu(x)

        return x


class s_block(nn.Module):
    def __init__(self, in_channel, out_channel, expand_factor=1, stride=1, activae=True):
        super(s_block, self).__init__()
        self.need_shortcut = False
        if in_channel == out_channel and stride == 1:
            self.need_shortcut = True
        hidden_size = in_channel * expand_factor
        self.expand = Conv1dBnRelu(in_channel, hidden_size, padding=0, kernel_size=1)
        self.dw = Conv1dBnRelu(hidden_size, hidden_size, group=hidden_size, stride=stride, kernel_size=3, padding=1)
        self.ag = Conv1dBnRelu(hidden_size, out_channel, padding=0, kernel_size=1, active=activae)

    def forward(self, x, mask):
        # x = self.ln1(x)
        res = x
        x = self.expand(x, mask)

        x = self.dw(x, mask)

        x = self.ag(x, mask)
        if self.need_shortcut:
            x = res + x
        return x


class AttSE1dBlock(nn.Module):
    def __init__(self, in_channel, radio=1.0):
        super(AttSE1dBlock, self).__init__()
        hidden_size = math.ceil(in_channel * radio)
        self.fc1 = nn.Linear(in_channel, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, in_channel, bias=False)

    def forward(self, x):
        weight = torch.mean(x, dim=-1)
        weight = self.fc1(weight)
        weight = F.relu(weight)
        weight = self.fc2(weight)
        weight = torch.softmax(weight, dim=-1)

        # weight = weight[:,:,None]

        return weight


class ResBlock(nn.Module):
    def __init__(self, in_channel=32, out_channel=32, ratio=2, stride=1):
        super(ResBlock, self).__init__()
        self.dot_conv1 = Conv1dBnRelu(in_channel, in_channel * ratio, kernel_size=1, bn=True, active=True, padding=0)
        self.dw_conv1 = Conv1dBnRelu(in_channel * ratio, in_channel * ratio, kernel_size=3, bn=True, active=True,
                                     padding=1, group=in_channel * ratio)
        self.dot_conv2 = Conv1dBnRelu(in_channel * ratio, out_channel, kernel_size=1, bn=True, active=False, padding=0)
        self.shortcut = False
        if in_channel == out_channel and stride == 1:
            self.shortcut = True

    def forward(self, x, mask):
        res = x
        if self.shortcut is not None:
            res = self.shortcut(res, mask)
        x = self.dot_conv1(x, mask)
        x = self.dw_conv1(x, mask)
        x = self.dot_conv2(x, mask)
        x = x + res
        x = torch.relu(x)
        return x


class BackModelConv(nn.Module):
    def __init__(self, in_channel=32):
        super(BackModelConv, self).__init__()

        self.conv1 = Conv1dBnRelu(in_channel*2, 8, kernel_size=1, active=True, bias=False, bn=False, padding=0)
        self.conv2 = Conv1dBnRelu(8, in_channel*2, kernel_size=1, active=False, bias=False, bn=False, padding=0)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SingleDimConv2(nn.Module):
    def __init__(self, in_channel=32, channel_num=None, input_size=32, dropout=0.5):
        super(SingleDimConv2, self).__init__()

        self.conv1 = Conv1dBnRelu(in_channel, 8, active=True, bn=True)
        self.conv2 = DeepWiseConv1d(8, 16, 1, stride=1)
        # self.down = Conv1dBnRelu(8,8,kernel_size=1, padding=0, bn=False, stride=2)
        self.conv3 = DeepWiseConv1d(16, 32, 1)

        self.mp = nn.MaxPool1d(2, ceil_mode=True)
        self.ap = nn.AvgPool1d(2, ceil_mode=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mp(x)
        x = self.conv2(x)
        x = self.mp(x)
        x = self.conv3(x)
        x = self.mp(x)
        return x


class SingleDimConv3(nn.Module):
    def __init__(self, in_channel=32, channel_num=None, input_size=32, dropout=0.5):
        super(SingleDimConv3, self).__init__()
        # self.conv0 = Conv1dBnRelu(in_channel, 8, stride=1, kernel_size=3, bn=True, active=True, padding=1)
        self.conv1 = Conv1dBnRelu(in_channel, 8, stride=1, kernel_size=3, bn=False, active=True, bias=True, padding=1)
        self.conv2 = Conv1dBnRelu(8, 16, stride=1, kernel_size=3, bn=False, active=True, bias=True, padding=1)
        self.conv3 = Conv1dBnRelu(16, 32, stride=1, kernel_size=3, bn=False, active=True, bias=True, padding=1)

        # self.conv4 = Conv1dBnRelu(16, 16, stride=1, kernel_size=3, bn=False, bias=True, active=True, padding=1, group=2)

        # self.conv4 = Conv1dBnRelu(8, 16, stride=1, kernel_size=3, bn=True, active=True, padding=1)
        self.mp = nn.MaxPool1d(2, ceil_mode=True)
        self.ap = nn.AvgPool1d(2, ceil_mode=True)
        self.dp = nn.Dropout(0.4)

    def forward(self, x):
        # res = self.conv0(x)
        x = self.conv1(x)
        x = self.mp(x)
        x = self.conv2(x)
        x = self.mp(x)
        x = self.conv3(x)
        x = self.mp(x)

        return x


class SingleDimConv4(nn.Module):
    def __init__(self, in_channel=32, channel_num=None, input_size=32, dropout=0.5):
        super(SingleDimConv4, self).__init__()
        # self.conv0 = Conv1dBnRelu(in_channel, 8, stride=1, kernel_size=3, bn=True, active=True, padding=0)
        self.conv1 = Conv1dBnRelu(in_channel, 16, stride=1, kernel_size=3, bn=False, active=True, bias=True, padding=1)
        self.conv2 = Conv1dBnRelu(16, 16, stride=1, kernel_size=3, bn=False, active=True, bias=True, padding=1)
        self.point0 = Conv1dBnRelu(in_channel, 16, stride=1, kernel_size=1, bn=False, active=True, bias=True, padding=0)

        self.conv3 = Conv1dBnRelu(16, 16, stride=1, kernel_size=3, bn=False, active=True, bias=True, padding=1)
        self.conv4 = Conv1dBnRelu(16, 16, stride=1, kernel_size=3, bn=False, bias=True, active=True, padding=1)
        # self.point1 = Conv1dBnRelu(16, 16, stride=2, kernel_size=1, bn=False, active=True, bias=True, padding=0)
        # self.conv5 = Conv1dBnRelu(32, 64, stride=2, kernel_size=3, bn=False, bias=True, active=True, padding=1)
        self.conv5 = Conv1dBnRelu(16, 32, stride=1, kernel_size=3, bn=False, active=True, bias=True, padding=1)
        self.conv6 = Conv1dBnRelu(32, 32, stride=1, kernel_size=3, bn=False, bias=True, active=True, padding=1)
        self.point1 = Conv1dBnRelu(16, 32, stride=1, kernel_size=1, bn=False, active=True, bias=True, padding=0)
        self.mp = nn.MaxPool1d(2, ceil_mode=True)
        self.ap = nn.AvgPool1d(2, ceil_mode=True)
        # self.dp = nn.Dropout(0.4)

    def forward(self, x):
        # res = self.conv0(x)
        res = x
        x = self.conv1(res)
        x = self.conv2(x) + self.point0(res)
        # x = self.dp(x)
        res = self.mp(x)

        x = self.conv3(res)
        x = self.conv4(x) + res
        # x = self.dp(x)
        res = self.mp(x)
        x = self.conv5(res)
        x = self.conv6(x) + self.point1(res)
        x = self.mp(x)

        return x


class CNN2d3Layers(nn.Module):
    def __init__(self, in_channel, channel_num):
        super(CNN2d3Layers, self).__init__()
        self.conv1 = Conv2dBnRelu(in_channel, channel_num[0], padding=0)
        self.conv2 = Conv2dBnRelu(channel_num[0], channel_num[1], padding=0)
        self.conv3 = Conv2dBnRelu(channel_num[1], channel_num[2], padding=0)
        self.mp = nn.MaxPool2d(kernel_size=2, ceil_mode=True)
        self.ap = nn.AvgPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mp(x)
        x = self.conv2(x)
        x = self.mp(x)
        x = self.conv3(x)
        x = self.mp(x)
        return x


class CNN2d3LayersSeparable(nn.Module):
    def __init__(self, in_channel, channel_num):
        super(CNN2d3LayersSeparable, self).__init__()
        self.conv = nn.Sequential(
            *self.create_separable_CNN(in_channel, channel_num[0]),
            nn.MaxPool2d(kernel_size=2, ceil_mode=True),
            *self.create_separable_CNN(channel_num[0], channel_num[1]),
            nn.MaxPool2d(kernel_size=2, ceil_mode=True),
            *self.create_separable_CNN(channel_num[1], channel_num[2]),
            nn.MaxPool2d(kernel_size=2, ceil_mode=True),
        )

    def create_separable_CNN(self, in_channel, out_channel):
        cnnlayers = []
        cnnlayers += [Conv2dBnRelu(in_channel, out_channel, kernel_size=(3, 1), padding=(1, 0))]
        cnnlayers += [Conv2dBnRelu(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1))]
        return cnnlayers

    def forward(self, x, mask):
        x = self.conv(x)
        return x


class MobileNet1DV1(nn.Module):
    def __init__(self, in_channel, channel_num=(4, 8, 16)):
        super(MobileNet1DV1, self).__init__()
        self.conv1 = MobileNetV1Block1D(in_channel, channel_num[0])
        self.mp1 = nn.MaxPool1d(kernel_size=2, ceil_mode=True)
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
        self.dw = nn.MaxPool2d(kernel_size=2, ceil_mode=True)
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
