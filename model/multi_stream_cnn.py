import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_after_conv_size
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channel, kernel_size=5, reduction=(2, 2)):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channel, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.mp = nn.MaxPool2d(kernel_size=reduction, ceil_mode=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.mp(x)
        return x


class MSBlock(nn.Module):
    def __init__(self, in_channel=1, n_channels=(8, 16, 32, 64, 128, 256, 512)):
        super(MSBlock, self).__init__()
        self.conv = nn.Sequential()
        pre_c = in_channel
        for i, c in enumerate(n_channels):
            self.conv.add_module('conv' + str(i), ConvBlock(pre_c, c))
            pre_c = c
        self.fc = nn.Linear(n_channels, n_channels)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MSCNN(nn.Module):
    def __init__(self, in_channel=1, n_channels=(8, 16, 32, 64, 128, 256, 512), out_size=7):
        super(MSCNN, self).__init__()
        self.range_conv = MSBlock(in_channel=in_channel, n_channels=n_channels)
        self.doppler_conv = MSBlock(in_channel=in_channel, n_channels=n_channels)
        self.angel_conv = MSBlock(in_channel=in_channel, n_channels=n_channels)
        self.fc = nn.Linear(n_channels*3, out_size)

    def forward(self, x):
        r = self.range_conv(x)
        d = self.doppler_conv(x)
        a = self.angel_conv(x)
        x = torch.cat((r, d, a), dim=-1)
        x = self.fc(x)
        return x
