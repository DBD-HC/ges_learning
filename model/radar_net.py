import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_after_conv_size
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence



class FrameBlock1(nn.Module):
    def __init__(self, feat_size=(32, 32), in_channels=6):
        super(FrameBlock1, self).__init__()
        self.avg_pool = nn.MaxPool2d(kernel_size=(2, 1), ceil_mode=True)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.active1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.active2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.active3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(in_channels)
        self.active4 = nn.ReLU()

    def forward(self, x):
        x = self.avg_pool(x)
        w = x.size(-2)
        h = x.size(-1)
        c = x.size(1)
        batch_size = x.size(0)
        x = torch.transpose(x, 1, -2)
        x = x.contiguous()
        res = x.view(-1, c, h)
        x = self.conv1(res)
        x = self.bn1(x)
        x = self.active1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = res + x
        x = self.active2(x)
        x = torch.transpose(x, 1, -2)
        x = x.contiguous()
        x = x.view(batch_size, -1, w, h)
        res = x
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.active3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = res + x
        x = self.active4(x)
        return x


class FrameBlock2(nn.Module):
    def __init__(self, feat_size=(32, 32), in_channels=6, need_shortcut=True):
        super(FrameBlock2, self).__init__()
        self.point1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.active1 = nn.ReLU()
        self.dw2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, groups=in_channels,
                             bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.active2 = nn.ReLU()
        self.point3 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.active3 = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2)
        self.need_shortcut = need_shortcut
        if need_shortcut:
            self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, stride=2, kernel_size=1,
                                      bias=False)
            self.active4 = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.point1(x)
        x = self.bn1(x)
        x = self.active1(x)

        x = self.dw2(x)
        x = self.bn2(x)
        x = self.active2(x)

        x = self.point3(x)
        x = self.bn3(x)
        x = self.active3(x)

        x = self.max_pool(x)

        if self.need_shortcut:
            res = self.shortcut(res)
            res = self.active4(res)
        x = x + res

        return x


class FrameModule(nn.Module):
    def __init__(self, feat_size=(32, 32), in_channels=6):
        super(FrameModule, self).__init__()
        self.block1 = FrameBlock1(in_channels=in_channels)
        self.block2 = FrameBlock2(in_channels=in_channels, need_shortcut=True)
        self.block3 = FrameBlock2(in_channels=in_channels, need_shortcut=False)
        linear_in = get_after_conv_size(feat_size[0] // 2, kernel_size=3, reduction=2, layer=2) * get_after_conv_size(
            feat_size[1], kernel_size=3, reduction=2, layer=2)
        self.dense4 = nn.Linear(linear_in, 64)
        self.active4 = nn.ReLU()
        self.dense5 = nn.Linear(64, 32)
        self.active5 = nn.ReLU()

    def forward(self, x, data_len=None):
        padding_len = x.size(1)
        batch_size = x.size(0)
        x = self.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(batch_size, padding_len, -1)
        x = self.dense4(x)
        x = self.active4(x)
        x = self.dense5(x)
        x = self.active5(x)

        return x


class TemporalModule(nn.Module):
    def __init__(self, hidden_size=32, out_size=7):
        super(TemporalModule, self).__init__()
        self.lstm1 = nn.LSTM(hidden_size=hidden_size, input_size=hidden_size, batch_first=True, num_layers=1)
        self.dense2 = nn.Linear(in_features=hidden_size, out_features=out_size)

    def forward(self, x, data_len=None):
        x = pack_padded_sequence(x, data_len, batch_first=True)
        x,(h,c) = self.lstm1(x)
        x = h[-1]
        x = self.dense2(x)
        return x


class RadarNet(nn.Module):
    def __init__(self, feat_size=(32, 32), in_channel=6, hidden_size=32, out_size=7):
        super(RadarNet, self).__init__()
        self.frame_model = FrameModule(feat_size=feat_size, in_channels=in_channel)
        self.temporal_model = TemporalModule(hidden_size=hidden_size, out_size=out_size)

    def forward(self, x, data_len=None):
        x = self.frame_model(x)
        x = self.temporal_model(x, data_len)

        return x