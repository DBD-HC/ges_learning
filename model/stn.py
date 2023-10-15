import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import *
import visdom

visdom = visdom.Visdom(env='stn', port=6006)


def get_axis(size, device):
    x = torch.arange((size), dtype=torch.float32, device=device)
    x = x * (2.0 / (size - 1)) - 1
    return x


def affine_grid(theta, feat_size):
    x = get_axis(feat_size[-1], theta.device)[None, None, :]
    y = get_axis(feat_size[-2], theta.device)[None, :, None]
    x = x.expand(feat_size[0], feat_size[-2], -1)
    y = y.expand(feat_size[0], -1, feat_size[-1])
    x = x * (theta[:, 0][:, None, None]) + theta[:, 2][:, None, None]
    y = y * (theta[:, 1][:, None, None]) + theta[:, 3][:, None, None]
    return torch.cat((x[:, :, :, None], y[:, :, :, None]), dim=-1)


class SpatialTransformerNetwork(nn.Module):
    def __init__(self, feat_size=(32, 32)):
        super(SpatialTransformerNetwork, self).__init__()

        # 空间变换定位网络(Localization Network)
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        linear_in = 250

        # 3x3仿射变换矩阵
        self.fc_loc = nn.Sequential(
            nn.Linear(linear_in, 4)
        )

        # 初始化仿射变换参数
        self.fc_loc[0].weight.data.zero_()
        self.fc_loc[0].bias.data.copy_(torch.tensor([1, 1, 0, 0], dtype=torch.float))

    def forward(self, x_theta, x):
        if x is None:
            x = x_theta
        # 特征提取
        xs = self.localization(x_theta)
        xs = xs.view(xs.size(0), -1)

        # 预测仿射变换参数
        theta = self.fc_loc(xs)
        # theta[:, 0:2] = torch.relu(theta[:, 0:2])
        # x = x[:, :, :, None]
        # 执行空间变换
        grid = affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x


class RAISTN(nn.Module):
    def __init__(self, feat_size=32):
        super(RAISTN, self).__init__()

        self.stn = SpatialTransformerNetwork(feat_size=feat_size)

    def forward(self, x, data_len, **kwargs):
        x_sum = torch.sum(x, dim=1)[:, None, :]
        x_mean = x_sum / (data_len[:, None, None, None])
        x = self.stn(x_mean, x)

        if not self.training and 'indexes' in kwargs:
            indexes = kwargs['indexes']
            ids = str(kwargs['epoch'] % 10)
            if 777 in indexes:
                index = indexes.index(777)
                x_t = x.view(len(indexes), -1, 32, 32)

                visdom.heatmap(x_t[index][10], win=ids + '_after_stn',
                               opts=dict(title=ids + 'after_stn' + str(kwargs['epoch'])))
        return x
