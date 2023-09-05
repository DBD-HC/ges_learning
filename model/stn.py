import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import *
import visdom

visdom = visdom.Visdom(env='stn', port=6006)

class SpatialTransformerNetwork(nn.Module):
    def __init__(self, feat_size=(32, 32)):
        super(SpatialTransformerNetwork, self).__init__()

        # 空间变换定位网络(Localization Network)
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        linear_in = (((feat_size[0] - 6)//2 - 4)//2)
        linear_in = linear_in * (((feat_size[1] - 6)//2 - 4)//2)
        linear_in = linear_in * 10

        # 3x3仿射变换矩阵
        self.fc_loc = nn.Sequential(
            nn.Linear(linear_in, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)  # 2x3 矩阵
        )

        # 初始化仿射变换参数
        # self.fc_loc[2].weight.data.zero_()
        # self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x_theta, x):
        if x is None:
            x = x_theta
        # 特征提取
        xs = self.localization(x_theta)
        xs = xs.view(xs.size(0), -1)

        # 预测仿射变换参数
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        theta[:, :, 0:2] = torch.tanh(theta[:, :, 0:2] ) + 1
        theta[:, 0, 1] = 0
        theta[:, 1, 0] = 0


        # 执行空间变换
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x


class RAISTN(nn.Module):
    def __init__(self, feat_size=(32, 32)):
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
