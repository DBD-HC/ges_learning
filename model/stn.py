import torch.nn as nn
import torch.nn.functional as F
import torch

from model.attention import MultiHeadAttention
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


class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()

        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(360, 32)
        )

        self.time_fusion =  MultiHeadAttention(query_size=32, key_size=32,
                                                                   value_size=32, num_hidden=32,
                                                                   num_heads=4, dropout=0.2, bias=False)

        # Regressor for the affine transformation
        self.fc_loc = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Linear(16, 2 * 3)  # 2 * 3 for the 2x3 affine matrix
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x, padding_len, data_len):
        xs = self.localization(x)

        xs = xs.view(-1, padding_len, x.size(-1))
        xs = self.time_fusion(xs, xs, xs, data_len)
        xs = xs.view(-1, x.size(-1))
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x, padding_len, data_len):
        # transform the input
        x = self.stn(x, padding_len, data_len)
        return x


class RAISTN(nn.Module):
    def __init__(self, feat_size=32):
        super(RAISTN, self).__init__()

        self.stn = SpatialTransformer(feat_size=feat_size)

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
