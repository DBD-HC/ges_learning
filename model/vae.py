import torch
import torch.nn as nn

from model.mobile_net import SingleDimConv3, SingleDimConv4
from data.cubelern_arm_dataset import *


class SpatialAttBlock2(nn.Module):
    def __init__(self, in_channel=2, out_size=32, dim=-1, dropout=0.2):
        super(SpatialAttBlock2, self).__init__()
        self.dp = nn.Dropout(0.2)
        if dim == -1:
            self.single_dim_conv = SingleDimConv3(in_channel * 3)
            linear_in = 128
        else:
            self.single_dim_conv = SingleDimConv4(in_channel * 3)
            linear_in = 128
        self.dim = dim
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(linear_in, out_size, bias=False),
            nn.ReLU(),
            nn.Linear(out_size, out_size, bias=True),
        )

    def forward(self, x):
        #x = self.dp(x)
        max_x = torch.max(x, dim=self.dim)[0]
        avg_x = torch.mean(x, dim=self.dim)
        std_x = torch.std(x, dim=self.dim)
        avg_max = torch.cat((max_x, avg_x, std_x), dim=1)
        score = self.single_dim_conv(avg_max)
        score = self.fc1(score)
        return score

class Encoder(nn.Module):
    def __init__(self, in_channel=1, feat_size=(32, 32), latent_dim=32, diff=False):
        super(Encoder, self).__init__()
        self.range_att = SpatialAttBlock2(dim=-1, in_channel=in_channel, out_size=feat_size[0])
        self.angel_att = SpatialAttBlock2(dim=-2, in_channel=in_channel, out_size=feat_size[1])
        # 均值和方差线性层
        self.fc_mean = nn.Linear(feat_size[1] + feat_size[0], latent_dim)
        self.fc_logvar = nn.Linear(feat_size[1] + feat_size[0], latent_dim)
        self.need_diff = diff


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        range_feat = self.range_att(x)
        angle_feat = self.angel_att(x)

        ra_feat = torch.cat((range_feat, angle_feat), dim=-1)
        mean = self.fc_mean(ra_feat)
        logvar = self.fc_logvar(ra_feat)
        z = self.reparameterize(mean, logvar)
        return z, mean, logvar


class Decoder(nn.Module):
    def __init__(self, feat_size=(32, 32), latent_dim=32):
        super(Decoder, self).__init__()
        self.decode = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, feat_size[0] * feat_size[1]),
            # nn.Sigmoid()  # 重构图像的输出范围在 [0, 1]
        )

    def forward(self, x):
        ra_feat = self.decode(x)
        return ra_feat


# 定义变分自编码器的网络结构
class VAE(nn.Module):
    def __init__(self, in_channel=1, feat_size=(32, 32), latent_dim=32):
        super(VAE, self).__init__()

        # 编码器层
        self.encoder = Encoder(in_channel, feat_size, latent_dim)
        self.in_channel = in_channel
        # 解码器层
        self.decoder = Decoder(feat_size=feat_size, latent_dim=latent_dim)


    def forward(self, x):
        # 编码器部分
        batch_size = x.size(0)
        padded_len = x.size(1)
        h = x.size(-2)
        w = x.size(-1)
        x = x.view(-1, self.in_channel, h, w)
        z, mu, logvar= self.encoder(x)

        # 解码器部分
        reconstructed = self.decoder(z)
        reconstructed = reconstructed.view(batch_size, padded_len, h, w)
        return reconstructed, mu, logvar

