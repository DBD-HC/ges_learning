import torch
from torchvision import models

from model.network import *
from torch.autograd import Function
import torch.nn.init as init

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        #print(f"before alpha:{ctx.alpha}+{grad_output[0, 0]}")
        output = grad_output.neg() * ctx.alpha
        #print(f"after alpha:{ctx.alpha}+{output[0, 0]}")
        return output, None


class Classifier(nn.Module):
    def __init__(self, out_size=6, need_hidden=True):
        super(Classifier, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(192, out_size),
        )
        self.need_hidden = need_hidden

    def forward(self, x):
        #x = ReverseLayerF.apply(x, -1)
        c = self.mlp(x)
        if not self.need_hidden:
            return c
        return x, c


class AmpEncoder(nn.Module):
    def __init__(self, in_channel=1, hidden_channel=1, kernel_size=3, eps=1e-9, zdim=8):
        super(AmpEncoder, self).__init__()
        self.eps = eps
        #padding = (kernel_size - 1) // 2
        padding = 0
        self.conv1 = nn.Sequential(
            #nn.Upsample(size=(48, 48), mode='bilinear'),
            #nn.InstanceNorm2d(1, affine=False),
            #nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=kernel_size, padding=padding,
                      stride=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=kernel_size, padding=padding,
                      stride=1, padding_mode='reflect'),
            nn.ReLU(),
        )

        self.zdim = zdim
        self.rand_fc = nn.Linear(self.zdim, 16, bias=True)
        
        #self.rand_fc.bias.data.fill_(0)
        #self.rand_fc.weight.data.uniform_(-1e-1, 1e-1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=kernel_size, padding=padding,
                      stride=1, padding_mode='reflect'),
            nn.ReLU(),

            #nn.Upsample(scale_factor=2),
            nn.Conv2d(8, 1, kernel_size, padding=padding, padding_mode='reflect'),
        )
        self.dp = nn.Dropout(0.1)
        self.alpha = nn.Parameter(torch.tensor([0.5, 0.5]), requires_grad=True)

    def get_adain(self, x):
        #x = self.intancenorm(x)
        
        mean_x = torch.mean(x, dim=(-2, -1), keepdim=True)
        std_x = torch.std(x, dim=(-2, -1), keepdim=True)
        x = (x - mean_x)/(std_x + self.eps)
        z = torch.randn(x.size(0), self.zdim, device=x.device)
        #z = z * 0.1 + 1
        h = self.rand_fc(z)
        #beta = self.rand_fc2(z)
        gamma, beta = torch.chunk(h, chunks=2, dim=-1)
        #gamma = F.sigmoid(gamma) + 0.5
        #alpha = F.sigmoid(self.alpha)
        #alpha = torch.randn(x.size(0), x.size(1), device=x.device)
        #alpha = alpha * 0.1 + 0.7
        #alpha = alpha[:, :, None, None, None]
        alpha = self.alpha
        #std_hat = torch.sqrt(F.relu(alpha * var_x + (1-alpha) * gamma[:, :, None, None]) + self.eps)
        mean_hat = alpha[0] * mean_x + alpha[1] * beta[:, :, None, None]
        std_hat = alpha[0] * std_x + alpha[1] * gamma[:, :, None, None]
        # x = x * std_x * gamma[:, :, None, None] + mean_x + beta[:, :, None, None]
        x = x * std_hat + mean_hat
        return x, gamma

    def forward(self, x, batch_size, padding_len):
        x0 = F.pad(x, pad=(4, 4, 4, 4), mode='reflect')
        x1 = self.conv1(x0)
        #x1 = x1.view(batch_size, padding_len, -1, 32, 32)
        x2, gamma = self.get_adain(x1)

        #x2 = self.alpha[0] * x1 + self.alpha[1]* x2
        #alpha = F.sigmoid(self.alpha)
        #alpha = torch.randn(x2.size(0), x2.size(1), 1, 1, 1, device=x.device)
        #alpha = 0.7
        #x2 = alpha * x1 + (1 - alpha) * x2

        #x2 = x2.view(batch_size * padding_len, -1, 32, 32)
        #x1 = x1.view(batch_size * padding_len, -1, 16, 16)

        #alpha = 0.7
        #print(self.alpha)
        x3 = self.conv2(x2)
        #x3 = self.alpha[0] * x + self.alpha[1] * x3
        #x4 = self.conv2(x1)
        #x4 = self.get_in(x4, x)
        #x3 = self.get_in(x3, x)
        return x3, None  # , x_rand


class Generator(nn.Module):
    def __init__(self, zdim=10):
        super(Generator, self).__init__()
        self.zdim = zdim
        self.amp_encoder1 = AmpEncoder(zdim=zdim, kernel_size=3)
        self.grl = ReverseLayerF()

    def forward(self, x, alpha=1, data_len=None, content=True):
        # res = x
        batch_size = x.size(0)
        padding_len = x.size(1)
        h = x.size(2)
        w = x.size(3)
        x = x.view(batch_size * padding_len, -1, h, w)
        x1, x_recon = self.amp_encoder1(x, batch_size, padding_len)
        #x1 = ReverseLayerF.apply(x1, -1)
        x1 = x1.view(batch_size, padding_len, h, w)

        x1_rev = ReverseLayerF.apply(x1, alpha)

        return x1, x1_rev, x_recon


def get_norm2(rai, data_len, eps=1e-9):
    rai_packed = rai[:data_len]
    # 计算每个维度的最大值、均值和标准差
    rai_mean = torch.mean(rai_packed, dim=(0, 1, 2))
    rai_var = torch.var(rai_packed, dim=(0, 1, 2))
    rai = (rai - rai_mean) / torch.sqrt(rai_var + eps)
    return rai


# 定义自编码器模型
class DANN(nn.Module):
    def __init__(self, out_size=6, val=False):
        super(DANN, self).__init__()
        self.predication = RAIRadarGestureClassifier(multistream=True, spatial_channels=(8, 16, 32),
                                                     conv1d_channels=(8, 16, 32),
                                                     heads=4,
                                                     track_channels=(4, 8, 16), track_out_size=64, conv2d_feat_size=64,
                                                     diff=True, out_size=out_size,
                                                     ra_feat_size=64, attention=True, frame_level_norm=True,
                                                     in_channel=1, enforce_sorted=False)
        self.predication.classifier = Classifier(out_size, need_hidden=not val)
        self.grl = ReverseLayerF()

        self.domain_classifier = nn.Sequential(
            nn.Linear(192, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2, bias=True)
        )

    def forward(self, x, track, data_lens, alpha, need_domain=True):
        x, c = self.predication(x, data_lens, track)
        # p = self.projection(x)
        if need_domain:
            r_x = ReverseLayerF.apply(x, alpha)
            d = self.domain_classifier(r_x)
            return c, d
        return c  # , p
