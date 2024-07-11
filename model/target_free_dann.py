import torch
from torchvision import models

from model.network import *
from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        # print(f"alpha:{ctx.alpha}+{output}")
        return output, None


class NotReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output * ctx.alpha
        # print(f"alpha:{ctx.alpha}+{output}")
        return output, None


class Classifier(nn.Module):
    def __init__(self, out_size=6, need_hidden=True):
        super(Classifier, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(192, out_size),
        )
        self.need_hidden = need_hidden

    def forward(self, x):
        c = self.mlp(x)
        if not self.need_hidden:
            return c
        return x, c


class AmpGenerator(nn.Module):
    def __init__(self, in_channel=1, hidden_channel=1, kernel_size=3, in_size=(32, 32), eps=1e-5, zdim=32,
                 need_adain=True):
        super(AmpGenerator, self).__init__()
        self.eps = eps
        padding = (kernel_size - 1) // 2
        mobilenet = models.mobilenet_v3_small(pretrained=True)
        self.encoder = mobilenet.features[:12]
        self.zdim = zdim
        self.need_adain = need_adain
        if need_adain:
            self.rand_fc1 = nn.Sequential(
                nn.Linear(self.zdim, 32),
            )
            self.rand_fc2 = nn.Sequential(
                nn.Linear(self.zdim, 32),
                nn.ReLU()
            )
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(inplace=True),

        )
        self.dp = nn.Dropout(0.)
        # self.norm = nn.InstanceNorm2d(hidden_channel, affine=False)
        self.alpha = nn.Parameter(torch.tensor(0.1), requires_grad=True)

    def get_adain(self, x, dim, gamma, beta):
        # origin = x
        x = self.dp(x)
        mean_x = torch.mean(x, dim=dim, keepdim=True)
        var_x = torch.var(x, dim=dim, keepdim=True)
        std_x = torch.sqrt(var_x + self.eps)
        x = (x - mean_x) / std_x

        alpha = F.sigmoid(self.alpha)
        std_hat = (1 - alpha) * std_x + alpha * gamma
        mean_hat = alpha * beta + (1 - alpha) * mean_x
        x = x * std_hat + mean_hat

        return x

    def get_random(self, x, ):
        z = torch.randn(len(x), self.zdim, device=x.device)
        beta = self.rand_fc1(z)[:, :, None, None]
        gamma = self.rand_fc2(z)[:, :, None, None]
        # gamma, beta = torch.chunk(h, chunks=2, dim=1)
        x = self.get_adain(x, (-1, -2), gamma, beta)
        return x

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
        if self.need_adain:
            x = self.get_random(x)
        x = self.decoder(x)
        return x


class Conv2dBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv2dBlocks, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride=stride if stride > 0 else 1, kernel_size=kernel_size,
                      padding=padding, padding_mode='replicate'),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        if stride < 0:
            self.conv1.append(nn.Upsample(scale_factor=-stride, mode='nearest'))

    def forward(self, x):
        x = self.conv1(x)
        return x


class AmpEncoder3(nn.Module):
    def __init__(self, in_channel=1, hidden_channel=1, kernel_size=3, in_size=(32, 32), eps=1e-5, zdim=8,
                 need_adain=True):
        super(AmpEncoder3, self).__init__()
        self.eps = eps
        padding = (kernel_size - 1) // 2
        self.conv0 = Conv2dBlocks(8, 8, stride=1)
        self.conv1 = Conv2dBlocks(8, 16, stride=2)
        self.conv2 = Conv2dBlocks(16, 16, stride=1)
        self.conv3 = Conv2dBlocks(16, 32, stride=2)

        self.conv4 = Conv2dBlocks(32, 32, stride=1)

        self.conv5 = Conv2dBlocks(32, 16, stride=-2)
        self.conv6 = Conv2dBlocks(16, 16, stride= 1)
        self.conv7 = Conv2dBlocks(16, 8, stride=-2)
        self.conv8 = Conv2dBlocks(8, 8, stride=1)

        self.zdim = zdim
        self.need_adain = need_adain
        if need_adain:
            self.rand_fc1 = nn.Sequential(
                nn.Linear(self.zdim, 8*2),
                nn.ReLU()
            )
            self.rand_fc2 = nn.Sequential(
                nn.Linear(self.zdim, 16 * 2),
                nn.ReLU()
            )
            self.rand_fc3 = nn.Sequential(
                nn.Linear(self.zdim, 32 * 2),
                nn.ReLU()
            )

        self.dp = nn.Dropout(0.)
        self.norm = nn.InstanceNorm2d(hidden_channel, affine=False)
        self.alpha1 = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.alpha2 = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.alpha3 = nn.Parameter(torch.tensor(0.1), requires_grad=True)

    def get_adains(self, x, fc, alpha):
        # origin = x
        x = self.dp(x)
        z = torch.randn(len(x), self.zdim, device=x.device)
        gamma, beta = torch.chunk(fc(z), 2, -1)
        #mean_x = torch.mean(x, dim=(-2, -1), keepdim=True)
        #std_x = torch.std(x, dim=(-2, -1), keepdim=True)
        # std_x = torch.sqrt(var_x + self.eps)
        #x = (x - mean_x)/(std_x + self.eps)
        x = self.norm(x)
        x = x * gamma[:, :, None, None] + beta[:, :, None, None]

        return x

    def forward(self, x):
        x = self.conv0(x) #32=>32 4x4
        x1 = self.conv1(x) #32=>16 4x8
        x2 = self.conv2(x1) #16=>16 8x8
        x3 = self.conv3(x2) #16=>8 8x16
        x3 = self.get_adains(x3, self.rand_fc3, self.alpha3)
        x4 = self.conv4(x3) #8=>8 16x16
        x5 = self.get_adains(self.conv5(x4), self.rand_fc2, self.alpha2) #8=>16 16x8
        x6 = self.conv6(x5) + x1   #16=>16 8x8
        x7 = self.get_adains(self.conv7(x6), self.rand_fc1, self.alpha1) #16=>32 8x4
        x = self.conv8(x7)  + x #32=>32 4x4
        return x


class AmpEncoder(nn.Module):
    def __init__(self, in_channel=1, hidden_channel=1, kernel_size=3, eps=1e-5, zdim=8):
        super(AmpEncoder, self).__init__()
        self.eps = eps
        padding = (kernel_size - 1)//2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=kernel_size, padding=padding,
                      stride=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=kernel_size, padding=padding,
                      stride=1, padding_mode='replicate'),
            nn.MaxPool2d(2),
            nn.ReLU(),)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=kernel_size, padding=padding,
                      stride=1, padding_mode='replicate'),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=kernel_size, padding=padding,
                      stride=1, padding_mode='replicate'),
            nn.ReLU(),
            # nn.Hardswish(),
        )
        self.zdim = zdim
        self.rand_fc = nn.Sequential(
            nn.Linear(self.zdim, 32),
            )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=kernel_size, padding=padding,
                      padding_mode='replicate'),
            #nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=kernel_size, padding=padding,
                      padding_mode='replicate'),
            nn.ReLU(),)
        self.conv4 = nn.Sequential(
            # nn.BatchNorm2d(in_channel),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=kernel_size, padding=padding,
                      padding_mode='replicate'),
            # nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size, padding=padding, padding_mode='replicate'),
        )
        self.dp = nn.Dropout(0.0)
        self.norm = nn.InstanceNorm2d(1, affine=False)
        self.alpha = nn.Parameter(torch.tensor(0.1), requires_grad=True)

    def get_adain(self, x, dim=(-2, -1)):
        z = torch.randn(len(x), self.zdim, device=x.device)
        h = self.rand_fc(z)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        mean_x = torch.mean(x, dim=dim, keepdim=True)
        var_x = torch.var(x, dim=dim, keepdim=True)
        x = (x - mean_x)/torch.sqrt(var_x + self.eps)

        #alpha = F.sigmoid(self.alpha)#[:,None, None]
        #std_hat = torch.sqrt((1 - alpha) * var_x + alpha * F.relu(gamma[:, :, None, None]) + self.eps)
        #mean_hat = alpha * beta[:, :, None, None] + (1 - alpha) * mean_x
        x = x * F.relu(gamma[:, :, None, None]) + beta[:, :, None, None]

        return x


    def forward(self, x):
        # res = x
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x_rand = self.get_adain(x2)
        #x = self.dp(x)
        #alpha = F.sigmoid(self.alpha)
        alpha = 0.6
        x2 = alpha * x2  + (1 - alpha) * x_rand
        x2 = self.conv3(x2)
        x2 = self.conv4(x2)
        mean_x = torch.mean(x, dim=(-2, -1), keepdim=True)
        std_x = torch.std(x, dim=(-2, -1), keepdim=True)
        x2 = self.norm(x2)
        x2 = x2 * std_x + mean_x

        #x_rand = self.conv2(x_rand)
        return x2 #, x_rand



class Generator(nn.Module):
    def __init__(self, zdim=8):
        super(Generator, self).__init__()
        self.zdim = zdim
        self.amp_encoder1 = AmpEncoder(zdim=zdim, kernel_size=3)
        self.grl = ReverseLayerF()

    def forward(self, x, alpha=1, data_len=None):
        # res = x
        batch_size = x.size(0)
        padding_len = x.size(1)
        h = x.size(2)
        w = x.size(3)
        x = x.view(batch_size * padding_len, -1, h, w)
        x1 = self.amp_encoder1(x)
        x1 = x1.view(batch_size, padding_len, h, w)
        x1_rev = ReverseLayerF.apply(x1, alpha)

        return x1, x1_rev


def get_norm(rai, data_len):
    rai_packed = rai[:data_len]
    # 计算每个维度的最大值、均值和标准差
    rai_mean = torch.mean(rai_packed, dim=0)
    rai_var = torch.var(rai_packed, dim=0)
    rai = (rai - rai_mean) / torch.sqrt(rai_var + 1e-5)
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
            nn.Linear(192, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x, track, data_lens, alpha, need_domain=True):
        x, c = self.predication(x, data_lens, track)
        # p = self.projection(x)
        if need_domain:
            r_x = ReverseLayerF.apply(x, alpha)
            d = self.domain_classifier(r_x)
            return c, d
        return c  # , p
