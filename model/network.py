import torch
import numpy as np
from cplxmodule.nn import CplxConv1d, CplxLinear, CplxDropout
from cplxmodule.nn import CplxModReLU, CplxParameter, CplxModulus, CplxToCplx
from cplxmodule.nn.modules.casting import TensorToCplx
from cplxmodule.nn import RealToCplx, CplxToReal
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from utils import *
from model.attention import *
from model.mobile_net import *
from model.tcn import TCN
from model.lstm import LSTM_2
import visdom
import torch.nn.init as init
from model.stn import RAISTN
from model.squeeze_and_excitation import *


class SeparableConvolution(nn.Module):
    def __init__(self, in_channel=1, out_channel=8):
        super(SeparableConvolution, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(3, 1), padding=(2, 0), padding_mode='reflect',
                               dilation=2)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 2), padding_mode='reflect',
                               dilation=2)
        self.bn = MaskedBatchNorm2d(out_channel)

    def forward(self, x, mask=None, **kwargs):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x, mask=mask)
        x = torch.relu(x)
        return x


class SeparableCFAR(nn.Module):
    def __init__(self, window_size, in_channel=1, expand_channel=4, input_size=(32, 32)):
        super(SeparableCFAR, self).__init__()
        self.input_size = input_size
        self.conv1 = SeparableConvolution(in_channel, expand_channel)
        self.conv2 = SeparableConvolution(in_channel, expand_channel)
        self.conv3 = SeparableConvolution(in_channel, in_channel)
        self.visdom = visdom.Visdom(env='CFAR', port=6006)
        self.bn3 = MaskedBatchNorm2d(in_channel)
        self.ln = nn.LayerNorm(input_size)

        self.relu0 = torch.nn.Hardswish()

    def forward(self, x, mask=None, **kwargs):
        # x = self.bn0(x, mask=mask)
        # x_mean = torch.mean(x, dim=1)[:,None, :]

        thr = self.conv1(x)
        thr = self.conv2(thr)
        thr = self.conv3(thr)
        weight = x - thr
        res = F.relu(weight) * F.sigmoid(self.ln(weight))
        res = self.bn3(res, mask=mask)
        if not self.training and 'indexes' in kwargs:
            indexes = kwargs['indexes']
            ids = str(kwargs['epoch'] % 10)
            if 777 in indexes:
                index = indexes.index(777)
                x_t = x.view(len(indexes), -1, 32, 32)
                th_t = thr.view(len(indexes), -1, 32, 32)
                res_t = res.view(len(indexes), -1, 32, 32)

                self.visdom.heatmap(x_t[index][11], win=ids + '_origin',
                                    opts=dict(title=ids + 'origin' + str(kwargs['epoch'])))
                self.visdom.heatmap(res_t[index][11], win=ids + '_cfar',
                                    opts=dict(title=ids + 'cfar ' + str(kwargs['epoch'])))
                self.visdom.heatmap(th_t[index][11], win=ids + '_cfar_thr',
                                    opts=dict(title=ids + 'cfar thr' + str(kwargs['epoch'])))

        return res


def soft_thresholding(x, threshold):
    return -1 * torch.sign(x) * torch.relu(torch.abs(x) - threshold) + x


def soft_thresholding2(x, threshold):
    return torch.sign(x) * torch.relu(torch.abs(x) - threshold)


class CFAR(nn.Module):
    def __init__(self, window_size, in_channel=1, expand_channel=4, input_size=(32, 32)):
        super(CFAR, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, expand_channel, padding=0,
                               kernel_size=1, groups=in_channel, dilation=1, bias=False)
        # self.conv2 = nn.Conv2d(expand_channel, expand_channel, padding=1, padding_mode='reflect',
        #                        kernel_size=3, dilation=1, groups=expand_channel, bias=False)
        self.conv2 = nn.Conv2d(expand_channel, expand_channel, padding=2, padding_mode='reflect',
                               kernel_size=5, dilation=1, groups=expand_channel, bias=False)
        self.conv4 = nn.Conv2d(expand_channel, in_channel, padding=0,
                               kernel_size=1, groups=in_channel, dilation=1, bias=False)
        # init.uniform_(self.conv3.weight, 0.001, 0.002)
        # init.uniform_(self.conv1.weight, 0.8, 1.2)
        # init.uniform_(self.conv4.weight, 0.12, 0.13)
        # init.normal_(self.conv2.weight, 0, 0.04)
        # init.uniform_(self.conv3.weight, 0.10, 0.12)

        self.visdom = visdom.Visdom(env='CFAR', port=6006)
        # self.bn3 = MaskedBatchNorm2d(in_channel)
        # self.ln = nn.LayerNorm(input_size)
        # self.bn = MaskedBatchNorm2d(1)
        self.bn = nn.BatchNorm2d(in_channel)
        self.ap = nn.AvgPool2d(kernel_size=(2, 2), ceil_mode=True)
        # self.mp = nn.MaxPool2d(kernel_size=2, ceil_mode=True)
        # self.relu0 = torch.nn.ReLU()
        # self.se1 = ThrSEBlock(expand_channel, radio=2)
        # self.se2 = ThrSEBlock(expand_channel, radio=2)

    #  self.se3 = ThrSEBlock(expand_channel, radio=2)

    def forward(self, x, mask=None, data_len=None, **kwargs):
        thr = self.conv1(x)
        thr = self.ap(thr)
        # thr = torch.repeat_interleave(thr, repeats=8, dim=-3)
        thr = self.conv2(thr)
        # thr = self.ap(thr)
        # w = self.se3(thr, mask)
        # thr = soft_thresholding(thr, w[:, :, None, None])
        # thr = self.ap(thr)

        thr = self.conv4(thr)
        thr = torch.repeat_interleave(thr, repeats=2, dim=-1)
        thr = torch.repeat_interleave(thr, repeats=2, dim=-2)
        res = x - thr
        res = F.relu(res)  # *torch.sigmoid(res)
        res = self.bn(res)
        if not self.training and 'indexes' in kwargs:
            indexes = kwargs['indexes']
            ids = str(kwargs['epoch'] % 10)
            if 777 in indexes:
                index = indexes.index(777)
                x_t = x.view(len(indexes), -1, 32, 32)
                th_t = thr.view(len(indexes), -1, 32, 32)
                res_t = res.view(len(indexes), -1, 32, 32)

                self.visdom.heatmap(x_t[index][13], win=ids + '_origin',
                                    opts=dict(title=ids + 'origin' + str(kwargs['epoch'])))
                self.visdom.heatmap(res_t[index][13], win=ids + '_cfar',
                                    opts=dict(title=ids + 'cfar ' + str(kwargs['epoch'])))
                self.visdom.heatmap(th_t[index][13], win=ids + '_cfar_thr',
                                    opts=dict(title=ids + 'cfar thr' + str(kwargs['epoch'])))

        return res


class SpatialAttBlock(nn.Module):
    def __init__(self, in_channel=2, channel_num=(16, 8, 8), diff=True, input_size=32, out_size=32, dim=-1, dropout=0.5,
                 eps=1e-5):
        super(SpatialAttBlock, self).__init__()
        self.single_dim_conv = SingleDimConv3(in_channel * 3, channel_num)
        # linear_in = get_after_conv_size(size=input_size, kernel_size=5, layer=3, padding=0,
        #                                 reduction=1) * channel_num[-1]
        linear_in = get_after_conv_size(size=input_size, kernel_size=3, layer=1, padding=1,
                                        reduction=1) * channel_num[-1]
        # linear_in = 32
        self.dim = dim
        self.fc1 = nn.Linear(linear_in, out_size, bias=False)

        # self.bn1 = MaskedBatchNorm1d(in_channel * 3)
        # self.bn1 = nn.BatchNorm1d(in_channel * 3)
        # self.bn2 = MaskedBatchNorm1d(linear_in)
        # self.bn3 = nn.BatchNorm1d(out_size)
        self.ln1 = nn.LayerNorm(out_size)
        self.dp = nn.Dropout(dropout)
        self.need_diff = diff
        if diff:
            self.diff = DifferentialNetwork(in_size=out_size)
        # self.stn = RAISTN(feat_size=32)

    def forward(self, x, batch_size, padding_len, data_len=None, mask=None, **kwargs):
        # feat_len = x.size(self.dim)
        max_x = torch.max(x, dim=self.dim)[0]
        avg_x = torch.mean(x, dim=self.dim)
        std_x = torch.std(x, dim=self.dim)
        avg_max = torch.cat((max_x, avg_x, std_x), dim=1)
        # avg_max = self.bn1(avg_max, mask=mask)
        # avg_max = self.bn1(avg_max)
        score = self.single_dim_conv(avg_max, mask)
        score = score.view(len(x), -1)
        score = self.dp(score)
        # score = F.hardswish(score)
        score = self.fc1(score)
        score = self.ln1(score)
        score = F.relu(score)
        if self.need_diff:
            score = self.diff(score, batch_size, padding_len)
        else:
            score = score.view(batch_size, padding_len, -1)
        return score


class SpatialModel(nn.Module):
    def __init__(self, num_channels, ra_conv=True, diff=False, in_size=(32, 32), ra_feat_size=32, out_size=128,
                 in_channels=1,
                 dropout=0.5):
        super(SpatialModel, self).__init__()
        self.ap0 = nn.AvgPool2d(kernel_size=4, ceil_mode=True)
        self.conv_back_modeling = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, bias=False)

        self.need_ra_conv = ra_conv

        self.dp = nn.Dropout(dropout)
        self.in_channels = in_channels
        # self.range_se = SE1dBlock(32)
        self.bn0 = nn.BatchNorm2d(self.in_channels)

        # 是否需要角度距离注意力
        linear_input = num_channels[-1] * get_after_conv_size(size=in_size[0], kernel_size=3, layer=3, padding=1,
                                                              reduction=2) ** 2
        # sp_feat_size = out_size
        if ra_conv:
            self.range_att = SpatialAttBlock(dim=-1, in_channel=in_channels, input_size=in_size[0], diff=diff,
                                             out_size=ra_feat_size)
            self.angel_att = SpatialAttBlock(dim=-2, in_channel=in_channels, input_size=in_size[1], diff=diff,
                                             out_size=ra_feat_size)
        sp_feat_size = out_size - ra_feat_size * 2

        self.spatial_conv = CNN2d3Layers(in_channels, num_channels)
        self.fc_1 = nn.Linear(linear_input, sp_feat_size, bias=False)

        # self.fc_2 = nn.Linear(linear_input, sp_feat_size//2, bias=False)
        # self.fc_3 = nn.Linear(linear_input, sp_feat_size//2, bias=False)
        self.ln1 = nn.LayerNorm(sp_feat_size)
        # self.ln2 = nn.LayerNorm(sp_feat_size//2)
        if diff:
            self.diff = DifferentialNetwork(in_size=sp_feat_size)
            # self.diff2 = DifferentialNetwork(in_size=ra_feat_size*2)

        self.need_diff = diff

    def forward(self, x, data_length, mask=None, **kwargs):
        batch_size = x.size(0)
        padded_len = x.size(1)
        h = x.size(-2)
        w = x.size(-1)
        # x = self.range_se(torch.squeeze(x), mask)
        x = x.view(-1, self.in_channels, h, w)
        #x = F.hardswish(x - t)
        range_score = None
        angel_score = None
        if self.need_ra_conv:
            range_score = self.range_att(x, batch_size=batch_size, padding_len=padded_len, data_len=data_length,
                                         mask=mask, **kwargs)
            angel_score = self.angel_att(x, batch_size=batch_size, padding_len=padded_len, data_len=data_length,
                                         mask=mask, **kwargs)
        x = self.spatial_conv(x, mask)
        x = x.view(x.size(0), -1)
        x = self.dp(x)
        x = self.fc_1(x)
        x = self.ln1(x)
        x = F.relu(x)
        if self.need_diff:
            x = self.diff(x, batch_size, padded_len)
        else:
            x = x.view(batch_size, padded_len, -1)

        return x, range_score, angel_score


def lstm_func(lstm_block, data, data_lens):
    if data_lens is not None:
        x = pack_padded_sequence(data, data_lens.cpu(), batch_first=True)
        output, (h_n, _) = lstm_block(x)
        output, out_len = pad_packed_sequence(output, batch_first=True)
    else:
        output, (h_n, _) = lstm_block(data)
    return output, h_n[-1]


def custom_dropout(input_tensor, p=0.5, training=True):
    if training:
        mask = torch.bernoulli(torch.full_like(input_tensor[:, 0, :], 1 - p))
        return input_tensor * mask[:, None, :] / (1 - p)
    else:
        return input_tensor


def custom_dropout_temporal(input_tensor, p=0.5, training=True):
    if training:
        mask = torch.bernoulli(torch.full_like(input_tensor[:, :, 0], 1 - p))
        return input_tensor * mask[:, :, None] / (1 - p)
    else:
        return input_tensor

def dynamic_temporal(input_tensor, training=True, mean=1, std=0.01):
    if training:
        amp_rand = torch.normal(mean, std, size=(input_tensor.size(0), input_tensor.size(1), 1), device=input_tensor.device)
        return input_tensor * amp_rand
    else:
        return input_tensor


class DifferentialNetwork(nn.Module):
    def __init__(self, in_size, dropout=0.5):
        super(DifferentialNetwork, self).__init__()
        self.linear_sp1 = nn.Sequential(

            nn.Linear(in_size, in_size // 2, bias=True),
            # nn.BatchNorm1d(in_size//2),
            nn.ReLU(),
        )
        self.linear_sp2 = nn.Sequential(
            nn.Linear(in_size // 2, in_size // 2, bias=True),
            nn.Hardswish(),
            nn.Linear(in_size // 2, in_size, bias=False),
            nn.LayerNorm(in_size),
            nn.Hardswish(),
        )
        # self.featmap1=None
        # self.featmap2=None
        self.dropout = dropout
        self.dp0 = nn.Dropout(dropout)

    def forward(self, x, batch_size, padded_len):
        x = self.linear_sp1(x)
        x = x.view(batch_size, padded_len, -1)
        # self.featmap1 = x
        x = x[:, 1:] - x[:, :-1]
        # self.featmap2 = x
        x = self.linear_sp2(x)

        return x


class TemporalModel(nn.Module):
    def __init__(self, feat_size1, out_size, attention=True, diff=True, ra_conv=True, feat_size2=None, heads=2,
                 dropout=0.5):
        super(TemporalModel, self).__init__()
        self.lstm_1 = nn.LSTM(input_size=feat_size1, hidden_size=feat_size1, num_layers=1, batch_first=True)
        hidden_size = feat_size1
        if ra_conv:
            self.lstm_2 = nn.LSTM(input_size=feat_size2, hidden_size=feat_size2, num_layers=1,
                                  batch_first=True)
            self.lstm_3 = nn.LSTM(input_size=feat_size2, hidden_size=feat_size2, num_layers=1,
                                  batch_first=True)
            hidden_size = feat_size1 + feat_size2 * 2

        self.need_attention = attention
        if attention:
            self.multi_head_attention = MultiHeadAttention(query_size=hidden_size, key_size=hidden_size,
                                                                   value_size=hidden_size, num_hidden=out_size,
                                                                   num_heads=heads, dropout=0.5, bias=False)
            # self.multi_head_attention = nn.MultiheadAttention(embed_dim=out_size, kdim=hidden_size, vdim=hidden_size,
            #                                                   batch_first=True, dropout=0.5, num_heads=heads,
            #                                                   bias=False)
        if hidden_size != out_size:
            self.fc = nn.Linear(hidden_size, out_size)
        else:
            self.fc = None
        self.bn = nn.BatchNorm1d(out_size)
        # self.ln = nn.LayerNorm(hidden_size)
        # self.bn_mid = nn.BatchNorm1d(hidden_size)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.dropout_3 = nn.Dropout(p=dropout)
        self.need_diff = diff

    def forward(self, x1, x2=None, x3=None, data_lens=None, mask=None, **kwargs):
        if self.need_diff and data_lens is not None:
            data_lens = data_lens - 1

        # x1 = self.dropout_1(x1)
        output, final_state = lstm_func(self.lstm_1, x1, data_lens)

        if x2 is not None:
            # x2 = self.dropout_2(x2)
            # x3 = self.dropout_2(x3)
            # ra = custom_dropout(ra, 0.5, self.training)
            output_2, final_state_2 = lstm_func(self.lstm_2, x2, data_lens)
            output_3, final_state_3 = lstm_func(self.lstm_3, x3, data_lens)
            output = torch.cat((output, output_2, output_3), dim=-1)
            final_state = torch.cat((final_state, final_state_2, final_state_3), dim=-1)
        final_state = final_state[:, None, :]
        if self.fc is not None:
            final_state = self.fc(final_state)
        if self.need_attention:
            # if data_lens is not None:
            #     mask = torch.arange((x1.size(-2)), dtype=torch.float32, device=x1.device)
            #     mask = mask[None, :] >= data_lens[:, None]
            # x1 = final_state + self.multi_head_attention(final_state, output, output, key_padding_mask=mask,
            #                                              need_weights=False)[0]
            x1 = final_state + self.multi_head_attention(final_state, output, output, data_lens)
        else:
            x1 = final_state

        x1 = x1.view(len(x1), -1)

        x1 = self.bn(x1)
        x1 = F.hardswish(x1)

        return x1


class GlobalConv(nn.Module):
    def __init__(self, num_channels, in_channels=3, in_size=(32, 32), out_size=64, dropout=0.5):
        super(GlobalConv, self).__init__()
        self.track_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, num_channels[0], kernel_size=(3, 3), bias=False),

            nn.BatchNorm2d(num_channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(num_channels[0], num_channels[1], kernel_size=(3, 3), bias=False),

            nn.BatchNorm2d(num_channels[1]),
            nn.ReLU(),
            nn.AvgPool2d(2, ceil_mode=True),
            nn.Conv2d(num_channels[1], num_channels[2], kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(num_channels[2]),
            nn.ReLU(),
            nn.AvgPool2d(2, ceil_mode=True),
        )
        self.dp0 = nn.Dropout(p=dropout)
        self.dp = nn.Dropout(p=dropout)
        linear_input = num_channels[-1] * get_after_conv_size(size=in_size[0], kernel_size=3, layer=3, reduction=2) \
                       * get_after_conv_size(size=in_size[1], kernel_size=3, layer=3, reduction=2)
        self.fc_1 = nn.Linear(linear_input, out_size, bias=False)
        self.bn0 = nn.BatchNorm2d(in_channels * 2)
        self.bn4 = nn.BatchNorm1d(out_size)

    def forward(self, x, data_length, **kwargs):
        if data_length is None:
            t_mean = torch.mean(x, dim=1)
            t_std = torch.std(x, dim=1)
        else:
            t_sum = torch.sum(x, dim=1)
            t_mean = t_sum / (data_length[:, None, None, None])
            t_std = (torch.sum(x * x, dim=1) / (data_length[:, None, None, None]) - t_mean * t_mean)
            t_std = torch.sqrt(F.relu(t_std))
        # t_mean = torch.mean(x, dim=1)
        # t_std = torch.std(x, dim=1)
        # t_max = torch.max(x, dim=1)[0]
        x = torch.cat((t_mean, t_std), dim=1)
        # x = self.bn0(x)
        x = self.track_conv(x)
        x = x.view(x.size(0), -1)
        x = self.dp(x)
        x = self.fc_1(x)
        x = self.bn4(x)
        x = F.hardswish(x)

        # x = self.dp(x)
        # x = self.dp0(x)
        return x


class RAIRadarGestureClassifier(nn.Module):
    def __init__(self, cfar=True, track=True, ra_conv=True, diff=True, attention=True, heads=4, in_size=(32, 32),
                 hidden_size=(128, 64), in_channel=1,
                 out_size=7, cfar_expand_channels=8, cfar_in_channel=1, dropout=0.5,
                 track_channels=(4, 8, 16), ra_feat_size=32, track_out_size=64, spatial_channels=(8, 16, 32)):
        super(RAIRadarGestureClassifier, self).__init__()

        self.need_track = track
        self.need_ra_conv = ra_conv
        self.hidden_size = hidden_size
        self.ra_feat_size = ra_feat_size
        self.sn = SpatialModel(ra_conv=ra_conv, in_size=in_size, num_channels=spatial_channels, diff=diff,
                               out_size=hidden_size[0], ra_feat_size=ra_feat_size, in_channels=in_channel)
        fc_feat_size = 0
        if track:
            self.tn = GlobalConv(in_channels=in_channel, num_channels=track_channels, in_size=in_size,
                                 out_size=track_out_size)
            fc_feat_size = fc_feat_size + track_out_size

        feat_size = hidden_size[0] - 2 * ra_feat_size
        if ra_conv:
            self.temporal_net = TemporalModel(feat_size1=feat_size, out_size=hidden_size[1], attention=attention,
                                              ra_conv=ra_conv, heads=heads, diff=diff,
                                              feat_size2=ra_feat_size)
            fc_feat_size = fc_feat_size + hidden_size[1]
        else:
            self.temporal_net = TemporalModel(feat_size1=feat_size, out_size=feat_size, attention=attention, diff=diff,
                                              ra_conv=ra_conv)
            fc_feat_size = fc_feat_size + feat_size
        self.fc_2 = nn.Linear(fc_feat_size, out_size)
        self.dropout_1 = nn.Dropout(p=dropout)

    def forward(self, x, data_length, **kwargs):
        bach_size = x.size(0)
        padded_lens = x.size(1)
        h = x.size(-2)
        w = x.size(-1)

        x = x.view(bach_size, padded_lens, -1, h, w)
        t = None
        if self.need_track:
            t = self.tn(x, data_length, **kwargs)
            t = t.view(bach_size, -1)

        x, r, a = self.sn(x, data_length, **kwargs)
        if self.need_ra_conv:
            x = self.temporal_net(x, r, a, data_lens=data_length, **kwargs)
        else:
            x = self.temporal_net(x, data_lens=data_length, **kwargs)
        if self.need_track:
            x = torch.cat((x, t), dim=-1)
        x = self.fc_2(x)

        return x





if __name__ == '__main__':
    net = RAIRadarGestureClassifier(cfar=True, track=True, spatial_channels=(4, 8, 16), ra_conv=True, heads=4,
                                    track_channels=(4, 8, 16), track_out_size=64, hidden_size=(128, 128),
                                    ra_feat_size=32, attention=True, cfar_expand_channels=8, in_channel=1)
    res = getModelSize(net)
