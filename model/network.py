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


class CFAR(nn.Module):
    def __init__(self, window_size, in_channel=1, expand_channel=4, input_size=(32, 32)):
        super(CFAR, self).__init__()
        if isinstance(window_size, tuple):
            padding = ((window_size[1] - 1) // 2, (window_size[2] - 1) // 2)
        else:
            padding = (window_size - 1) // 2

            # conv_weight = torch.from_numpy(init_weight(window_size, 6)).unsqueeze(0).unsqueeze(0)
            # self.expand = nn.Conv2d(in_channel, channel_nums[0], kernel_size=1)
            self.conv1 = nn.Conv2d(in_channel, expand_channel, padding=2, padding_mode='reflect',
                                   kernel_size=3, dilation=2, bias=False)
            self.conv2 = nn.Conv2d(expand_channel, expand_channel, padding=2, padding_mode='reflect',
                                   kernel_size=5, groups=expand_channel, bias=False)
            self.conv3 = nn.Conv2d(expand_channel, in_channel, padding=2, padding_mode='reflect',
                                   kernel_size=3, dilation=2, bias=False)
            # init.uniform_(self.conv1.weight, 0.02, 0.06)
            # init.uniform_(self.conv2.weight, 0.08, 0.13)
            # init.uniform_(self.conv3.weight, 0.08, 0.13)

            # self.conv1.weight = torch.nn.Parameter(conv_weight.float())
            self.visdom = visdom.Visdom(env='CFAR', port=6006)
            self.bn3 = MaskedBatchNorm2d(in_channel)
            self.ln = nn.LayerNorm(input_size)

            self.relu0 = torch.nn.Hardswish()
            self.sigmoid = torch.nn.Tanh()
            # self.bias = nn.Parameter(torch.mul(torch.ones(size=input_size), 100), requires_grad=True)

    def forward(self, x, mask=None, **kwargs):
        # x = self.bn0(x, mask=mask)
        thr = self.conv1(x)
        # thr = self.relu0(thr)
        # thr = self.bn1(thr, mask=mask)
        thr = self.relu0(thr)
        # thr = self.av_pool(thr)
        thr = self.conv2(thr)
        # thr = self.bn2(thr, mask=mask)
        thr = self.relu0(thr)
        # thr = self.av_pool(thr)
        thr = self.conv3(thr)
        # thr = self.relu0(thr)
        weight = x - thr
        # thr = self.av_pool(thr)
        res = F.relu(weight)*F.sigmoid(self.ln(weight))
        # res = x * weight
        # weight = self.ln(weight)
        # weight = self.sigmoid(weight)
        # res = x * weight

        # res = F.hardswish(res)
        res = self.bn3(res, mask=mask)
        # res = self.bn1(res, mask=mask)
        if not self.training and 'indexes' in kwargs:
            indexes = kwargs['indexes']
            ids = str(kwargs['epoch'] % 10)
            if 777 in indexes:
                index = indexes.index(777)
                x_t = x.view(len(indexes), -1, 32, 32)
                th_t = thr.view(len(indexes), -1, 32, 32)
                res_t = res.view(len(indexes), -1, 32, 32)

                self.visdom.heatmap(x_t[index][10], win=ids + '_origin',
                                    opts=dict(title=ids + 'origin' + str(kwargs['epoch'])))
                self.visdom.heatmap(res_t[index][10], win=ids + '_cfar',
                                    opts=dict(title=ids + 'cfar ' + str(kwargs['epoch'])))
                self.visdom.heatmap(th_t[index][10], win=ids + '_cfar_thr',
                                    opts=dict(title=ids + 'cfar thr' + str(kwargs['epoch'])))

        return res


class SIGNAL_NET_BETA(nn.Module):
    def __init__(self, cfar=True):
        super(SIGNAL_NET_BETA, self).__init__()
        self.need_cfar = cfar
        # self.fc_1 = nn.Linear(128, 64)
        # self.fc_2 = nn.Linear(64, 32)
        self.dp = nn.Dropout(p=0.5)
        if self.need_cfar:
            self.CFAR = CFAR(3)
            self.conv1 = nn.Conv2d(2, 16, kernel_size=(3, 3), padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1, bias=False)
        self.bn1 = MaskedBatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1, bias=False)
        self.bn2 = MaskedBatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1, bias=False)
        self.bn3 = MaskedBatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)

    def forward(self, x, data_length, **kwargs):
        # x = x.view(-1, 128, 32*32)
        # x = torch.transpose(x, 1, 2)
        # x = self.fc_1(x)
        # x = self.fc_2(x)
        # x = torch.transpose(x, 1, 2)
        # x = x.contiguous()
        padded_len = x.size(1)
        x = x.view(-1, 1, 32, 32)
        if self.need_cfar:
            detect = self.CFAR(x, padded_len, data_length, **kwargs)
            x = torch.cat((x, detect), dim=1)
            x = x.contiguous()
        x = self.conv1(x)
        x = self.bn1(x, padded_len, data_length)
        # x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.bn2(x, padded_len, data_length)
        # x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.bn3(x, padded_len, data_length)
        # x = self.bn3(x)
        x = F.relu(x)
        x = self.maxpool(x)
        return x


def get_cnn_1d_block(in_channel=4, channel_num=(16, 16, 4)):
    range_conv = nn.Sequential(
        nn.Conv1d(in_channel, channel_num[0], kernel_size=3, stride=1, padding=1, bias=False),
        nn.LeakyReLU(),
        # nn.MaxPool1d(2, ceil_mode=True),
        nn.Conv1d(channel_num[0], channel_num[-2], kernel_size=3, stride=1, padding=1, bias=False),
        nn.LeakyReLU(),
        # nn.AvgPool1d(2, ceil_mode=True),
        nn.Conv1d(channel_num[-2], channel_num[-1], kernel_size=3, stride=1, padding=1, bias=False),
        nn.LeakyReLU(),
        # nn.AvgPool1d(2, ceil_mode=True),
    )
    return range_conv


class SpatialAttBlock(nn.Module):
    def __init__(self, in_channel=2, channel_num=(8, 16, 32), input_size=32, out_size=32, dim=-1, dropout=0.5, eps=1e-5):
        super(SpatialAttBlock, self).__init__()
        self.single_dim_conv = SingleDimConv(in_channel * 3, channel_num)
        linear_in = get_after_conv_size(size=input_size, kernel_size=3, layer=3, padding=1,
                                        reduction=2) * channel_num[-1]
        self.dim = dim
        self.fc1 = nn.Linear(linear_in, out_size, bias=False)
        #self.fc2 = nn.Linear(out_size, out_size, bias=False)

        self.bn1 = MaskedBatchNorm1d(in_channel * 3)
        self.bn3 = MaskedBatchNorm1d(out_size)
        self.dp = nn.Dropout(dropout)

    def forward(self, x, batch_size, padding_len, mask=None, **kwargs):
        # feat_len = x.size(self.dim)
        max_x = torch.max(x, dim=self.dim)[0]
        avg_x = torch.mean(x, dim=self.dim)
        std_x = torch.std(x, dim=self.dim)
        avg_max = torch.cat((max_x, avg_x, std_x), dim=1)
        avg_max = self.bn1(avg_max, mask=mask)
        score = self.single_dim_conv(avg_max, mask)
        score = score.view(len(x), -1)
        score = self.dp(score)
        score = self.fc1(score)

        score = self.bn3(score, mask=mask)
        score = score.view(batch_size, padding_len, -1)
        return score

class SpatialAttBlock2(nn.Module):
    def __init__(self, in_channel=2, channel_num=(8, 16, 32), input_size=32, out_size=32, dim=-1, dropout=0.5, eps=1e-5):
        super(SpatialAttBlock2, self).__init__()
        self.single_dim_conv = SingleDimConv2(in_channel * 3, channel_num)
        linear_in = get_after_conv_size(size=input_size, kernel_size=3, layer=3, padding=1,
                                        reduction=1) * channel_num[-1]
        self.dim = dim
        self.fc1 = nn.Linear(174, out_size, bias=False)
        #self.fc2 = nn.Linear(out_size, out_size, bias=False)

        self.bn1 = MaskedBatchNorm1d(in_channel * 3)
        self.bn3 = MaskedBatchNorm1d(out_size)
        self.dp = nn.Dropout(dropout)

    def forward(self, x, batch_size, padding_len, mask=None, **kwargs):
        # feat_len = x.size(self.dim)
        x = torch.squeeze(x)
        max_x = torch.max(x, dim=self.dim)[0]
        avg_x = torch.mean(x, dim=self.dim)
        std_x = torch.std(x, dim=self.dim)
        avg_max = torch.cat((max_x, avg_x, std_x), dim=-1)
        if self.dim == -1:
            x = torch.transpose(x , -2, -1)

        # avg_max = self.bn1(avg_max, mask=mask)

        score = self.single_dim_conv(x, mask)
        score = score.view(len(x), -1)
        score = torch.cat((score, avg_max), dim=-1)
        score = self.dp(score)
        score = self.fc1(score)

        score = self.bn3(score, mask=mask)
        score = score.view(batch_size, padding_len, -1)
        return score

class SpatialModel2(nn.Module):
    def __init__(self, num_channels, ra_conv=True, in_size=(32, 32), ra_feat_size=32, out_size=128, in_channels=1,
                 dropout=0.5):
        super(SpatialModel2, self).__init__()
        self.need_ra_conv = ra_conv
        if self.need_ra_conv:
            self.spatial_conv = Conv2dBnRelu(in_channel=1, out_channel=num_channels[-1])
            conv_h = get_after_conv_size(size=in_size[0], kernel_size=3, layer=1, padding=1, reduction=1)
            conv_w = get_after_conv_size(size=in_size[1], kernel_size=3, layer=1, padding=1, reduction=1)
        else:
            self.spatial_conv = CNN2d3Layers2(in_channels, num_channels)
            conv_h = get_after_conv_size(size=in_size[0], kernel_size=3, layer=1, padding=1, reduction=1)
            conv_w = get_after_conv_size(size=in_size[1], kernel_size=3, layer=1, padding=1, reduction=1)
        self.dp = nn.Dropout(dropout)
        self.in_channels = in_channels

        # 是否需要角度距离注意力
        linear_input = num_channels[-1] * conv_h * conv_w
        if ra_conv:
            self.range_att = SpatialAttBlock2(dim=-1, in_channel=num_channels[-1], input_size=conv_h,
                                             out_size=ra_feat_size)
            self.angel_att = SpatialAttBlock2(dim=-2, in_channel=num_channels[-1], input_size=conv_w,
                                             out_size=ra_feat_size)
            # self.fc_1 = nn.Linear(2 * ra_feat_size, 2 * ra_feat_size, bias=False)
            # self.bn4 = MaskedBatchNorm1d(2 * ra_feat_size)
            # self.dp_2 = nn.Dropout(dropout)
        else:
            self.fc_1 = nn.Linear(linear_input, out_size, bias=False)
            self.bn4 = MaskedBatchNorm1d(out_size)

    def forward(self, x, data_length, mask=None, **kwargs):
        batch_size = len(data_length)
        padded_len = x.size(0)//batch_size
        h = x.size(-2)
        w = x.size(-1)

        x = x.view(-1, self.in_channels, h, w)
        x = self.spatial_conv(x, mask)
        if self.need_ra_conv:
            range_score = self.range_att(x, batch_size=batch_size, padding_len=padded_len, mask=mask, **kwargs)
            angel_score = self.angel_att(x, batch_size=batch_size, padding_len=padded_len, mask=mask, **kwargs)
            return torch.cat((range_score, angel_score), dim=-1)


        # 提取RA二维频谱图特征
        x = self.spatial_conv(x, mask)
        x = x.view(x.size(0), -1)
        x = self.dp(x)
        x = self.fc_1(x)
        x = self.bn4(x, mask=mask)

        x = x.view(len(data_length), padded_len, -1)
        x = F.hardswish(x)
        return x


class SpatialModel(nn.Module):
    def __init__(self, num_channels, ra_conv=True, in_size=(32, 32), ra_feat_size=32, out_size=128, in_channels=1,
                 dropout=0.5):
        super(SpatialModel, self).__init__()
        self.need_ra_conv = ra_conv
        self.spatial_conv = CNN2d3Layers(in_channels, num_channels)
        self.dp = nn.Dropout(dropout)
        self.in_channels = in_channels

        # 是否需要角度距离注意力
        linear_input = num_channels[-1] * get_after_conv_size(size=in_size[0], kernel_size=3, layer=3, padding=1,
                                                              reduction=2) \
                       * get_after_conv_size(size=in_size[1], kernel_size=3, layer=3, padding=1, reduction=2)
        if ra_conv:
            self.range_att = SpatialAttBlock(dim=-1, in_channel=in_channels, input_size=in_size[0],
                                             out_size=ra_feat_size)
            self.angel_att = SpatialAttBlock(dim=-2, in_channel=in_channels, input_size=in_size[1],
                                             out_size=ra_feat_size)
            self.fc_1 = nn.Linear(linear_input, out_size - 2 * ra_feat_size, bias=False)
            self.bn4 = MaskedBatchNorm1d(out_size - 2 * ra_feat_size)
            self.dp_2 = nn.Dropout(dropout)
        else:
            self.fc_1 = nn.Linear(linear_input, out_size, bias=False)
            self.bn4 = MaskedBatchNorm1d(out_size)

    def forward(self, x, data_length, mask=None, **kwargs):
        batch_size = len(data_length)
        padded_len = x.size(0)//batch_size
        h = x.size(-2)
        w = x.size(-1)

        x = x.view(-1, self.in_channels, h, w)
        if self.need_ra_conv:
            range_score = self.range_att(x, batch_size=batch_size, padding_len=padded_len, mask=mask, **kwargs)
            angel_score = self.angel_att(x, batch_size=batch_size, padding_len=padded_len, mask=mask, **kwargs)


        # 提取RA二维频谱图特征
        x = self.spatial_conv(x, mask)
        x = x.view(x.size(0), -1)
        x = self.dp(x)
        x = self.fc_1(x)
        x = self.bn4(x, mask=mask)

        x = x.view(len(data_length), padded_len, -1)
        if self.need_ra_conv:
            x = torch.cat((x, range_score, angel_score), dim=-1)
        x = F.hardswish(x)
        return x

class TemporalModel2(nn.Module):
    def __init__(self, input_size, out_size, attention=True, ra_conv=True, ra_feat_size=None, heads=4, dropout=0.5):
        super(TemporalModel2, self).__init__()
        self.lstm_1 = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
        self.need_attention = attention
        if attention:
            # self.multi_head_attention = MultiHeadAttention(query_size=hidden_size, key_size=hidden_size,
            #                                                value_size=hidden_size, num_hidden=out_size,
            #                                                num_heads=heads, dropout=nn.Dropout(p=0.1))
            self.multi_head_attention = ResidualMultiHeadAttention(query_size=64, key_size=64,
                                                           value_size=64, num_hidden=64,
                                                           num_heads=heads, dropout=0.4)
        self.bn0 = MaskedBatchNorm1d(64)
        self.bn = nn.BatchNorm1d(64)
        self.dropout_2 = nn.Dropout(p=0.5)

    def forward(self,x=None, data_lens=None, mask=None, **kwargs):
        # x = self.bn0(x, mask=mask)
        x = pack_padded_sequence(x, data_lens.cpu(), batch_first=True)
        output, (h_n, _) = self.lstm_1(x)
        output, out_len = pad_packed_sequence(output, batch_first=True)
        final_state = h_n[-1]
        final_state = final_state[:, None, :]
        if self.need_attention:
            # q_x = torch.sum(output, dim=1)[:, :]
            # q_x = q_x/data_lens[:, None]
            # q_x = q_x[:, None, :]
            # x = self.multi_head_attention(q_x, output, output, data_lens)
            # x = final_state + self.dropout_2(x)
            x = self.multi_head_attention(final_state, output, output, data_lens)
            # x = final_state + self.dropout_2(x)
        else:
            x = final_state

        x = torch.squeeze(x)
        x = F.hardswish(x)
        x = self.bn(x)
        # x = self.dropout_2(x)

        return x

class TemporalModel(nn.Module):
    def __init__(self, input_size, out_size, attention=True, ra_conv=True, ra_feat_size=None, heads=4, dropout=0.5):
        super(TemporalModel, self).__init__()
        self.lstm_1 = nn.LSTM(input_size=input_size, hidden_size=input_size, num_layers=1, batch_first=True)
        hidden_size = input_size
        if ra_conv:
            self.lstm_2 = nn.LSTM(input_size=ra_feat_size * 2, hidden_size=ra_feat_size * 2, num_layers=1,
                                  batch_first=True)
            hidden_size = input_size + ra_feat_size * 2
        self.need_attention = attention
        if attention:
            # self.multi_head_attention = MultiHeadAttention(query_size=hidden_size, key_size=hidden_size,
            #                                                value_size=hidden_size, num_hidden=out_size,
            #                                                num_heads=heads, dropout=nn.Dropout(p=0.1))
            self.multi_head_attention = ResidualMultiHeadAttention(query_size=hidden_size, key_size=hidden_size,
                                                           value_size=hidden_size, num_hidden=out_size,
                                                           num_heads=heads, dropout=0.4)
        if hidden_size != out_size:
            self.fc = nn.Linear(hidden_size, out_size)
        else:
            self.fc = None
        self.bn = nn.BatchNorm1d(out_size)
        self.bn_mid = nn.BatchNorm1d(hidden_size)
        self.dropout_2 = nn.Dropout(p=0.5)

    def forward(self, x, ra=None, data_lens=None, mask=None, **kwargs):
        x = pack_padded_sequence(x, data_lens.cpu(), batch_first=True)
        output, (h_n, _) = self.lstm_1(x)
        output, out_len = pad_packed_sequence(output, batch_first=True)
        final_state = h_n[-1]

        if ra is not None:
            ra = pack_padded_sequence(ra, data_lens.cpu(), batch_first=True)
            output_2, (h_n_2, _) = self.lstm_2(ra)
            output_2, out_len = pad_packed_sequence(output_2, batch_first=True)
            output = torch.cat((output, output_2), dim=-1)
            final_state = torch.cat((final_state, h_n_2[-1]), dim=-1)
        final_state = final_state[:, None, :]
        if self.fc is not None:
            final_state = self.fc(final_state)
        if self.need_attention:
            # q_x = torch.sum(output, dim=1)[:, :]
            # q_x = q_x/data_lens[:, None]
            # q_x = q_x[:, None, :]
            # x = self.multi_head_attention(q_x, output, output, data_lens)
            # x = final_state + self.dropout_2(x)
            x = self.multi_head_attention(final_state, output, output, data_lens)
            # x = final_state + self.dropout_2(x)
        else:
            x = final_state

        x = torch.squeeze(x)
        x = F.hardswish(x)
        x = self.bn(x)
        # x = self.dropout_2(x)

        return x


class TRACK_NET(nn.Module):
    def __init__(self, num_channels, in_channels=3, in_size=(32, 32), out_size=64, dropout=0.5):
        super(TRACK_NET, self).__init__()
        self.track_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_channels[0], kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(num_channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(num_channels[0], num_channels[1], kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(num_channels[1]),
            # qweqweqwenn.Dropout(p=0.4),
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
        self.fc_1 = nn.Linear(linear_input, out_size)
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.bn4 = nn.BatchNorm1d(out_size)

    def forward(self, x, data_length, **kwargs):
        t_sum = torch.sum(x, dim=1)[:, None, :]
        t_mean = t_sum / (data_length[:, None, None, None])
        #t_std = torch.sum((x - t_mean) ** 2, dim=1) / (data_length[:, None, None, ])
        #t_std = torch.sqrt(t_std)[:, None, :]
        t_std= (torch.sum(x*x, dim=1)[:, None, :]/(data_length[:, None, None, None]) - t_mean*t_mean)
        t_std = torch.sqrt(F.relu(t_std))
        # mask = mask
        # x[not mask] = x.min()
        x_min = torch.min(x)
        x = pack_padded_sequence(x, data_length.cpu(), batch_first=True)
        x, _ = pad_packed_sequence(x, padding_value=x_min.item(), batch_first=True)
        t_max = torch.max(x, dim=1)[0][:, None, :]
        x = torch.cat((t_mean, t_max, t_std), dim=1)
        # x = self.bn0(x)
        x = self.track_conv(x)
        x = x.view(x.size(0), -1)
        # x = self.dp(x)
        x = self.fc_1(x)
        x = F.hardswish(x)
        x = self.bn4(x)
        # x = self.dp(x)

        return x

class RAIRadarGestureClassifier_2(nn.Module):
    def __init__(self, cfar=True, track=True, ra_conv=True, attention=True, heads=4, in_size=(32, 32), hidden_size=(128, 64),
                 out_size=7, cfar_expand_channels=8, cfar_in_channel=1, dropout=0.5,
                 track_channels=(4, 8, 16), ra_feat_size=32, track_out_size=64, spatial_channels=(8, 16, 32)):
        super(RAIRadarGestureClassifier_2, self).__init__()
        self.sn = SpatialModel(ra_conv=ra_conv, in_size=in_size, num_channels=spatial_channels,
                               out_size=hidden_size[0], ra_feat_size=ra_feat_size, in_channels=1)
        self.need_track = track
        self.need_ra_conv = ra_conv
        self.hidden_size = hidden_size
        self.need_cfar = cfar
        self.bn0 = MaskedBatchNorm2d(1)
        self.stn = RAISTN(feat_size=in_size)
        if self.need_cfar:
            self.CFAR = CFAR(3, in_channel=cfar_in_channel, expand_channel=cfar_expand_channels)
        if track:
            self.tn = TRACK_NET(num_channels=track_channels, in_size=in_size, out_size=track_out_size)
            self.fc_2 = nn.Linear(track_out_size + 64, out_size)

        else:
            self.fc_2 = nn.Linear(64, out_size)
        if ra_conv:
            self.ra_feat_size = ra_feat_size
            self.temporal_net = TemporalModel2(hidden_size[0] - 2 * ra_feat_size, hidden_size[1], attention=attention,
                                              ra_conv=ra_conv, heads=heads,
                                              ra_feat_size=ra_feat_size)
        else:
            self.temporal_net = TemporalModel2(hidden_size[0], hidden_size[1], attention=attention, ra_conv=ra_conv)
        self.dropout_1 = nn.Dropout(p=dropout)

    def forward(self, x, track, data_length, **kwargs):
        bach_size = x.size(0)
        padded_lens = x.size(1)
        h = x.size(-2)
        w = x.size(-1)
        mask = torch.arange(x.size(1), dtype=torch.float32, device=x.device)
        mask = mask[None, :] < data_length[:, None]
        mask = mask.reshape(-1)
        x = self.stn(x, data_length, **kwargs)
        x = x.view(-1, 1, h, w)
        if self.need_cfar:
            x = self.CFAR(x, mask, **kwargs)


        if self.need_track:
            t = self.tn(x.view(bach_size, padded_lens, h, w), data_length, **kwargs)
            t = t.view(bach_size, -1)

        # x = x.view(bach_size * padded_lens, -1, h, w)
        x = self.sn(x, data_length, mask, **kwargs)
        feat_size = self.hidden_size[0] - 2 * self.ra_feat_size
        ra = x[:, :, feat_size:]
        x = x[:, :, :feat_size]
        #x = self.dropout_1(ra)
        x = ra
        x = self.temporal_net(x, data_length, mask, **kwargs)

        if self.need_track:
            x = torch.cat((x, t), dim=-1)
        x = self.fc_2(x)

        return x

class RAIRadarGestureClassifier(nn.Module):
    def __init__(self, cfar=True, track=True, ra_conv=True, attention=True, heads=4, in_size=(32, 32), hidden_size=(128, 64),
                 out_size=7, cfar_expand_channels=8, cfar_in_channel=1, dropout=0.5,
                 track_channels=(4, 8, 16), ra_feat_size=32, track_out_size=64, spatial_channels=(8, 16, 32)):
        super(RAIRadarGestureClassifier, self).__init__()
        self.sn = SpatialModel(ra_conv=ra_conv, in_size=in_size, num_channels=spatial_channels,
                               out_size=hidden_size[0], ra_feat_size=ra_feat_size, in_channels=1)
        self.need_track = track
        self.need_ra_conv = ra_conv
        self.hidden_size = hidden_size
        self.need_cfar = cfar
        self.bn0 = MaskedBatchNorm2d(1)
        if self.need_cfar:
            self.CFAR = CFAR(3, in_channel=cfar_in_channel, expand_channel=cfar_expand_channels)
        if track:
            self.tn = TRACK_NET(num_channels=track_channels, in_size=in_size, out_size=track_out_size)
            self.fc_2 = nn.Linear(track_out_size + hidden_size[1], out_size)

        else:
            self.fc_2 = nn.Linear(hidden_size[1], out_size)
        if ra_conv:
            self.ra_feat_size = ra_feat_size
            self.temporal_net = TemporalModel(hidden_size[0] - 2 * ra_feat_size, hidden_size[1], attention=attention,
                                              ra_conv=ra_conv, heads=heads,
                                              ra_feat_size=ra_feat_size)
        else:
            self.temporal_net = TemporalModel(hidden_size[0], hidden_size[1], attention=attention, ra_conv=ra_conv)
        self.dropout_1 = nn.Dropout(p=dropout)

    def forward(self, x, track, data_length, **kwargs):
        bach_size = x.size(0)
        padded_lens = x.size(1)
        h = x.size(-2)
        w = x.size(-1)
        mask = torch.arange(x.size(1), dtype=torch.float32, device=x.device)
        mask = mask[None, :] < data_length[:, None]
        mask = mask.reshape(-1)
        x = x.view(-1, 1, h, w)
        if self.need_cfar:
            x = self.CFAR(x, mask, **kwargs)
        if self.need_track:
            t = self.tn(x.view(bach_size, padded_lens, h, w), data_length, **kwargs)
            t = t.view(bach_size, -1)

        # x = x.view(bach_size * padded_lens, -1, h, w)
        # x = self.bn0(x, mask=mask)
        x = self.sn(x, data_length, mask, **kwargs)
        x = self.dropout_1(x)
        ra = None
        if self.need_ra_conv:
            feat_size = self.hidden_size[0] - 2 * self.ra_feat_size
            ra = x[:, :, feat_size:]
            x = x[:, :, :feat_size]
        x = self.temporal_net(x, ra, data_length, mask, **kwargs)

        if self.need_track:
            x = torch.cat((x, t), dim=-1)
        x = self.fc_2(x)

        return x


class DRAI_2DCNNLSTM_DI_GESTURE_BETA(nn.Module):
    def __init__(self, cfar=True, track=True):
        super(DRAI_2DCNNLSTM_DI_GESTURE_BETA, self).__init__()
        self.sn = SIGNAL_NET_BETA(cfar)
        self.track = track
        if track:
            self.tn = TRACK_NET()
            self.fc_2 = nn.Linear(384, 7)
        else:
            self.fc_2 = nn.Linear(256, 7)
        self.fc_4 = nn.Linear(1024, 256)
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=1, batch_first=True)
        self.multi_head_attention = MultiHeadAttention(query_size=256, key_size=256, value_size=256, num_hidden=256,
                                                       num_heads=8, dropout=nn.Dropout(p=0.5))
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, tracks, data_length, **kwargs):
        bach_size = len(x)
        x = self.sn(x, data_length, **kwargs)
        x = x.view(-1, 1024)
        x = self.fc_4(x)
        x = F.leaky_relu(x)
        x = x.view(bach_size, -1, 256)
        x = self.dropout(x)
        x = pack_padded_sequence(x, data_length, batch_first=True)
        output, (h_n, c_n) = self.lstm(x)
        output, out_len = pad_packed_sequence(output, batch_first=True)
        final_state = h_n[-1]
        final_state = final_state.view(bach_size, 1, -1)

        x = self.multi_head_attention(final_state, output, output, out_len)
        x = x.view(bach_size, -1)
        x = self.bn1(x + torch.squeeze(final_state))
        x = F.relu(x)
        if self.track:
            t = self.tn(tracks, **kwargs)
            t = t.view(bach_size, -1)
            # x = self.bn1(x)
            x = torch.cat((x, t), dim=1)

        x = self.dropout(x)
        x = self.fc_2(x)

        return x


class DRAI_2DCNNLSTM_DI_GESTURE_DELTA(nn.Module):
    def __init__(self, cfar=True, track=True, ra_conv=True, attention=True, in_size=(32, 32), hidden_size=128,
                 out_size=7, cfar_expand_channels=8, cfar_in_channel=1,
                 track_channels=(4, 8, 16), ra_feat_size=32, track_out_size=64, spatial_channels=(8, 16, 32)):
        super(DRAI_2DCNNLSTM_DI_GESTURE_DELTA, self).__init__()
        self.sn = SpatialModel(cfar=cfar, cfar_expand_channels=cfar_expand_channels, ra_conv=ra_conv, in_size=in_size,
                               num_channels=spatial_channels, cfar_in_channel=cfar_in_channel,
                               out_size=hidden_size, ra_feat_size=ra_feat_size)
        self.need_track = track
        if track:
            self.tn = TRACK_NET(num_channels=track_channels, in_size=in_size, out_size=track_out_size)
            self.fc_2 = nn.Linear(track_out_size + hidden_size, out_size)
        else:
            self.fc_2 = nn.Linear(hidden_size, out_size)
        self.lstm = nn.LSTM(input_size=hidden_size // 2, hidden_size=hidden_size // 2, num_layers=1, batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=hidden_size // 2, hidden_size=hidden_size // 2, num_layers=1, batch_first=True)

        self.need_attention = attention
        if attention:
            self.multi_head_attention = MultiHeadAttention(query_size=hidden_size, key_size=hidden_size,
                                                           value_size=hidden_size, num_hidden=hidden_size,
                                                           num_heads=4, dropout=nn.Dropout(p=0.5))
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        # self.ln1 = MaskLayerNorm(hidden_size)
        # self.ln2 = nn.LayerNorm(hidden_size)
        self.ln = MyLayerNorm()
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x, tracks, data_length, **kwargs):
        bach_size = len(x)
        x = self.sn(x, data_length, **kwargs)

        x = self.dropout1(x)
        x = x.reshape(x.shape[0], x.shape[1], 2, -1)
        ra = x[:, :, 1, :]
        x = pack_padded_sequence(x[:, :, 0, :], data_length.cpu(), batch_first=True)
        output, (h_n, _) = self.lstm(x)
        ra = pack_padded_sequence(ra, data_length.cpu(), batch_first=True)
        output_2, (h_n_2, _) = self.lstm_2(ra)
        final_state = torch.cat((h_n[-1], h_n_2[-1]), dim=-1)
        # final_state = self.ln1(final_state)
        # output = self.ln1(output)
        # final_state = self.bn1(final_state)
        final_state = final_state[:, None, :]

        if self.need_attention:
            output, out_len = pad_packed_sequence(output, batch_first=True)
            output_2, out_len = pad_packed_sequence(output_2, batch_first=True)
            output = torch.cat((output, output_2), dim=-1)
            # output = self.ln(output)
            x = self.multi_head_attention(final_state, output, output, data_length)
            x = final_state + self.dropout2(x)
        else:
            x = final_state
        x = x.view(bach_size, -1)
        x = self.bn2(x)
        # x = self.dropout2(x)
        if self.need_track:
            t = self.tn(tracks, **kwargs)
            t = t.view(bach_size, -1)
            x = torch.cat((x, t), dim=-1)

        x = self.fc_2(x)

        return x


class DRAI_2DCNNLSTM_DI_GESTURE(nn.Module):
    def __init__(self):
        super(DRAI_2DCNNLSTM_DI_GESTURE, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.bn3 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
        self.fc_2 = nn.Linear(288, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc_3 = nn.Linear(128, 7)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, data_length):
        x = x.view(-1, 1, 32, 32)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = x.view(-1, 288)
        x = self.fc_2(x)
        x = self.dropout(x)
        x = x.view(len(data_length), -1, 128)
        x = pack_padded_sequence(x, data_length, batch_first=True)
        output, (h_n, c_n) = self.lstm(x)
        # output, out_len = pad_packed_sequence(output, batch_first=True)
        x = h_n[-1]
        x = self.fc_3(x)
        # x = self.softmax(x)

        return x


class DiGesture(nn.Module):
    def __init__(self, input_size=(32, 32), channel_num=(8, 16, 32), spatial_feat_size=128, hidden_size=128):
        super(DiGesture, self).__init__()
        self.conv1 = nn.Conv1d(1, channel_num[0], 3)
        self.bn1 = nn.BatchNorm1d(channel_num[0])
        self.input_size = input_size
        self.spatial_feat_size = spatial_feat_size
        self.conv2 = nn.Conv1d(channel_num[0], channel_num[1], 3)
        self.bn2 = nn.BatchNorm1d(channel_num[1])
        self.conv3 = nn.Conv1d(channel_num[1], channel_num[2], 3)
        self.bn3 = nn.BatchNorm1d(channel_num[2])
        self.maxpool = nn.MaxPool1d(2, ceil_mode=True)
        self.lstm = nn.LSTM(input_size=spatial_feat_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        linear_input = channel_num[-1] * get_after_conv_size(size=input_size[0], kernel_size=3, layer=3,
                                                             reduction=1) \
                       * get_after_conv_size(size=input_size[1], kernel_size=3, layer=3, reduction=1)
        self.fc_2 = nn.Linear(linear_input, spatial_feat_size)
        self.dropout = nn.Dropout(p=0.5)
        self.fc_3 = nn.Linear(hidden_size, 7)
        # self.flatten = nn.Flatten
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, data_length):
        batch_size = x.size(0)
        padding_len = x.size(1)
        h = x.size(2)
        w = x.size(3)
        x = x.view(-1, 1, h, w)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        # x = self.maxpool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        # x = self.maxpool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        # x = self.maxpool(x)
        x = x.view(batch_size, padding_len, -1)
        x = self.fc_2(x)
        x = self.dropout(x)
        # x = x.view(batch_size, padding_len, 128)
        x = pack_padded_sequence(x, data_length, batch_first=True)
        output, (h_n, c_n) = self.lstm(x)
        # output, out_len = pad_packed_sequence(output, batch_first=True)
        x = h_n[-1]
        x = self.fc_3(x)
        # x = self.softmax(x)

        return x


if __name__ == '__main__':
    net = DRAI_2DCNNLSTM_DI_GESTURE_DENOISE()
    ipt = torch.zeros((128, 128, 32, 32))
    net.forward(ipt, np.random.random(size=(1, 127, 128)))
