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


class SpatialAttBlock(nn.Module):
    def __init__(self, in_channel=2, channel_num=(16, 8, 64), diff=True, input_size=32, out_size=32, dim=-1, dropout=0.5,
                 eps=1e-5):
        super(SpatialAttBlock, self).__init__()
        if dim == -1:
            self.single_dim_conv = SingleDimConv3(in_channel * 3, channel_num)
            linear_in = 128
        else:
            self.single_dim_conv = SingleDimConv3(in_channel * 3, channel_num)
            linear_in = 128
        self.dim = dim
        self.dp1 = nn.Dropout(0.2)
        self.dp2 = nn.Dropout(0.5)
        self.need_diff = diff
        if diff:
            self.diff = DifferentialNetwork(in_size=linear_in, out_size=out_size)
        else:
            self.fc_1 = nn.Sequential(
                nn.Linear(linear_in, out_size, bias=True),
                nn.ReLU()
            )
        # self.stn = RAISTN(feat_size=32)

    def forward(self, x, batch_size, padding_len, data_len=None, mask=None, **kwargs):
        max_x = torch.max(x, dim=self.dim)[0]
        avg_x = torch.mean(x, dim=self.dim)
        std_x = torch.std(x, dim=self.dim)
        avg_max = torch.cat((max_x, avg_x, std_x), dim=1)
        #avg_max = self.dp(avg_max)
        # avg_max = self.bn1(avg_max, mask=mask)
        #avg_max = self.bn1(avg_max)
        score = self.single_dim_conv(avg_max)
        score = score.view(len(x), -1)
        score = self.dp1(score)
        if self.need_diff:
            score = self.diff(score, batch_size, padding_len)
        else:
            score = self.fc_1(score)
            score = score.view(batch_size, padding_len, -1)
        score = self.dp2(score)
        return score


class SpatialModel(nn.Module):
    def __init__(self, num_channels, ra_conv=True, diff=False, in_size=(32, 32), rda_feat_size=32, conv2d_feat_size=64,
                 in_channels=1, dropout=0.5):
        super(SpatialModel, self).__init__()

        self.need_ra_conv = ra_conv

        self.dp = nn.Dropout(dropout)
        self.in_channels = in_channels

        # 是否需要角度距离注意力
        # sp_feat_size = out_size
        if ra_conv:
            self.range_att = SpatialAttBlock(dim=-1, in_channel=in_channels, input_size=in_size[0], diff=diff,
                                             out_size=rda_feat_size)
            self.angel_att = SpatialAttBlock(dim=-2, in_channel=in_channels, input_size=in_size[1], diff=diff,
                                             out_size=rda_feat_size)
        else:
            linear_input = num_channels[-1] * get_after_conv_size(size=in_size[0], kernel_size=3, layer=3, padding=0,
                                                                  reduction=2) ** 2
            self.spatial_conv = CNN2d3Layers(in_channels, num_channels)
            if diff:
                self.diff = DifferentialNetwork(in_size=linear_input, out_size=conv2d_feat_size)
            else:
                self.fc_1 = nn.Sequential(
                    nn.Linear(linear_input, conv2d_feat_size, bias=False),
                    nn.ReLU()
                )
            self.need_diff = diff

    def forward(self, rai, data_len):
        batch_size = rai.size(0)
        padded_len = rai.size(1)
        h = rai.size(-2)
        w = rai.size(-1)
        rai = rai.view(-1, self.in_channels, h, w)
        range_feats = None
        angel_feats = None
        combine_feats = None
        if self.need_ra_conv:
            range_feats = self.range_att(rai, batch_size=batch_size, padding_len=padded_len, data_len=data_len)
            angel_feats = self.angel_att(rai, batch_size=batch_size, padding_len=padded_len, data_len=data_len)
            #combine_feats = torch.cat((range_feats, angel_feats), dim=-1)
        else:
            rai = self.spatial_conv(rai)
            rai = rai.view(rai.size(0), -1)
            rai = self.dp(rai)
            if self.need_diff:
                combine_feats = self.diff(rai, batch_size, padded_len)
            else:
                combine_feats = self.fc_1(rai)
                combine_feats = combine_feats.view(batch_size, padded_len, -1)

        return combine_feats, range_feats, angel_feats


def lstm_func(lstm_block, data, data_lens):
    if data_lens is not None:
        x = pack_padded_sequence(data, data_lens.cpu(), batch_first=True)
        output, (h_n, _) = lstm_block(x)
        output, out_len = pad_packed_sequence(output, batch_first=True)
    else:
        output, (h_n, _) = lstm_block(data)
    return output, h_n[-1]


class DifferentialNetwork(nn.Module):
    def __init__(self, in_size, out_size):
        super(DifferentialNetwork, self).__init__()
        self.linear_sp1 = nn.Sequential(
            nn.Linear(in_size, out_size, bias=False),
            nn.LayerNorm(out_size),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(out_size, out_size // 2, bias=True),

            nn.ReLU(),
            # nn.BatchNorm1d(in_size//2),
        )
        self.linear_sp2 = nn.Sequential(
            nn.Linear(out_size // 2, out_size // 2, bias=True),
            # nn.LayerNorm(out_size//2),
            nn.ReLU(),
            nn.Linear(out_size // 2, out_size, bias=False),
            nn.LayerNorm(out_size),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )

    def forward(self, x, batch_size, padded_len):
        x = self.linear_sp1(x)

        x = x.view(batch_size, padded_len, -1)
        x = x[:, 1:] - x[:, :-1]
        x = self.linear_sp2(x)
        return x


class TemporalModel(nn.Module):
    def __init__(self, feat_size1, attention=True, diff=True, ra_conv=True, conv_2d=False, feat_size2=None, heads=2,
                 dropout=0.5):
        super(TemporalModel, self).__init__()
        hidden_size = None
        if conv_2d:
            self.lstm_1 = nn.LSTM(input_size=feat_size1, hidden_size=feat_size1, num_layers=1, batch_first=True)
            hidden_size = feat_size1
        if ra_conv:
            self.lstm_2 = nn.LSTM(input_size=feat_size2, hidden_size=feat_size2, num_layers=1,
                                  batch_first=True)
            self.lstm_3 = nn.LSTM(input_size=feat_size2, hidden_size=feat_size2, num_layers=1,
                                  batch_first=True)
            hidden_size = feat_size2 * 2
            # self.funsion_fc =  nn.Sequential(
            #             nn.Linear(hidden_size, hidden_size, bias=False),
            #             nn.LayerNorm(hidden_size),
            #             nn.ReLU(),)

        self.need_attention = attention
        if attention:
            self.multi_head_attention = MultiHeadAttention(query_size=hidden_size, key_size=hidden_size,
                                                                   value_size=hidden_size, num_hidden=hidden_size,
                                                                   num_heads=heads, dropout=0.2, bias=False)
            # self.multi_head_attention = nn.MultiheadAttention(embed_dim=out_size, kdim=hidden_size, vdim=hidden_size,
            #                                                   batch_first=True, dropout=0.5, num_heads=heads,
            #                                                   bias=False)
            #self.w_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        # self.bn_mid = nn.BatchNorm1d(hidden_size)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.dropout_3 = nn.Dropout(p=dropout)
        self.need_diff = diff

    def forward(self, x1=None, x2=None, x3=None, data_lens=None, mask=None, **kwargs):
        if self.need_diff and data_lens is not None:
            data_lens = data_lens - 1

        # x1 = self.dropout_1(x1)
        if x1 is not None:
            output, final_state = lstm_func(self.lstm_1, x1, data_lens)
        else:
            # x2 = self.dropout_2(x2)
            # x3 = self.dropout_2(x3)
            # ra = custom_dropout(ra, 0.5, self.training)
            output_2, final_state_2 = lstm_func(self.lstm_2, x2, data_lens)
            output_3, final_state_3 = lstm_func(self.lstm_3, x3, data_lens)
            output = torch.cat((output_2, output_3), dim=-1)
            final_state = torch.cat((final_state_2, final_state_3), dim=-1)
            # final_state = self.funsion_fc(final_state)
        final_state = final_state[:, None, :]

        if self.need_attention:
            # if data_lens is not None:
            #     mask = torch.arange((x1.size(-2)), dtype=torch.float32, device=x1.device)
            #     mask = mask[None, :] >= data_lens[:, None]
            # x1 = final_state + self.multi_head_attention(final_state, output, output, key_padding_mask=mask,
            #                                              need_weights=False)[0]
            #output = self.dropout_1(output)
            x1 = self.multi_head_attention(final_state, output, output, data_lens)
            x1 = final_state + x1
            #x1 = F.relu(x1)
            #x1 = self.w_o(x1)
            # x1 = torch.sum(x1, dim=1)/data_lens[:, None]
        else:
            x1 = final_state

        # x1 = self.dropout_1(x1)
        x1 = x1.view(len(x1), -1)
        # x1 = self.bn1(x1)
        x1 = self.bn2(x1)
        x1 = F.hardswish(x1)

        return x1


class GlobalConv(nn.Module):
    def __init__(self, num_channels, in_channels=3, in_size=(32, 32), out_size=64, dropout=0.5):
        super(GlobalConv, self).__init__()
        self.track_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, num_channels[0], kernel_size=(3, 3), bias=False),

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

    def forward(self, track):
        x = self.track_conv(track)
        x = x.view(x.size(0), -1)
        x = self.dp(x)
        x = self.fc_1(x)
        x = self.bn4(x)
        x = F.hardswish(x)

        # x = self.dp(x)
        # x = self.dp0(x)
        return x


class RAIRadarGestureClassifier(nn.Module):
    def __init__(self, cfar=True, track=True, ra_conv=True, diff=True, attention=True, heads=4, in_size=(32, 32, 32),
                 in_channel=1,conv2d_feat_size = 64,
                 out_size=7, cfar_expand_channels=8, cfar_in_channel=1, dropout=0.5,
                 track_channels=(4, 8, 16), ra_feat_size=32, track_out_size=64, spatial_channels=(8, 16, 32)):
        super(RAIRadarGestureClassifier, self).__init__()

        self.need_track = track
        self.need_ra_conv = ra_conv
        self.ra_feat_size = ra_feat_size
        self.sn = SpatialModel(ra_conv=ra_conv, in_size=in_size, num_channels=spatial_channels, diff=diff,
                               conv2d_feat_size=conv2d_feat_size, rda_feat_size=ra_feat_size, in_channels=in_channel)
        fc_feat_size = 0
        if track:
            self.tn = GlobalConv(in_channels=in_channel, num_channels=track_channels, in_size=in_size,
                                 out_size=track_out_size)
            fc_feat_size = fc_feat_size + track_out_size

        if ra_conv:
            self.temporal_net = TemporalModel(feat_size1=conv2d_feat_size, attention=attention, conv_2d=False,
                                              ra_conv=ra_conv, heads=heads, diff=diff,
                                              feat_size2=ra_feat_size)
            fc_feat_size = fc_feat_size + ra_feat_size * 2
        else:
            self.temporal_net = TemporalModel(feat_size1=conv2d_feat_size, attention=attention, diff=diff,
                                              ra_conv=ra_conv, conv_2d=True)
            fc_feat_size = fc_feat_size + conv2d_feat_size
        self.fc_2 = nn.Linear(fc_feat_size, out_size)
        self.dropout_1 = nn.Dropout(p=dropout)

    def forward(self, rai, track, data_length, **kwargs):
        bach_size = rai.size(0)
        padded_lens = rai.size(1)
        h = rai.size(-2)
        w = rai.size(-1)

        rai = rai.view(bach_size, padded_lens, -1, h, w)
        t = None
        if self.need_track:
            t = self.tn(track)
            t = t.view(bach_size, -1)
        x, r, a = self.sn(rai, data_length)
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
