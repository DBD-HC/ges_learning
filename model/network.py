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
import visdom

class CFAR(nn.Module):
    def __init__(self, window_size, in_channel=1, expand_channel=8):
        super(CFAR, self).__init__()
        if isinstance(window_size, tuple):
            padding = ((window_size[1] - 1) // 2, (window_size[2] - 1) // 2)
        else:
            padding = (window_size - 1) // 2

        # conv_weight = torch.from_numpy(init_weight(window_size, 6)).unsqueeze(0).unsqueeze(0)
        self.conv1 = nn.Conv2d(in_channel, expand_channel, padding=padding, padding_mode='circular',
                               kernel_size=window_size)
        self.conv2 = nn.Conv2d(expand_channel, expand_channel, padding=padding, padding_mode='circular',
                               kernel_size=window_size)
        self.conv3 = nn.Conv2d(expand_channel, in_channel, padding=padding, padding_mode='circular',
                               kernel_size=window_size)
        # self.conv1.weight = torch.nn.Parameter(conv_weight.float())
        # self.visdom = visdom.Visdom(env='CFAR', port=6006)
        self.bn1 = MaskedBatchNorm2d(in_channel)
        self.av_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.relu = torch.nn.LeakyReLU()

    def forward(self, x, mask=None, **kwargs):
        thr = self.conv1(x)
        thr = self.relu(thr)
        # thr = self.av_pool(thr)
        thr = self.conv2(thr)
        thr = self.relu(thr)
        # thr = self.av_pool(thr)
        thr = self.conv3(thr)
        thr = self.relu(thr)
        # thr = self.av_pool(thr)
        thr = x - thr
        thr = self.bn1(thr, mask=mask)
        thr = self.relu(thr)
        #         if not self.training and 'indexes' in kwargs:
        #             indexes = kwargs['indexes']
        #             ids = str(kwargs['epoch'] % 10)
        #             if 367 in indexes:
        #                 index = indexes.index(333)
        #                 x_t = x.view(len(indexes), -1, 32, 32)
        #                 th_t = thr.view(len(indexes), -1, 32, 32)
        #                 self.visdom.heatmap(x_t[index][10], win=ids + '_origin',
        #                                     opts=dict(title=ids + 'origin' + str(kwargs['epoch'])))
        #                 self.visdom.heatmap(th_t[index][10], win=ids + '_cfar',
        #                                     opts=dict(title=ids + 'cfar ' + str(kwargs['epoch'])))

        return thr


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
    def __init__(self, in_channel=2, channel_num=(2, 4, 4), input_size=32, out_size=32, dim=-1, dropout=0.5, eps=1e-5):
        super(SpatialAttBlock, self).__init__()
        self.single_dim_conv = SingleDimConv(in_channel * 2, channel_num)

        # self.down_sample = nn.Conv1d(in_channel*2, channel_num[-1], kernel_size=1, bias=False)
        linear_in = get_after_conv_size(size=input_size, kernel_size=3, layer=3, padding=1,
                            reduction=1)*channel_num[-1]
        self.dim = dim
        self.fc1 = nn.Linear(linear_in, out_size, bias=False)
        self.bn1 = MaskedBatchNorm1d(channel_num[-1])
        self.bn2 = MaskedBatchNorm1d(input_size * channel_num[-1])
        self.bn3 = MaskedBatchNorm1d(out_size)
        self.ln1 = nn.LayerNorm(out_size * channel_num[-1])
        self.ln2 = nn.LayerNorm(out_size)
        # self.visdom = visdom.Visdom(env='SpatialAttBlock', port=6006)
        self.dp = nn.Dropout(dropout)

    def forward(self, x, batch_size, padding_len, mask=None, **kwargs):
        # feat_len = x.size(self.dim)
        max_x = torch.max(x, dim=self.dim)[0]
        avg_x = torch.mean(x, dim=self.dim)
        avg_max = torch.cat((max_x, avg_x), dim=1)
        score = self.single_dim_conv(avg_max, mask)
        # avg_max = self.bn1(avg_max, mask=mask)
        # score = self.range_conv(avg_max)
        # score = self.ln(score, batch_size, padding_len)
        # score = score + self.down_sample(avg_max)
        # score = self.bn1(score, mask=mask)
        #         if not self.training and 'indexes' in kwargs:
        #             indexes = kwargs['indexes']
        #             ids = str(kwargs['epoch'] % 10)
        #             if self.dim == -1:
        #                 ids = ids + '_range'
        #             else:
        #                 ids = ids + '_angel'
        #             if 367 in indexes:
        #                 index = indexes.index(333)
        #                 x_t = x.view(len(indexes), -1, 32, 32)
        #                 # att_x_t = att_x.view(len(indexes), -1, 32, 32)
        #                 score_t = score.view(len(indexes), -1, 4, 32)
        #                 self.visdom.heatmap(x_t[index][10], win=ids + '_origin',
        #                                     opts=dict(title=ids + 'origin' + str(kwargs['epoch'])))
        #                 self.visdom.heatmap(score_t[index][10], win=ids + '_att_score',
        #                                     opts=dict(title=ids + 'att score' + str(kwargs['epoch'])))
        # score = self.ln(score, batch_size, padding_len)
        # score = score.view(len(x), -1)
        # score = self.bn1(score, mask=mask)
        score = score.view(len(x), -1)
        # score = F.relu(score)
        # score = self.dp(score)
        score = self.dp(score)
        score = self.fc1(score)
        score = self.bn3(score, mask=mask)
        # score = self.ln2(score)
        # score = F.relu(score)
        score = score.view(batch_size, padding_len, -1)
        score = F.leaky_relu(score)
        return score


class SpatialModel(nn.Module):
    def __init__(self, num_channels, cfar=True, cfar_in_channel=1, cfar_expand_channels=8, ra_conv=True,
                 in_size=(32, 32), ra_feat_size=32,
                 out_size=128, in_channels=1,
                 dropout=0.5):
        super(SpatialModel, self).__init__()
        self.need_cfar = cfar
        self.need_ra_conv = ra_conv
        self.dp = nn.Dropout(p=0.5)
        self.CFAR = CFAR(3, in_channel=cfar_in_channel, expand_channel=cfar_expand_channels)

        if cfar:
            self.spatial_conv = CNN2d3LayersV2(in_channels * 2, num_channels)
        else:
            self.spatial_conv = CNN2d3Layers(in_channels, num_channels)
        self.dp = nn.Dropout(dropout)

        # 是否需要角度距离注意力
        linear_input = num_channels[-1] * get_after_conv_size(size=in_size[0], kernel_size=3, layer=3, padding=1,
                                                              reduction=2) \
                       * get_after_conv_size(size=in_size[1], kernel_size=3, layer=3, padding=1, reduction=2)
        if ra_conv:
            self.range_att = SpatialAttBlock(dim=-1, in_channel=in_channels, input_size=in_size[0], out_size=ra_feat_size)
            self.angel_att = SpatialAttBlock(dim=-2, in_channel=in_channels, input_size=in_size[1], out_size=ra_feat_size)
            self.fc_1 = nn.Linear(linear_input, out_size - 2 * ra_feat_size, bias=False)
            self.fc_2 = nn.Linear(out_size, out_size // 2, bias=False)
            self.bn4 = MaskedBatchNorm1d(out_size - 2 * ra_feat_size)
            self.bn5 = MaskedBatchNorm1d(out_size // 2)
            self.dp_2 = nn.Dropout(dropout)
        else:
            self.fc_1 = nn.Linear(linear_input, out_size, bias=False)
            self.bn4 = MaskedBatchNorm1d(out_size)

        # self.rang_stn = SpatialTransLayer(in_size[0])
        # self.angel_stn = SpatialTransLayer(in_size[1])
        # self.spatial_stn = SpatialTransLayer(64)
        self.ln4 = nn.LayerNorm(out_size - 2 * ra_feat_size)

    def forward(self, x, data_length, **kwargs):
        padded_len = x.size(1)
        batch_size = len(data_length)
        h = x.size(-2)
        w = x.size(-1)

        mask = torch.arange(padded_len, dtype=torch.float32, device=x.device)
        # valid_length = valid_length.to(input.device)
        mask = mask[None, :] < data_length[:, None]
        mask = mask.reshape(-1)
        x = x.view(-1, 1, h, w)
        if self.need_ra_conv:
            # x = x.view(-1, h, w)
            range_score = self.range_att(x, batch_size=batch_size, padding_len=padded_len, mask=mask, **kwargs)
            # range_score = self.rang_stn(range_score)
            angel_score = self.angel_att(x, batch_size=batch_size, padding_len=padded_len, mask=mask, **kwargs)
            # angel_score = self.angel_stn(angel_score)
        if self.need_cfar:
            detect = self.CFAR(x, mask, **kwargs)
            detect = detect.view(-1, 1, h, w)
            x = torch.cat((x, detect), dim=1)

        # 提取RA二维频谱图特征
        x = self.spatial_conv(x, mask)
        x = x.view(x.size(0), -1)
        x = self.dp(x)
        x = self.fc_1(x)
        x = self.bn4(x, mask=mask)
        # x = self.dp(x)
        # x  = self.ln3(x, batch_size, padded_len)
        # x = self.ln4(x)
        x = F.leaky_relu(x)
        x = x.view(len(data_length), padded_len, -1)
        # x = self.spatial_stn(x)
        if self.need_ra_conv:
            x = torch.cat((x, range_score, angel_score), dim=-1)

        return x


class LstmAttention(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, attention=True, heads=4, dropout=0.5):
        super(LstmAttention, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        if attention:
            self.need_attention = attention
            self.multi_head_attention = MultiHeadAttention(query_size=hidden_size, key_size=hidden_size,
                                                           value_size=hidden_size, num_hidden=out_size,
                                                           num_heads=heads, dropout=nn.Dropout(p=dropout))
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, x, data_lens, **kwargs):
        x = self.dropout_1(x)
        x = pack_padded_sequence(x, data_lens.cpu(), batch_first=True)
        output, (h_n, _) = self.lstm(x)
        output, out_len = pad_packed_sequence(output, batch_first=True)
        final_state = h_n[-1]
        final_state = final_state[:, None, :]
        if self.need_attention:
            x = self.multi_head_attention(final_state, output, output, data_lens)
            x = final_state + self.dropout_1(x)
        else:
            x = final_state
        x = torch.squeeze(x)
        return x


class TemporalModel(nn.Module):
    def __init__(self, input_size, out_size, attention=True, ra_conv=True, ra_feat_size=None, heads=4, dropout=0.5):
        super(TemporalModel, self).__init__()
        self.lstm_1 = nn.LSTM(input_size=input_size, hidden_size=input_size, num_layers=1, batch_first=True)
        hidden_size = input_size
        if ra_conv:
            self.lstm_2 = nn.LSTM(input_size=ra_feat_size*2, hidden_size=ra_feat_size*2, num_layers=1,
                                  batch_first=True)
            hidden_size = input_size + ra_feat_size * 2
        if attention:
            self.need_attention = attention
            self.multi_head_attention = MultiHeadAttention(query_size=hidden_size, key_size=hidden_size,
                                                           value_size=hidden_size, num_hidden=out_size,
                                                           num_heads=heads, dropout=nn.Dropout(p=0.2))
        else:
            self.fc = nn.Linear(hidden_size, out_size)
        self.bn = nn.BatchNorm1d(out_size)
        self.ln_1 = nn.LayerNorm(input_size)
        if ra_conv:
            self.ln_2 = nn.LayerNorm(ra_feat_size * 2)
        self.ln_3 = nn.LayerNorm(out_size)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.dropout_3 = nn.Dropout(p=dropout)
        self.dropout_x =  nn.Dropout(p=dropout)
        self.dropout_ra = nn.Dropout(p=dropout)

    def forward(self, x, ra=None, data_lens=None, **kwargs):
        # x = self.dropout_x(x)
        k_x = torch.cat((x, ra), dim=-1)
        q_x = torch.mean(k_x, dim=-2)[:, None, :]
        packed_x = pack_padded_sequence(x, data_lens.cpu(), batch_first=True)
        output, (h_n, _) = self.lstm_1(packed_x)
        output, out_len = pad_packed_sequence(output, batch_first=True)
        # output = self.ln_1(output)
        final_state = h_n[-1]

        if ra is not None:
            # ra = self.dropout_ra(ra)
            packed_ra = pack_padded_sequence(ra, data_lens.cpu(), batch_first=True)
            output_2, (h_n_2, _) = self.lstm_2(packed_ra)
            output_2, out_len = pad_packed_sequence(output_2, batch_first=True)
            # output_2 = self.ln_2(output_2)
            output = torch.cat((output, output_2), dim=-1)
            final_state = torch.cat((final_state, h_n_2[-1]), dim=-1)
        final_state = final_state[:, None, :]
        # output = self.dropout_2(output)
        if self.need_attention:
            x = self.multi_head_attention(q_x, k_x, output, data_lens)
            # x = torch.cat((final_state ,x), dim=-1)
            x = final_state + self.dropout_2(x)
        else:
            x = self.dropout_1(final_state)
            x = self.fc(x)

        x = torch.squeeze(x)
        # x = self.dropout_2(x)
        x = self.bn(x)

        return x


class TRACK_NET(nn.Module):
    def __init__(self, num_channels, in_channels=1, in_size=(32, 32), out_size=64, dropout=0.5):
        super(TRACK_NET, self).__init__()
        self.track_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_channels[0], kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(num_channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(num_channels[0], num_channels[1], kernel_size=(3, 3), bias=False),
            nn.BatchNorm2d(num_channels[1]),
            nn.ReLU(),
            nn.AvgPool2d(2, ceil_mode=True),
            nn.Conv2d(num_channels[1], num_channels[2], kernel_size=(3, 3), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_channels[2]),
            nn.AvgPool2d(2, ceil_mode=True),
        )
        self.dp = nn.Dropout(p=dropout)
        linear_input = num_channels[-1] * get_after_conv_size(size=in_size[0], kernel_size=3, layer=3, reduction=2) \
                       * get_after_conv_size(size=in_size[1], kernel_size=3, layer=3, reduction=2)
        self.fc_1 = nn.Linear(linear_input, out_size)
        self.bn4 = nn.BatchNorm1d(out_size)

    def forward(self, x, **kwargs):
        x = self.track_conv(x)
        x = x.view(x.size(0), -1)
        x = self.dp(x)
        x = self.fc_1(x)
        x = self.bn4(x)
        # x = self.dp(x)
        x = F.relu(x)

        return x


class RAIRadarGestureClassifier(nn.Module):
    def __init__(self, cfar=True, track=True, ra_conv=True, attention=True, in_size=(32, 32), hidden_size=(128, 64),
                 out_size=7, cfar_expand_channels=8, cfar_in_channel=1,dropout=0.5,
                 track_channels=(4, 8, 16), ra_feat_size=32, track_out_size=64, spatial_channels=(8, 16, 32)):
        super(RAIRadarGestureClassifier, self).__init__()
        self.sn = SpatialModel(cfar=cfar, cfar_expand_channels=cfar_expand_channels, ra_conv=ra_conv, in_size=in_size,
                               num_channels=spatial_channels, cfar_in_channel=cfar_in_channel,
                               out_size=hidden_size[0], ra_feat_size=ra_feat_size)
        self.need_track = track
        self.need_ra_conv = ra_conv
        self.hidden_size = hidden_size
        if track:
            self.tn = TRACK_NET(num_channels=track_channels, in_size=in_size, out_size=track_out_size)
            self.fc_2 = nn.Linear(track_out_size + hidden_size[1], 32)

        else:
            self.fc_2 = nn.Linear(hidden_size[1], 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc_3 = nn.Linear(32, out_size)
        if ra_conv:
            self.ra_feat_size = ra_feat_size
            self.temporal_net = TemporalModel(hidden_size[0] - 2 * ra_feat_size, hidden_size[1], attention=attention, ra_conv=ra_conv,
                                              ra_feat_size=ra_feat_size)
        else:
            self.temporal_net = TemporalModel(hidden_size[0], hidden_size[1], attention=attention, ra_conv=ra_conv)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.ln1 = MyLayerNorm(hidden_size)


    def forward(self, x, tracks, data_length, **kwargs):
        bach_size = len(x)

        x= self.sn(x, data_length, **kwargs)
        x = self.dropout_1(x)
        ra = None
        if self.need_ra_conv:
            feat_size = self.hidden_size[0] - 2 * self.ra_feat_size
            ra = x[:, :, feat_size:]
            x = x[:, :, :feat_size]
        x = self.temporal_net(x, ra, data_length, **kwargs)

        if self.need_track:
            t = self.tn(tracks, **kwargs)
            t = t.view(bach_size, -1)
            x = torch.cat((x, t), dim=-1)
        x = self.fc_2(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc_3(x)

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


class DRAI_2DCNNLSTM_DI_GESTURE_GAMA(nn.Module):
    def __init__(self, cfar=True, track=True, ra_conv=True, attention=True, contrast=True, theta=0.001,
                 in_size=(32, 32), hidden_size=128,
                 out_size=7, cfar_expand_channels=8, cfar_in_channel=1,
                 track_channels=(4, 8, 16), ra_feat_size=32, track_out_size=64, spatial_channels=(8, 16, 32)):
        super(DRAI_2DCNNLSTM_DI_GESTURE_GAMA, self).__init__()
        self.sn = SpatialModel(cfar=cfar, cfar_expand_channels=cfar_expand_channels, ra_conv=ra_conv, in_size=in_size,
                               num_channels=spatial_channels, cfar_in_channel=cfar_in_channel,
                               out_size=hidden_size, ra_feat_size=ra_feat_size)
        self.need_track = track
        self.theta = theta
        if track:
            self.tn = TRACK_NET(num_channels=track_channels, in_size=in_size, out_size=track_out_size)
            self.fc_2 = nn.Linear((track_out_size + hidden_size), out_size)
            self.template = nn.Parameter(torch.zeros(1, out_size, track_out_size + hidden_size), requires_grad=False)
            self.forget = nn.Parameter(torch.ones((track_out_size + hidden_size) * 2, track_out_size + hidden_size))
            self.fc_3 = nn.Linear(out_size * (track_out_size + hidden_size), out_size)
        else:
            self.fc_2 = nn.Linear(hidden_size, out_size)
            self.template = nn.Parameter(torch.zeros(1, out_size, hidden_size))
            self.forget = nn.Parameter(torch.ones(hidden_size * 2, hidden_size))
            self.fc_3 = nn.Linear(out_size * hidden_size, out_size)
        self.lstm = nn.LSTM(input_size=hidden_size // 4, hidden_size=hidden_size // 4, num_layers=1, batch_first=True)
        self.need_attention = attention
        if attention:
            self.multi_head_attention = MultiHeadAttention(query_size=hidden_size, key_size=hidden_size,
                                                           value_size=hidden_size, num_hidden=hidden_size,
                                                           num_heads=4, dropout=nn.Dropout(p=0.5))
        # self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        # self.ln1 = nn.LayerNorm(hidden_size)
        # self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x, tracks, data_length, **kwargs):
        bach_size = len(x)
        x = self.sn(x, data_length, **kwargs)

        x = self.dropout1(x)
        x = pack_padded_sequence(x, data_length, batch_first=True)
        x = split_by_heads(x, 4)
        output, (h_n, _) = self.lstm(x)
        final_state = h_n[-1]
        # final_state = self.ln1(final_state)
        # output = self.ln1(output)
        final_state = final_state.view(bach_size, 1, -1)
        final_state = recover(final_state, 4)
        # x = self.bn1(x)
        if self.need_attention:
            output, out_len = pad_packed_sequence(output, batch_first=True)
            output = recover(output, 4)
            x = self.multi_head_attention(final_state, output, output, out_len)
            x = final_state + self.dropout2(x)
        else:
            x = final_state
        x = x.view(bach_size, -1)
        x = self.bn2(x)
        if self.need_track:
            t = self.tn(tracks, **kwargs)
            t = t.view(bach_size, -1)
            x = torch.cat((x, t), dim=-1)
        weight = self.fc_2(x)
        weight = F.softmax(weight).view(bach_size, -1, 1)
        x2 = x.view(bach_size, 1, -1) * weight
        concat = torch.cat((x2, self.template.repeat(bach_size, 1, 1)), axis=-1)
        concat = concat.view(-1, concat.size(-1))
        f = F.sigmoid(concat @ self.forget).view(bach_size, -1, x2.size(-1))
        template = self.template + self.theta * torch.mean((f * x2), dim=0)[None, :, :]
        if self.training:
            self.template = nn.Parameter(template, requires_grad=False)
        template = template - torch.mean(template, dim=-1)[:, :, None]
        x = x[:, None, :]
        x = x - torch.mean(x, dim=-1)[:, :, None]
        x = torch.sum(x * template, dim=-1) / (torch.sqrt(torch.sum(x ** 2, dim=-1)) * torch.sqrt(
            torch.sum(template ** 2, dim=-1)))
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


class DRAI_1DCNNLSTM_DI_GESTURE(nn.Module):
    def __init__(self):
        super(DRAI_1DCNNLSTM_DI_GESTURE, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, 3)
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(8, 16, 3)
        self.bn2 = nn.BatchNorm1d(16)
        self.conv3 = nn.Conv1d(16, 32, 3)
        self.bn3 = nn.BatchNorm1d(32)
        self.maxpool = nn.MaxPool1d(2, ceil_mode=True)
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
        self.fc_2 = nn.Linear(3168, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc_3 = nn.Linear(128, 4)
        # self.flatten = nn.Flatten
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, data_length):
        x = x.view(-1, 1, 800)
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
        x = x.view(-1, 3168)
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





if __name__ == '__main__':
    net = DRAI_2DCNNLSTM_DI_GESTURE_DENOISE()
    ipt = torch.zeros((128, 128, 32, 32))
    net.forward(ipt, np.random.random(size=(1, 127, 128)))
