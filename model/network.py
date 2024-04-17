from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from model.attention import *
from model.mobile_net import *
from model.squeeze_and_excitation import *


class Conv1dBnRelu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, group=1, padding=1, dilation=1, bias=False,
                 bn=True, active=True):
        super(Conv1dBnRelu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, dilation=dilation,
                      kernel_size=kernel_size, padding=padding, stride=stride, bias=bias, groups=group)
        )
        if bn:
            self.conv.append(nn.BatchNorm1d(out_channel))
        if active:
            self.conv.append(nn.ReLU())

    def forward(self, x):
        x = self.conv(x)
        return x


class Conv2dBnRelu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, group=1, padding=1, dilation=1, bias=False,
                 bn=True, active=True):
        super(Conv2dBnRelu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, dilation=dilation,
                      kernel_size=kernel_size, padding=padding, stride=stride, bias=bias, groups=group)
        )
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channel))
        if active:
            self.conv.append(nn.ReLU())

    def forward(self, x):
        x = self.conv(x)
        return x


class RIAIConv(nn.Module):
    def __init__(self, in_channel=2, num_channels=(8, 16, 32), diff=True, input_size=32, out_size=32, dim=-1):
        super(RIAIConv, self).__init__()

        self.conv = nn.Sequential(
            Conv1dBnRelu(in_channel*3, num_channels[0], stride=2, kernel_size=3, bn=False, active=True, bias=True,
                         padding=0),
            Conv1dBnRelu(num_channels[0], num_channels[1], stride=2, kernel_size=3, bn=True, active=True, bias=False,
                         padding=0),
            Conv1dBnRelu(num_channels[1], num_channels[2], stride=2, kernel_size=3, bn=True, active=True, bias=False,
                         padding=0),
        )
        self.dim = dim
        self.dp1 = nn.Dropout(0.2)
        self.need_diff = diff

        linear_in = get_after_conv_size(input_size, kernel_size=3, stride=2, padding=0, layer=3) * num_channels[-1]
        if diff:
            self.diff = DifferentialNetwork(in_size=linear_in, out_size=out_size)
        else:
            self.fc_1 = nn.Sequential(
                nn.Linear(linear_in, out_size, bias=True),
                nn.ReLU()
            )

    def forward(self, x, batch_size, padding_len, y= None):
        max_x = torch.max(x, dim=self.dim)[0]
        avg_x = torch.mean(x, dim=self.dim)
        std_x = torch.std(x, dim=self.dim)
        avg_max = torch.cat((max_x, avg_x, std_x), dim=1)
        score = self.conv(avg_max)
        score = score.view(len(x), -1)
        score = self.dp1(score)
        before_diff = None
        if self.need_diff:
            score, before_diff = self.diff(score, batch_size, padding_len, y)
        else:
            score = self.fc_1(score)
            score = score.view(batch_size, padding_len, -1)
        #score = self.dp2(score)
        return score, before_diff


class FrameModel(nn.Module):
    def __init__(self, conv1d_channels, conv2d_channels, multistream=True, diff=False, in_size=(32, 32),
                 conv1d_feat_size=32, conv2d_feat_size=64, in_channels=1, dropout=0.5):
        super(FrameModel, self).__init__()

        self.multistream = multistream
        self.dp = nn.Dropout(dropout)
        self.in_channels = in_channels

        if multistream:
            self.range_conv = RIAIConv(dim=-1, in_channel=in_channels, input_size=in_size[0],
                                       num_channels=conv1d_channels,
                                       diff=diff, out_size=conv1d_feat_size)
            self.angel_conv = RIAIConv(dim=-2, in_channel=in_channels, input_size=in_size[1],
                                       num_channels=conv1d_channels,
                                       diff=diff, out_size=conv1d_feat_size)
        else:
            linear_input = conv2d_channels[-1] * get_after_conv_size(size=in_size[0], kernel_size=3, layer=3, padding=0,
                                                                     reduction=2) ** 2
            self.spatial_conv = nn.Sequential(
                Conv2dBnRelu(in_channels, conv2d_channels[0], padding=0),
                nn.MaxPool2d(kernel_size=2, ceil_mode=True),
                Conv2dBnRelu(conv2d_channels[0], conv2d_channels[1], padding=0),
                nn.MaxPool2d(kernel_size=2, ceil_mode=True),
                Conv2dBnRelu(conv2d_channels[1], conv2d_channels[2], padding=0),
                nn.MaxPool2d(kernel_size=2, ceil_mode=True)
            )
            if diff:
                self.diff = DifferentialNetwork(in_size=linear_input, out_size=conv2d_feat_size)
            else:
                self.fc_1 = nn.Sequential(
                    nn.Linear(linear_input, conv2d_feat_size, bias=True),
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
        conv2d_feats = None
        if self.multistream:
            range_feats, range_before_diff = self.range_conv(rai, batch_size=batch_size, padding_len=padded_len)
            angel_feats, _ = self.angel_conv(rai, y=range_before_diff, batch_size=batch_size, padding_len=padded_len)
        else:
            rai = self.spatial_conv(rai)
            rai = rai.view(rai.size(0), -1)
            rai = self.dp(rai)
            if self.need_diff:
                conv2d_feats, _ = self.diff(rai, batch_size, padded_len)
            else:
                conv2d_feats = self.fc_1(rai)
                conv2d_feats = conv2d_feats.view(batch_size, padded_len, -1)

        return conv2d_feats, range_feats, angel_feats


class DifferentialNetwork(nn.Module):
    def __init__(self, in_size, out_size):
        super(DifferentialNetwork, self).__init__()
        self.linear_sp1 = nn.Sequential(
            nn.Linear(in_size, out_size, bias=False),
            nn.LayerNorm(out_size),
            nn.ReLU(),
            nn.Linear(out_size, out_size // 2, bias=True),
            #nn.LayerNorm(out_size//2),
            nn.Hardswish(),
            # nn.Dropout(0.2),
            nn.Linear(out_size // 2, out_size // 4, bias=True),
            nn.Hardswish(),
            # nn.BatchNorm1d(in_size//2),
        )
        self.linear_sp2 = nn.Sequential(
            nn.Linear(out_size // 4, out_size, bias=False),
            nn.LayerNorm(out_size),
            #nn.Dropout(0.5),
        )

    def forward(self, x, batch_size, padded_len, y=None):
        x = self.linear_sp1(x)
        before_diff = x
        if y is not None:
            x = before_diff * y
        x = x.view(batch_size, padded_len, -1)
        x = x[:, 1:] - x[:, :-1]
        x = x.view(-1, x.size(-1))
        x = self.linear_sp2(x)
        x = x.view(batch_size, -1, x.size(-1))
        return x, before_diff


def lstm_func(lstm_block, data, data_lens):
    if data_lens is not None:
        x = pack_padded_sequence(data, data_lens.cpu(), batch_first=True)
        output, (h_n, _) = lstm_block(x)
        output, out_len = pad_packed_sequence(output, batch_first=True)
    else:
        output, (h_n, _) = lstm_block(data)
    return output, h_n[-1]


class TemporalModel(nn.Module):
    def __init__(self, feat_size1=32, feat_size2=32, attention=True, diff=True, multistream=False, heads=2):
        super(TemporalModel, self).__init__()
        if not multistream:
            self.lstm_1 = nn.LSTM(input_size=feat_size1, hidden_size=feat_size1, num_layers=1, batch_first=True)
            hidden_size = feat_size1
        else:
            self.lstm_2 = nn.LSTM(input_size=feat_size2, hidden_size=feat_size2, num_layers=1,
                                  batch_first=True)
            self.lstm_3 = nn.LSTM(input_size=feat_size2, hidden_size=feat_size2, num_layers=1,
                                  batch_first=True)
            hidden_size = feat_size2 * 2

        self.need_attention = attention
        if attention:
            self.multi_head_attention = MultiHeadAttention(query_size=hidden_size, key_size=hidden_size,
                                                           value_size=hidden_size, num_hidden=hidden_size,
                                                           num_heads=heads, dropout=0.2, bias=False)

        self.bn_1 = nn.BatchNorm1d(hidden_size)
        self.need_diff = diff

    def forward(self, x1=None, x2=None, x3=None, data_lens=None):
        if self.need_diff and data_lens is not None:
            data_lens = data_lens - 1

        if x1 is not None:
            output, final_state = lstm_func(self.lstm_1, x1, data_lens)
        else:
            output_2, final_state_2 = lstm_func(self.lstm_2, x2, data_lens)
            output_3, final_state_3 = lstm_func(self.lstm_3, x3, data_lens)
            output = torch.cat((output_2, output_3), dim=-1)
            final_state = torch.cat((final_state_2, final_state_3), dim=-1)
            # final_state = self.funsion_fc(final_state)
        final_state = final_state[:, None, :]

        if self.need_attention:
            x1 = self.multi_head_attention(final_state, output, output, data_lens)
            x1 = final_state + x1
        else:
            x1 = final_state
        x1 = x1.view(len(x1), -1)
        x1 = self.bn_1(x1)
        x1 = F.hardswish(x1)

        return x1


class TrackConv(nn.Module):
    def __init__(self, num_channels, in_channels=3, in_size=(32, 32), out_size=64):
        super(TrackConv, self).__init__()
        linear_input = num_channels[-1] * get_after_conv_size(size=in_size[0], kernel_size=3, layer=3, reduction=2) \
                       * get_after_conv_size(size=in_size[1], kernel_size=3, layer=3, reduction=2)
        self.track_conv = nn.Sequential(
            Conv2dBnRelu(in_channels * 3, num_channels[0], kernel_size=(3, 3), stride=2, bias=False, bn=True, padding=0),
            Conv2dBnRelu(num_channels[0], num_channels[1], kernel_size=(3, 3), stride=2, bias=False, bn=True, padding=0),
            Conv2dBnRelu(num_channels[1], num_channels[2], kernel_size=(3, 3), stride=2, bias=False, bn=True, padding=0),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(linear_input, out_size, bias=False),
            nn.BatchNorm1d(out_size),
            nn.Hardswish()
        )

    def forward(self, track):
        x = self.track_conv(track)
        return x


class RAIRadarGestureClassifier(nn.Module):
    def __init__(self, multistream=True, diff=True, attention=True, heads=4, in_size=(32, 32, 32),
                 in_channel=1, conv2d_feat_size=64, out_size=7, track_channels=(4, 8, 16),
                 ra_feat_size=32, track_out_size=64, spatial_channels=(8, 16, 32), conv1d_channels=(8, 16, 32)):
        super(RAIRadarGestureClassifier, self).__init__()

        self.multistream = multistream
        self.ra_feat_size = ra_feat_size
        self.frame_model = FrameModel(multistream=multistream, in_size=in_size, conv2d_channels=spatial_channels,
                                      conv1d_channels=conv1d_channels, diff=diff,
                                      conv2d_feat_size=conv2d_feat_size, conv1d_feat_size=ra_feat_size,
                                      in_channels=in_channel)
        fc_feat_size = 0
        self.temporal_model = TemporalModel(feat_size1=conv2d_feat_size, attention=attention, multistream=multistream,
                                            heads=heads, diff=diff, feat_size2=ra_feat_size)
        if multistream:
            self.tn = TrackConv(in_channels=in_channel, num_channels=track_channels, in_size=in_size,
                                out_size=track_out_size)

            fc_feat_size = fc_feat_size + track_out_size + ra_feat_size * 2
        else:
            fc_feat_size = fc_feat_size + conv2d_feat_size

        self.classifier = nn.Linear(fc_feat_size, out_size)

    def forward(self, rai, data_length, track):
        bach_size = rai.size(0)
        padded_lens = rai.size(1)
        h = rai.size(-2)
        w = rai.size(-1)

        rai = rai.view(bach_size, padded_lens, -1, h, w)
        t = None
        if self.multistream:
            t = self.tn(track)
            t = t.view(bach_size, -1)
        x, r, a = self.frame_model(rai, data_length)
        x = self.temporal_model(x, r, a, data_lens=data_length)
        if t is not None:
            x = torch.cat((x, t), dim=-1)
        x = self.classifier(x)

        return x


if __name__ == '__main__':
    print('let do it')
