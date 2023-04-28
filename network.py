import torch
import numpy as np
from cplxmodule.nn import CplxConv1d, CplxLinear, CplxDropout
from cplxmodule.nn import CplxModReLU, CplxParameter, CplxModulus, CplxToCplx
from cplxmodule.nn.modules.casting import TensorToCplx
from cplxmodule.nn import RealToCplx, CplxToReal
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch.nn as nn
import math
import torch.nn.functional as F
import visdom


class Range_Fourier_Net(nn.Module):
    def __init__(self):
        super(Range_Fourier_Net, self).__init__()
        self.range_nn = CplxLinear(256, 256, bias=False)
        range_weights = np.zeros((256, 256), dtype=np.complex64)
        for j in range(0, 256):
            for h in range(0, 256):
                range_weights[h][j] = np.exp(-1j * 2 * np.pi * (j * h / 256))
        range_weights = TensorToCplx()(torch.view_as_real(torch.from_numpy(range_weights)))
        self.range_nn.weight = CplxParameter(range_weights)

    def forward(self, x):
        x = self.range_nn(x)
        return x


class Range_Fourier_Net_Air_Writing(nn.Module):
    def __init__(self):
        super(Range_Fourier_Net_Air_Writing, self).__init__()
        self.range_nn = CplxLinear(100, 100, bias=False)
        range_weights = np.zeros((100, 100), dtype=np.complex64)
        for j in range(0, 100):
            for h in range(0, 100):
                range_weights[h][j] = np.exp(-1j * 2 * np.pi * (j * h / 100))
        range_weights = TensorToCplx()(torch.view_as_real(torch.from_numpy(range_weights)))
        self.range_nn.weight = CplxParameter(range_weights)

    def forward(self, x):
        x = self.range_nn(x)
        return x


class Doppler_Fourier_Net(nn.Module):
    def __init__(self):
        super(Doppler_Fourier_Net, self).__init__()
        self.doppler_nn = CplxLinear(128, 128, bias=False)
        doppler_weights = np.zeros((128, 128), dtype=np.complex64)
        for j in range(0, 128):
            for h in range(0, 128):
                if 63 <= h <= 65:
                    continue
                hh = h + 64
                if hh >= 128:
                    hh = hh - 128
                doppler_weights[h][j] = np.exp(-1j * 2 * np.pi * (j * hh / 128))
        doppler_weights = TensorToCplx()(torch.view_as_real(torch.from_numpy(doppler_weights)))
        self.doppler_nn.weight = CplxParameter(doppler_weights)

    def forward(self, x):
        x = self.doppler_nn(x)
        return x


class DenoisingCNN(nn.Module):
    def __init__(self):
        super(DenoisingCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv3d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv3d(32, 1, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class Range_Fourier_Net_Small(nn.Module):
    def __init__(self):
        super(Range_Fourier_Net, self).__init__()
        self.range_nn = CplxLinear(128, 128, bias=False)
        range_weights = np.zeros((128, 128), dtype=np.complex64)
        for j in range(0, 128):
            for h in range(0, 128):
                range_weights[h][j] = np.exp(-1j * 2 * np.pi * (j * h / 128))
        range_weights = TensorToCplx()(torch.view_as_real(torch.from_numpy(range_weights)))
        self.range_nn.weight = CplxParameter(range_weights)

    def forward(self, x):
        x = self.range_nn(x)
        return x


class Doppler_Fourier_Net_Small(nn.Module):
    def __init__(self):
        super(Doppler_Fourier_Net, self).__init__()
        self.doppler_nn = CplxLinear(64, 64, bias=False)
        doppler_weights = np.zeros((64, 64), dtype=np.complex64)
        for j in range(0, 64):
            for h in range(0, 64):
                hh = h + 32
                if hh >= 64:
                    hh = hh - 64
                doppler_weights[h][j] = np.exp(-1j * 2 * np.pi * (j * hh / 64))
        doppler_weights = TensorToCplx()(torch.view_as_real(torch.from_numpy(doppler_weights)))
        self.doppler_nn.weight = CplxParameter(doppler_weights)

    def forward(self, x):
        x = self.doppler_nn(x)
        return x


class AOA_Fourier_Net(nn.Module):
    def __init__(self):
        super(AOA_Fourier_Net, self).__init__()
        self.aoa_nn = CplxLinear(8, 64, bias=False)
        aoa_weights = np.zeros((64, 8), dtype=np.complex64)
        for j in range(8):
            for h in range(64):
                hh = h + 32
                if hh >= 64:
                    hh = hh - 64
                h_idx = h
                aoa_weights[h_idx][j] = np.exp(-1j * 2 * np.pi * (j * hh / 64))

        aoa_weights = TensorToCplx()(torch.view_as_real(torch.from_numpy(aoa_weights)))
        self.aoa_nn.weight = CplxParameter(aoa_weights)

    def forward(self, x):
        x = self.aoa_nn(x)
        return x


class MaskedBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(MaskedBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input, padded_length=None, valid_length=None):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mask = torch.arange(padded_length, dtype=torch.float32, device=input.device)
            valid_length = valid_length.to(input.device)
            mask = mask[None, :] < valid_length[:, None]
            mask = mask.reshape(-1)

            mean = input[mask].mean([0, 2, 3])
            # use biased var in train
            var = input[mask].var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input


def masked_softmax(inputs, valid_length):
    if valid_length is None:
        return nn.functional.softmax(inputs, dim=-1)
    else:
        shape = inputs.shape
        if valid_length.dim() == 1:
            valid_length = torch.repeat_interleave(valid_length, shape[-2], dim=0)
        else:
            valid_length = valid_length.reshape(-1)
        mask = torch.arange((shape[-1]), dtype=torch.float32, device=inputs.device)
        valid_length = valid_length.to(inputs.device)
        mask = mask[None, :] >= valid_length[:, None]
        inputs = inputs.reshape(-1, shape[-1])
        inputs[mask] = -1e9
        return nn.functional.softmax(inputs.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):
    def __init__(self, dropout):
        super(DotProductAttention, self).__init__()
        self.dropout = dropout

    def forward(self, queries, keys, values, valid_length):
        d = queries.shape[-1]
        scores = torch.bmm(queries, torch.transpose(keys, 1, 2)) / (d ** 0.5)
        self.attention_weights = masked_softmax(scores, valid_length)
        return torch.bmm(self.dropout(self.attention_weights), values)


def split_by_heads(x, num_heads):
    x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)
    x = torch.transpose(x, 1, 2)
    return x.reshape(-1, x.shape[2], x.shape[3])


class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hidden, num_heads, dropout, bias=False):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.w_q = nn.Linear(query_size, num_hidden, bias=bias)
        self.w_k = nn.Linear(key_size, num_hidden, bias=bias)
        self.w_v = nn.Linear(value_size, num_hidden, bias=bias)
        self.w_o = nn.Linear(num_hidden, num_hidden, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        queries = split_by_heads(self.w_q(queries), self.num_heads)
        keys = split_by_heads(self.w_k(keys), self.num_heads)
        values = split_by_heads(self.w_v(values), self.num_heads)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, self.num_heads, dim=0)
        outputs = self.attention(queries, keys, values, valid_lens)
        outputs = outputs.reshape(-1, self.num_heads, outputs.shape[1], outputs.shape[2])
        outputs = torch.transpose(outputs, 1, 2)
        return self.w_o(outputs.reshape(outputs.shape[0], outputs.shape[1], -1))


def init_weight(window_size, sigma=6):
    if isinstance(window_size, tuple):
        half_x = window_size[0] // 2
        half_y = window_size[1] // 2
    else:
        half_x = window_size // 2
        half_y = half_x
    x = np.array([i if i <= half_x else window_size - i - 1 for i in range(window_size)])
    y = np.array([i if i <= half_y else window_size - i - 1 for i in range(window_size)])
    x, y = np.meshgrid(x, y)
    gaussian_kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    # 对高斯核进行归一化
    gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)

    return gaussian_kernel


class CFAR(nn.Module):
    def __init__(self, window_size):
        super(CFAR, self).__init__()
        if isinstance(window_size, tuple):
            padding = ((window_size[1] - 1) // 2, (window_size[2] - 1) // 2)
        else:
            padding = (window_size - 1) // 2

        # conv_weight = torch.from_numpy(init_weight(window_size, 6)).unsqueeze(0).unsqueeze(0)
        self.conv1 = nn.Conv2d(1, 8, padding=padding, padding_mode='circular', kernel_size=window_size)
        self.conv2 = nn.Conv2d(8, 8, padding=padding, padding_mode='circular', kernel_size=window_size)
        self.conv3 = nn.Conv2d(8, 1, padding=padding, padding_mode='circular', kernel_size=window_size)
        # self.conv1.weight = torch.nn.Parameter(conv_weight.float())
        self.visdom = visdom.Visdom(env='CFAR', port=6006)
        self.bn1 = MaskedBatchNorm2d(1)
        self.av_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.relu = torch.nn.LeakyReLU()

    def forward(self, x, padded_len, valid_len, **kwargs):
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
        thr = self.bn1(thr, padded_len, valid_len)
        thr = self.relu(thr)

        if not self.training:
            indexes = kwargs['indexes']
            ids = str(kwargs['epoch'] % 10)
            if 1415 in indexes:
                index = indexes.index(1415)
                x_t = x.view(len(valid_len), -1, 32, 32)
                thr_t = thr.view(len(valid_len), -1, 32, 32)
                self.visdom.heatmap(x_t[index][0], win=ids + '_origin',
                                    opts=dict(title='origin' + str(kwargs['epoch'])))
                self.visdom.heatmap(thr_t[index][0], win=ids + '_cfar_output',
                                    opts=dict(title='cfar output' + str(kwargs['epoch'])))
        return thr


# 定义位置编码器
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        # 创建一个含所有位置的位置编码矩阵
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # 将位置编码矩阵注册为模型的可学习参数
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将位置编码矩阵添加到词嵌入上
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x


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


class TRACK_NET(nn.Module):
    def __init__(self):
        super(TRACK_NET, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3, 3), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        # self.avg_pool = nn.AvgPool2d(2, ceil_mode=True)
        # self.avg_pool_2 = nn.AvgPool2d(3, stride=1, ceil_mode=True)
        self.dp = nn.Dropout(p=0.5)
        self.max_pool = nn.MaxPool2d(2, ceil_mode=True)
        self.visdom = visdom.Visdom(env='TRACK', port=6006)
        self.fc_1 = nn.Linear(2048, 128)
        self.bn4 = nn.BatchNorm1d(128)

    def forward(self, x, **kwargs):
        o = x.view(-1, 1, 32, 32)
        x = self.dp(o)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x1 = self.max_pool(x)
        x = self.conv2(x1)
        x = self.bn2(x)
        x = F.relu(x)
        x2 = self.max_pool(x)
        x = self.conv3(x2)
        x = self.bn3(x)
        x3 = F.relu(x)
        # x3 = self.avg_pool(x)
        # x3 = self.max_pool(x3)
        x = x3.view(x3.size(0), -1)
        x = self.fc_1(x)
        x = self.bn4(x)
        x = F.relu(x)

        if not self.training:
            indexes = kwargs['indexes']
            ids = str(kwargs['epoch'] % 5)
            if 1415 in indexes:
                index = indexes.index(1415)
                self.visdom.heatmap(o[index][0], win=ids + '_origin',
                                    opts=dict(title='origin' + str(kwargs['epoch'])))
                self.visdom.heatmap(x1[index][0], win=ids + '_x1',
                                    opts=dict(title='track output x1' + str(kwargs['epoch'])))
                self.visdom.heatmap(x2[index][0], win=ids + '_x2',
                                    opts=dict(title='track output x2' + str(kwargs['epoch'])))
                self.visdom.heatmap(x3[index][0], win=ids + '_x3',
                                    opts=dict(title='track output x3' + str(kwargs['epoch'])))

        return x


class DRAI_2DCNNLSTM_DI_GESTURE_BETA(nn.Module):
    def __init__(self, cfar=True, track=True):
        super(DRAI_2DCNNLSTM_DI_GESTURE_BETA, self).__init__()
        self.sn = SIGNAL_NET_BETA(cfar)
        self.track = track
        if track:
            self.tn = TRACK_NET()
            # self.fc_3 = nn.Linear(16384, 512)
            self.fc_2 = nn.Linear(384, 7)
        else:
            # self.fc_3 = nn.Linear(16384, 512)
            self.fc_2 = nn.Linear(256, 7)
        self.fc_4 = nn.Linear(1024, 256)
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=1, batch_first=True)
        self.multi_head_attention = MultiHeadAttention(query_size=256, key_size=256, value_size=256, num_hidden=256,
                                                       num_heads=8, dropout=nn.Dropout(p=0.5))
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.5)

        # self.bn2 = nn.BatchNorm1d(128)

        # self.fc_1 = nn.Linear(128, 7)
        # self.flatten = nn.Flatten
        # self.softmax = nn.Softmax(dim=1)

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
        # x = self.dropout(x)
        # x = self.fc_3(x)
        x = self.bn1(x)
        x = F.relu(x)

        if self.track:
            t = self.tn(tracks, **kwargs)
            t = t.view(bach_size, -1)
            # x = self.bn1(x)
            x = torch.cat((x, t), dim=1)

        x = self.dropout(x)
        x = self.fc_2(x)
        # x = F.relu(x)
        # x = self.fc_1(x)
        # x = self.softmax(x)

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
        self.fc_2 = nn.Linear(21632, 128)
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
        x = x.view(-1, 21632)
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


class RT_2DCNN(nn.Module):
    def __init__(self):
        super(RT_2DCNN, self).__init__()
        self.range_net = Range_Fourier_Net()
        self.cplx_transpose = CplxToCplx[torch.transpose]
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, 3)
        self.bn3 = nn.BatchNorm2d(16)
        self.maxpool = nn.MaxPool2d((1, 2), ceil_mode=True)

        self.fc_1 = nn.Linear(1984, 512)
        self.fc_2 = nn.Linear(512, 128)
        self.fc_3 = nn.Linear(128, 12)

    def forward(self, x):
        # input shape is batchsize, 10, 128, 12, 256
        x = x[:, :, 0, 0, :]
        x = x.contiguous()
        x = self.range_net(x)
        x = CplxModulus()(x)
        x = x.view(-1, 1, 10, 256)
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
        x = x.view(-1, 1984)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = F.relu(x)
        x = self.fc_3(x)
        return x


class DT_2DCNN(nn.Module):
    def __init__(self):
        super(DT_2DCNN, self).__init__()
        self.range_net = Range_Fourier_Net()
        self.doppler_net = Doppler_Fourier_Net()
        self.cplx_transpose = CplxToCplx[torch.transpose]
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, 3)
        self.bn3 = nn.BatchNorm2d(16)
        self.maxpool = nn.MaxPool2d((1, 2), ceil_mode=True)

        self.fc_1 = nn.Linear(960, 512)
        self.fc_2 = nn.Linear(512, 128)
        self.fc_3 = nn.Linear(128, 12)

    def forward(self, x):
        # input shape is batchsize, 10, 128, 12, 256
        x = x[:, :, :, 0, :]
        x = x.contiguous()
        x = self.range_net(x)
        x = x.view(-1, 128, 256)
        x = self.cplx_transpose(1, 2)(x)
        x = self.doppler_net(x)
        x = CplxModulus()(x)
        x = torch.sum(x, dim=1)
        x = x.view(-1, 1, 10, 128)
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
        x = x.view(-1, 960)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = F.relu(x)
        x = self.fc_3(x)
        return x


class AT_2DCNN(nn.Module):
    def __init__(self):
        super(AT_2DCNN, self).__init__()
        self.range_net = Range_Fourier_Net()
        self.aoa_net = AOA_Fourier_Net()
        self.cplx_transpose = CplxToCplx[torch.transpose]
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, 3)
        self.bn3 = nn.BatchNorm2d(16)
        self.maxpool = nn.MaxPool2d((1, 2), ceil_mode=True)

        self.fc_1 = nn.Linear(448, 512)
        self.fc_2 = nn.Linear(512, 128)
        self.fc_3 = nn.Linear(128, 12)

    def forward(self, x):
        # input shape is batchsize, 10, 128, 12, 256
        x = x[:, :, 0, 0:8, :]
        x = x.contiguous()
        x = self.range_net(x)
        x = x.view(-1, 8, 256)
        x = self.cplx_transpose(1, 2)(x)
        x = self.aoa_net(x)
        x = CplxModulus()(x)
        x = torch.sum(x, dim=1)
        x = x.view(-1, 1, 10, 64)
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
        x = x.view(-1, 448)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = F.relu(x)
        x = self.fc_3(x)
        return x


class RDT_2DCNNLSTM_Air_Writing(nn.Module):
    def __init__(self):
        super(RDT_2DCNNLSTM_Air_Writing, self).__init__()
        self.range_net = Range_Fourier_Net_Air_Writing()
        self.doppler_net = Doppler_Fourier_Net()
        self.cplx_transpose = CplxToCplx[torch.transpose]
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, 3)
        self.bn3 = nn.BatchNorm2d(16)
        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)
        self.lstm = nn.LSTM(input_size=2640, hidden_size=128, num_layers=1, batch_first=True)

        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x[:, :, :, 0, :]
        x = x.contiguous()
        x = self.range_net(x)
        x = x.view(-1, 128, 100)
        x = self.cplx_transpose(1, 2)(x)
        x = self.doppler_net(x)
        x = CplxModulus()(x)
        x = x.view(-1, 1, 100, 128)
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
        x = x.view(-1, 100, 2640)
        output, (h_n, c_n) = self.lstm(x)
        x = output[:, -1, :]
        x = self.fc_2(x)
        x = F.relu(x)
        x = self.fc_3(x)
        return x


class RDT_2DCNNLSTM(nn.Module):
    def __init__(self):
        super(RDT_2DCNNLSTM, self).__init__()
        self.range_net = Range_Fourier_Net()
        self.doppler_net = Doppler_Fourier_Net()
        self.cplx_transpose = CplxToCplx[torch.transpose]
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, 3)
        self.bn3 = nn.BatchNorm2d(16)
        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)
        self.lstm = nn.LSTM(input_size=7440, hidden_size=512, num_layers=1, batch_first=True)

        self.fc_2 = nn.Linear(512, 128)
        self.fc_3 = nn.Linear(128, 12)

    def forward(self, x):
        x = x[:, :, :, 0, :]
        x = x.contiguous()
        x = self.range_net(x)
        x = x.view(-1, 128, 256)
        x = self.cplx_transpose(1, 2)(x)
        x = self.doppler_net(x)
        x = CplxModulus()(x)
        x = x.view(-1, 1, 256, 128)
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
        x = x.view(-1, 10, 7440)
        output, (h_n, c_n) = self.lstm(x)
        x = output[:, -1, :]
        x = self.fc_2(x)
        x = F.relu(x)
        x = self.fc_3(x)
        return x


class RDT_3DCNN_air_writing(nn.Module):
    def __init__(self):
        super(RDT_3DCNN_air_writing, self).__init__()
        self.range_net = Range_Fourier_Net_Air_Writing()
        self.doppler_net = Doppler_Fourier_Net()
        self.cplx_transpose = CplxToCplx[torch.transpose]
        self.denoise = DenoisingCNN()
        # self.stn = STN3d()
        self.conv1 = nn.Conv3d(1, 4, 3)
        self.bn1 = nn.BatchNorm3d(4)
        self.conv2 = nn.Conv3d(4, 8, 3)
        self.bn2 = nn.BatchNorm3d(8)
        self.conv3 = nn.Conv3d(8, 16, 3)
        self.bn3 = nn.BatchNorm3d(16)
        self.maxpool = nn.MaxPool3d((1, 2, 2), ceil_mode=True)
        self.fc_1 = nn.Linear(248160, 256)
        self.fc_2 = nn.Linear(256, 64)
        self.fc_3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x[:, :, :, 0, :]
        x = x.contiguous()
        x = self.range_net(x)
        x = x.view(-1, 128, 100)
        x = self.cplx_transpose(1, 2)(x)
        x = self.doppler_net(x)
        x = CplxModulus()(x)
        x = x.view(-1, 1, 10, 100, 128)
        # x = self.denoise(x)
        # x = self.stn(x)
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
        x = x.view(-1, 248160)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = F.relu(x)
        x = self.fc_3(x)
        return x


class RDT_3DCNN(nn.Module):
    def __init__(self):
        super(RDT_3DCNN, self).__init__()
        self.range_net = Range_Fourier_Net()
        self.doppler_net = Doppler_Fourier_Net()
        self.cplx_transpose = CplxToCplx[torch.transpose]
        self.conv1 = nn.Conv3d(1, 4, 3)
        self.bn1 = nn.BatchNorm3d(4)
        self.conv2 = nn.Conv3d(4, 8, 3)
        self.bn2 = nn.BatchNorm3d(8)
        self.conv3 = nn.Conv3d(8, 16, 3)
        self.bn3 = nn.BatchNorm3d(16)
        self.maxpool = nn.MaxPool3d((1, 2, 2), ceil_mode=True)
        self.fc_1 = nn.Linear(29760, 512)
        self.fc_2 = nn.Linear(512, 128)
        self.fc_3 = nn.Linear(128, 12)

    def forward(self, x):
        x = x[:, :, :, 0, :]
        x = x.contiguous()
        x = self.range_net(x)
        x = x.view(-1, 128, 256)
        x = self.cplx_transpose(1, 2)(x)
        x = self.doppler_net(x)
        x = CplxModulus()(x)
        x = x.view(-1, 1, 10, 256, 128)
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
        x = x.view(-1, 29760)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = F.relu(x)
        x = self.fc_3(x)
        return x


class RAT_2DCNNLSTM(nn.Module):
    def __init__(self):
        super(RAT_2DCNNLSTM, self).__init__()
        self.range_net = Range_Fourier_Net()
        self.aoa_net = AOA_Fourier_Net()
        self.cplx_transpose = CplxToCplx[torch.transpose]
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, 3)
        self.bn3 = nn.BatchNorm2d(16)
        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)
        self.lstm = nn.LSTM(3472, 512, 1, batch_first=True)
        self.fc_2 = nn.Linear(512, 128)
        self.fc_3 = nn.Linear(128, 12)

    def forward(self, x):
        x = x[:, :, 0, 0:8, :]
        x = x.contiguous()
        x = self.range_net(x)
        x = x.view(-1, 8, 256)
        x = self.cplx_transpose(1, 2)(x)
        x = self.aoa_net(x)
        x = CplxModulus()(x)
        x = x.view(-1, 1, 256, 64)
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
        x = x.view(-1, 10, 3472)
        output, (h_n, c_n) = self.lstm(x)
        x = output[:, -1, :]
        x = self.fc_2(x)
        x = F.relu(x)
        x = self.fc_3(x)
        return x


class RAT_3DCNN(nn.Module):
    def __init__(self):
        super(RAT_3DCNN, self).__init__()
        self.range_net = Range_Fourier_Net()
        self.aoa_net = AOA_Fourier_Net()
        self.cplx_transpose = CplxToCplx[torch.transpose]
        self.conv1 = nn.Conv3d(1, 4, 3)
        self.bn1 = nn.BatchNorm3d(4)
        self.conv2 = nn.Conv3d(4, 8, 3)
        self.bn2 = nn.BatchNorm3d(8)
        self.conv3 = nn.Conv3d(8, 16, 3)
        self.bn3 = nn.BatchNorm3d(16)
        self.maxpool = nn.MaxPool3d((1, 2, 2), ceil_mode=True)
        self.fc_1 = nn.Linear(13888, 512)
        self.fc_2 = nn.Linear(512, 128)
        self.fc_3 = nn.Linear(128, 12)

    def forward(self, x):
        x = x[:, :, 0, 0:8, :]
        x = x.contiguous()
        x = self.range_net(x)
        x = x.view(-1, 8, 256)
        x = self.cplx_transpose(1, 2)(x)
        x = self.aoa_net(x)
        x = CplxModulus()(x)
        x = x.view(-1, 1, 10, 256, 64)
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
        x = x.view(-1, 13888)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = F.relu(x)
        x = self.fc_3(x)
        return x


class DAT_2DCNNLSTM(nn.Module):
    def __init__(self):
        super(DAT_2DCNNLSTM, self).__init__()
        self.range_net = Range_Fourier_Net_Small()
        self.doppler_net = Doppler_Fourier_Net_Small()
        self.aoa_net = AOA_Fourier_Net()
        self.cplx_transpose = CplxToCplx[torch.transpose]
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, 3)
        self.bn3 = nn.BatchNorm2d(16)
        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)
        self.lstm = nn.LSTM(784, 512, 1, batch_first=True)
        self.fc_2 = nn.Linear(512, 128)
        self.fc_3 = nn.Linear(128, 12)

    def forward(self, x):
        x = x[:, :, :, 0:8, :]
        x = x.contiguous()
        x = self.range_net(x)
        x = x.view(-1, 64, 8, 128)
        x = self.cplx_transpose(1, 3)(x)
        x = self.doppler_net(x)
        x = self.cplx_transpose(2, 3)(x)
        x = self.aoa_net(x)
        x = CplxModulus()(x)
        x = torch.sum(x, dim=1)
        x = x.view(-1, 1, 64, 64)
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
        x = x.view(-1, 10, 784)
        output, (h_n, c_n) = self.lstm(x)
        x = output[:, -1, :]
        x = self.fc_2(x)
        x = F.relu(x)
        x = self.fc_3(x)
        return x


class DAT_3DCNN(nn.Module):
    def __init__(self):
        super(DAT_3DCNN, self).__init__()
        self.range_net = Range_Fourier_Net_Small()
        self.doppler_net = Doppler_Fourier_Net_Small()
        self.aoa_net = AOA_Fourier_Net()
        self.cplx_transpose = CplxToCplx[torch.transpose]
        self.conv1 = nn.Conv3d(1, 4, 3)
        self.bn1 = nn.BatchNorm3d(4)
        self.conv2 = nn.Conv3d(4, 8, 3)
        self.bn2 = nn.BatchNorm3d(8)
        self.conv3 = nn.Conv3d(8, 16, 3)
        self.bn3 = nn.BatchNorm3d(16)
        self.maxpool = nn.MaxPool3d((1, 2, 2), ceil_mode=True)
        self.fc_1 = nn.Linear(3136, 512)
        self.fc_2 = nn.Linear(512, 128)
        self.fc_3 = nn.Linear(128, 12)

    def forward(self, x):
        x = x[:, :, :, 0:8, :]
        x = x.contiguous()
        x = self.range_net(x)
        x = x.view(-1, 64, 8, 128)
        x = self.cplx_transpose(1, 3)(x)
        x = self.doppler_net(x)
        x = self.cplx_transpose(2, 3)(x)
        x = self.aoa_net(x)
        x = CplxModulus()(x)
        x = torch.sum(x, dim=1)
        x = x.view(-1, 1, 10, 64, 64)
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
        x = x.view(-1, 3136)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = F.relu(x)
        x = self.fc_3(x)
        return x


class RDAT_3DCNNLSTM(nn.Module):
    def __init__(self):
        super(RDAT_3DCNNLSTM, self).__init__()
        self.range_net = Range_Fourier_Net_Small()
        self.doppler_net = Doppler_Fourier_Net_Small()
        self.aoa_net = AOA_Fourier_Net()
        self.cplx_transpose = CplxToCplx[torch.transpose]
        self.conv1 = nn.Conv3d(1, 4, 3)
        self.bn1 = nn.BatchNorm3d(4)
        self.conv2 = nn.Conv3d(4, 8, 3)
        self.bn2 = nn.BatchNorm3d(8)
        self.conv3 = nn.Conv3d(8, 16, 3)
        self.bn3 = nn.BatchNorm3d(16)
        self.maxpool = nn.MaxPool3d(2, ceil_mode=True)
        self.lstm = nn.LSTM(11760, 512, 1, batch_first=True)
        self.fc_2 = nn.Linear(512, 128)
        self.fc_3 = nn.Linear(128, 12)

    def forward(self, x):
        x = x[:, :, :, 0:8, :]
        x = x.contiguous()
        x = self.range_net(x)
        x = x.view(-1, 64, 8, 128)
        x = self.cplx_transpose(1, 3)(x)
        x = self.doppler_net(x)
        x = self.cplx_transpose(2, 3)(x)
        x = self.aoa_net(x)
        x = CplxModulus()(x)
        x = x.view(-1, 1, 128, 64, 64)
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
        x = x.view(-1, 10, 11760)
        output, (h_n, c_n) = self.lstm(x)
        x = output[:, -1, :]
        x = self.fc_2(x)
        x = F.relu(x)
        x = self.fc_3(x)
        return x


if __name__ == '__main__':
    net = DRAI_2DCNNLSTM_DI_GESTURE_DENOISE()
    ipt = torch.zeros((128, 128, 32, 32))
    net.forward(ipt, np.random.random(size=(1, 127, 128)))
