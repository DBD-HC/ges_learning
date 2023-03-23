import torch
import numpy as np
from cplxmodule.nn import CplxConv1d, CplxLinear, CplxDropout
from cplxmodule.nn import CplxModReLU, CplxParameter, CplxModulus, CplxToCplx
from cplxmodule.nn.modules.casting import TensorToCplx
from cplxmodule.nn import RealToCplx, CplxToReal
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


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


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv3d(1, 4, kernel_size=3)
        self.conv2 = torch.nn.Conv3d(4, 8, kernel_size=3)
        self.conv3 = torch.nn.Conv3d(8, 16, kernel_size=3)
        self.conv4 = torch.nn.Conv3d(16, 32, kernel_size=3)
        self.maxpool = nn.MaxPool3d(2, ceil_mode=True)
        self.fc1 = nn.Linear(23200, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3 * 4)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)
        x = x.view(batchsize, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 3,
                                                                                                         3).repeat(
            batchsize, 1, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x.view(-1, 3, 4)
        x = torch.cat((x, iden), dim=1)
        return x


class DRAI_2DCNNLSTM_DI_GESTURE_DENOISE(nn.Module):
    def __init__(self):
        super(DRAI_2DCNNLSTM_DI_GESTURE_DENOISE, self).__init__()
        self.stn = STN3d()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.bn3 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)
        self.lstm = nn.LSTM(input_size=21632, hidden_size=128, num_layers=1, batch_first=True)

        # self.fc_2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(p=0.5)
        self.fc_3 = nn.Linear(128, 13)
        # self.flatten = nn.Flatten
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, data_length):
        x = x.view(-1, 1, 128, 32, 32)
        x = self.stn(x)
        x = x.view(-1, 1, 32, 32)
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
        x = x.view(len(data_length), -1, 21632)
        x = pack_padded_sequence(x, data_length, batch_first=True)
        output, (h_n, c_n) = self.lstm(x)
        # output, out_len = pad_packed_sequence(output, batch_first=True)
        x = h_n[-1]

        # x = self.fc_2(x)
        x = self.dropout(x)
        x = self.fc_3(x)
        x = self.softmax(x)

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
        self.lstm = nn.LSTM(input_size=21632, hidden_size=128, num_layers=1, batch_first=True)

        # self.fc_2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(p=0.5)
        self.fc_3 = nn.Linear(128, 13)
        # self.flatten = nn.Flatten
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, data_length):
        x = x.view(-1, 1, 32, 32)
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
        x = x.view(len(data_length), -1, 21632)
        x = pack_padded_sequence(x, data_length, batch_first=True)
        output, (h_n, c_n) = self.lstm(x)
        # output, out_len = pad_packed_sequence(output, batch_first=True)
        x = h_n[-1]

        # x = self.fc_2(x)
        x = self.dropout(x)
        x = self.fc_3(x)
        x = self.softmax(x)

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
    inputs = torch.zeros((128, 128, 32, 32))
    net.forward(inputs, np.random.random(size=(1, 127, 128)))
