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
from network import masked_softmax


class DRAI_2DCNNLSTM_M_GESTURE(nn.Module):
    def __init__(self):
        super(DRAI_2DCNNLSTM_M_GESTURE, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc_2 = nn.Linear(8192, 128)
        self.fc_3 = nn.Linear(128, 6)

    def forward(self, x, data_length):
        x = x.view(-1, 3, 128, 128)
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
        x = x.view(-1, 8192)
        x = self.fc_2(x)
        x = self.dropout(x)
        x = x.view(len(data_length), -1, 128)
        x = pack_padded_sequence(x, data_length, batch_first=True)
        output, (h_n, c_n) = self.lstm(x)
        x = h_n[-1]
        x = self.fc_3(x)

        return x


class DRAI_2DCNNLSTM_air_writing(nn.Module):
    def __init__(self):
        super(DRAI_2DCNNLSTM_air_writing, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc_2 = nn.Linear(1568, 128)
        self.fc_3 = nn.Linear(128, 10)

    def forward(self, x, data_length):
        x = x.view(-1, 1, 50, 50)
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
        x = x.view(-1, 1568)
        x = self.fc_2(x)
        x = self.dropout(x)
        x = x.view(len(data_length), -1, 128)
        x = pack_padded_sequence(x, data_length, batch_first=True)
        output, (h_n, c_n) = self.lstm(x)
        x = h_n[-1]
        x = self.fc_3(x)

        return x

class DRAI_2DCNNLSTM_air_writing_2(nn.Module):
    def __init__(self):
        super(DRAI_2DCNNLSTM_air_writing_2, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc_2 = nn.Linear(1568, 128)
        self.fc_3 = nn.Linear(128, 10)

    def forward(self, x, data_length):
        x = x.view(-1, 1, 50, 50)
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
        x = x.view(-1, 1568)
        x = self.fc_2(x)
        x = self.dropout(x)
        x = self.fc_3(x)

        return x