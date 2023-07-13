import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch.nn as nn
import math
import torch.nn.functional as F
from utils import *
from model.tcn import *



class DRAI_2DCNNLSTM_DI_GESTURE_LITE(nn.Module):
    def __init__(self, in_size=(32,32), num_channels=(8, 16, 32), hidden_size=64, conv_embedding=32, labels=7):
        super(DRAI_2DCNNLSTM_DI_GESTURE_LITE, self).__init__()
        self.conv1 = nn.Conv2d(1, num_channels[0], 3)
        self.bn1 = nn.BatchNorm2d(num_channels[0])
        self.conv2 = nn.Conv2d(num_channels[0], num_channels[1], 3)
        self.bn2 = nn.BatchNorm2d(num_channels[1])
        self.conv3 = nn.Conv2d(num_channels[1], num_channels[2], 3)
        self.bn3 = nn.BatchNorm2d(num_channels[2])
        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)
        self.lstm = nn.LSTM(input_size=conv_embedding, hidden_size=hidden_size, num_layers=1, batch_first=True)
        linear_input = num_channels[-1] * get_after_conv_size(size=in_size[0], kernel_size=3, layer=3,
                                                              reduction=2) \
                       * get_after_conv_size(size=in_size[1], kernel_size=3, layer=3, reduction=2)
        self.fc_1 = nn.Linear(linear_input, conv_embedding)
        self.dropout = nn.Dropout(p=0.5)
        self.fc_2 = nn.Linear(hidden_size, labels)

    def forward(self, x, data_length):
        batch_size = x.size(0)
        data_len = x.size(1)
        h = x.size(2)
        w = x.size(3)
        x = x.view(-1, 1, h, w)
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
        x = x.view(batch_size*data_len, -1)
        x = self.fc_1(x)
        x = self.dropout(x)
        x = x.view(batch_size, data_len, -1)
        x = pack_padded_sequence(x, data_length, batch_first=True)
        output, (h_n, c_n) = self.lstm(x)
        x = h_n[-1]
        x = self.fc_1(x)
        return x