import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from model.attention import MultiHeadAttention

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2,self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        if self.downsample:
            x = self.downsample(x)

        return self.relu(out + x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, dropout=0.2, kernel_size=3):
        super(TCN, self).__init__()

        layers = []
        num_levels = len(num_channels)
        self.output_size = output_size

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     padding=(kernel_size - 1) * dilation_size,
                                     dilation=dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x, data_lens):
        x = torch.transpose(x, 1, 2)
        out = self.network(x)
        out = torch.transpose(out, 1, 2)
        out = self.fc(out)
        final_state = torch.zeros((len(data_lens), 1, self.output_size), device=x.device)
        ceil =  torch.zeros((len(data_lens), 1, self.output_size), device=x.device)
        for i,v in enumerate(data_lens):
            final_state[i] = out[i, v-1]
        return out, (final_state, ceil)


class SimpleTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, dropout=0.2):
        super(SimpleTemporalBlock, self).__init__()

        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.attention = MultiHeadAttention(query_size=out_channels, key_size=out_channels, value_size=out_channels, num_hidden=out_channels,
                                                       num_heads=4, dropout=nn.Dropout(p=0.2))
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)
        self.relu = nn.ReLU()

    def forward(self, x, valid_length):
        x = torch.transpose(x, 1, 2)
        out = self.net(x)
        out = torch.transpose(out, 1, 2)
        out = self.attention(out, out, out, valid_length)
        return self.relu(out)

class AttTCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super(AttTCN, self).__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     padding=(kernel_size - 1) * dilation_size,
                                     dilation=dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x, valid_length):
        out = self.network(x, valid_length)
        return out