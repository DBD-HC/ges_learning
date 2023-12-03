import torch.nn as nn
import torch.nn.functional as F
import torch
from model.mask_batchnorm import *
from utils import *


class SE1dBlock(nn.Module):
    def __init__(self, in_channel, radio=1.0):
        super(SE1dBlock, self).__init__()
        hidden_size = math.ceil(in_channel * radio)
        self.fc1 = nn.Linear(in_channel, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, in_channel, bias=False)

    def forward(self, x):
        weight = torch.mean(x, dim=-1)
        weight = self.fc1(weight)
        weight = F.relu(weight)
        weight = self.fc2(weight)
        weight = torch.sigmoid(weight)

        weight = weight[:, :, None]
        weight = weight * x

        return x + weight


class ThrSEBlock(nn.Module):
    def __init__(self, in_channel, radio=1.0):
        super(ThrSEBlock, self).__init__()
        hidden_size = math.ceil(in_channel * radio)
        self.fc1 = nn.Linear(in_channel, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, in_channel, bias=False)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.bn1 = nn.BatchNorm1d(in_channel)


    def forward(self, x, mask=None):
        x = torch.abs(x)
        x = torch.mean(x, dim=(-2, -1))
        weight = self.fc1(x)
        weight = F.relu(weight)
        weight = self.bn(weight)
        weight = self.fc2(weight)
        #weight = self.bn1(weight)
        weight = torch.sigmoid(weight)
        weight = x * weight

        return weight


class ThrSEBlock1d(nn.Module):
    def __init__(self, in_channel, radio=1.0):
        super(ThrSEBlock1d, self).__init__()
        hidden_size = math.ceil(in_channel * radio)
        self.fc1 = nn.Linear(in_channel, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, in_channel, bias=False)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.bn1 = nn.BatchNorm1d(in_channel)
        self.ln = nn.LayerNorm(hidden_size)


    def forward(self, x, mask=None):
        x = torch.abs(x)
        # x = torch.mean(x, dim=(-2, -1))
        weight = self.fc1(x)
        weight = F.relu(weight)
        weight = self.ln(weight)
        weight = self.fc2(weight)
        #weight = self.bn1(weight)
        weight = torch.sigmoid(weight)
        weight = x * weight

        return weight