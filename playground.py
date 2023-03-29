import numpy
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from network import *
import numpy as np

# 生成示例数据
inputs = torch.rand((2, 2, 4))
valid_lens = torch.Tensor([2, 3])

print(masked_softmax(inputs, valid_lens))
