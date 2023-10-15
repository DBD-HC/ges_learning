import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np
from model.mobile_net import Conv1dBnRelu

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

def masked_softmax(inputs, valid_length):
    if valid_length is None:
        return F.softmax(inputs, dim=-1)
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
        return F.softmax(inputs.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):
    def __init__(self, dropout):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_length):
        d = queries.shape[-1]
        scores = torch.bmm(queries, torch.transpose(keys, 1, 2)) / (d ** 0.5)
        attention_weights = masked_softmax(scores, valid_length)
        return torch.bmm(self.dropout(attention_weights), values)


def split_by_heads(x, num_heads):
    x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)
    x = torch.transpose(x, 1, 2)
    return x.reshape(-1, x.shape[2], x.shape[3])

def recover(x, num_heads):
    outputs = x.reshape(-1, num_heads, x.shape[1], x.shape[2])
    outputs = torch.transpose(outputs, 1, 2)
    return outputs.reshape(outputs.shape[0], outputs.shape[1], -1)


class LiteMultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hidden, num_heads, dropout, bias=False):
        super(LiteMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.w_q = nn.Linear(query_size, num_hidden, bias=bias)
        self.w_k = nn.Linear(query_size, num_hidden, bias=bias)
        self.w_o = nn.Linear(num_hidden, num_hidden, bias=bias)
        nn.init.normal_(self.w_q.weight, mean=0, std=np.sqrt(2.0 / (query_size + key_size)))
        nn.init.normal_(self.w_k.weight, mean=0, std=np.sqrt(2.0 / (query_size + key_size)))

    def forward(self, queries, keys, values, valid_lens=None):
        queries = split_by_heads(self.w_q(queries), self.num_heads)
        keys = split_by_heads(self.w_k(keys), self.num_heads)
        values = split_by_heads(values, self.num_heads)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, self.num_heads, dim=0)
        outputs = self.attention(queries, keys, values, valid_lens)
        outputs = recover(outputs, self.num_heads)
        return outputs



class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hidden, num_heads, dropout, bias=False):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.w_q = nn.Linear(query_size, num_hidden, bias=bias)
        self.w_k = nn.Linear(key_size, num_hidden, bias=bias)
        self.w_v = nn.Linear(value_size, num_hidden, bias=bias)
        self.w_o = nn.Linear(num_hidden, num_hidden, bias=bias)
        nn.init.normal_(self.w_q.weight, mean=0, std=np.sqrt(2.0 / (query_size + key_size)))
        nn.init.normal_(self.w_k.weight, mean=0, std=np.sqrt(2.0 / (query_size + key_size)))
        nn.init.normal_(self.w_v.weight, mean=0, std=np.sqrt(2.0 / (query_size + value_size)))

    def forward(self, queries, keys, values, valid_lens=None):
        queries = split_by_heads(self.w_q(queries), self.num_heads)
        keys = split_by_heads(self.w_k(keys), self.num_heads)
        values = split_by_heads(self.w_v(values), self.num_heads)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, self.num_heads, dim=0)
        outputs = self.attention(queries, keys, values, valid_lens)
        outputs = recover(outputs, self.num_heads)
        return self.w_o(outputs)


class ResidualMultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hidden, num_heads, dropout, bias=False):
        super(ResidualMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.w_q = nn.Linear(query_size, num_hidden, bias=bias)
        self.w_k = nn.Linear(key_size, num_hidden, bias=bias)
        self.w_v = nn.Linear(value_size, num_hidden, bias=bias)
        # self.w_o = nn.Linear(num_hidden, num_hidden, bias=bias)
        self.dropout = nn.Dropout(p=0.)
        nn.init.normal_(self.w_q.weight, mean=0, std=np.sqrt(2.0 / (query_size + key_size)))
        nn.init.normal_(self.w_k.weight, mean=0, std=np.sqrt(2.0 / (query_size + key_size)))
        nn.init.normal_(self.w_v.weight, mean=0, std=np.sqrt(2.0 / (query_size + value_size)))

    def forward(self, queries, keys, values, valid_lens=None):
        res = queries
        queries = split_by_heads(self.w_q(queries), self.num_heads)
        keys = split_by_heads(self.w_k(keys), self.num_heads)
        values = split_by_heads(self.w_v(values), self.num_heads)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, self.num_heads, dim=0)
        outputs = self.attention(queries, keys, values, valid_lens)
        # outputs = recover(queries + self.dropout(outputs), self.num_heads)
        outputs = recover(outputs, self.num_heads)
        return self.dropout(outputs) + res
        #return outputs

class TemporalAttention(nn.Module):
    def __init__(self, in_size, dropout=0.5, squeeze_ratio=8, bias=False):
        super(TemporalAttention, self).__init__()
        conv_in = in_size // squeeze_ratio
        self.squeeze = nn.Linear(in_size, conv_in, bias=bias)
        self.dp = nn.Dropout(dropout)
        self.conv1 = Conv1dBnRelu(conv_in, conv_in//2, bias)
        self.conv2 = Conv1dBnRelu(conv_in//2, conv_in//4, bias)
        self.conv3 = Conv1dBnRelu(conv_in // 4, 1, bias)


    def forward(self, k, v, mask):
        # x = self.dp(k)
        x = self.squeeze(k)
        x = torch.transpose(x, -1, -2)
        x = self.conv1(x, mask)
        x = self.conv2(x, mask)
        x = self.conv3(x, mask)
        x = F.softmax(x)
        x = torch.bmm(x, v)
        return x