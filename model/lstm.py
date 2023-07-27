import torch.nn as nn
import torch
from model.attention import *


def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xi, W_hi, b_i = three()  # 输入门参数
    W_xf, W_hf, b_f = three()  # 遗忘门参数
    W_xo, W_ho, b_o = three()  # 输出门参数
    W_xc, W_hc, b_c = three()  # 候选记忆元参数
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))


def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        hn, cn = h0[0].to(x.device), c0[0].to(x.device)

        for i in range(x.size(1)):
            hn, cn = self.lstm(x[:, i, :], (hn, cn))

        out = self.fc(hn)
        return out


class AttentionLstm(nn.Module):
    def __init__(self, input_size, hidden_size, head=4, dropout=0.5):
        super(AttentionLstm, self).__init__()
        self.hidden_size = hidden_size

        self.multi_head_attention = MultiHeadAttention(query_size=input_size, key_size=input_size,
                                                       value_size=input_size, num_hidden=hidden_size,
                                                       num_heads=head, dropout=nn.Dropout(dropout))
        self.fc_i = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, data_lens):
        x = self.fc_i(x)
        h0 = x[:, 0, :]
        h0 = h0[:, None, :]
        outputs = []
        final_state = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        for i in range(x.size(1)):
            x_i = x[:, i, :]
            x_i = x_i[:, None, :]
            h0 = self.multi_head_attention(x_i, h0, h0)
            h0 = h0 + x_i
            outputs.append(h0)
        outputs = torch.cat(outputs, dim=1)
        for i, l in enumerate(data_lens):
            final_state[i, :] = outputs[i, l-1, :]
        return outputs, final_state


class LSTM_2(nn.Module):
    def __init__(self, input_size, abstract_size, hidden_size):
        super(LSTM_2, self).__init__()
        self.abstract_size = abstract_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.pre_w = nn.Linear(input_size, abstract_size)
        self.temp_w = nn.Linear(input_size+abstract_size, hidden_size)
        self.out_w = nn.Linear(hidden_size, hidden_size)

    def forward(self, seqs):
        seq_len = seqs.size(1)
        batch_size = seqs.size(0)
        ab0 = torch.zeros(batch_size, self.abstract_size, device=seqs.device)

        hidden_states = []

        # 对于每个时间步
        for t in range(seq_len):
            input_seq = seqs[:,t,:]
            temp = torch.cat((ab0, input_seq), dim=-1)
            temp = self.temp_w(temp)
            temp = F.hardswish(temp)
            # 将隐藏状态保存到列表中
            hidden_states.append(temp[:, None, :])
            ab0 = self.pre_w(input_seq)
            ab0 = F.hardswish(ab0)

        # 将隐藏状态列表拼接并转换为PackedSequence
        hidden_states = torch.cat(hidden_states, dim=1)

        return hidden_states


