import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

# 假设 batch_size=3，序列长度分别为 5，4，2
x = [torch.randn(5, 10), torch.randn(4, 10), torch.randn(2, 10)]
lengths = [5, 4, 2]

# 使用 pack_padded_sequence 打包序列
packed_sequence = pad_sequence(torch.stack(x), batch_first=True)



# 使用 pad_packed_sequence 恢复序列
output_sequence, _ = pad_packed_sequence(packed_sequence, batch_first=True)