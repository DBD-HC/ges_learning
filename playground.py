import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

import numpy as np

# 生成示例数据
data = np.array([[20, 20, 21], [20, 30, 40], [30, 40, 50]])

# 计算每一列的和
col_sum = np.sum(data, axis=0)

# 计算每一列的百分比
col_pct = 100 * data / col_sum[np.newaxis, :]
col_pct = np.round(col_pct, 1)

print(col_pct)


