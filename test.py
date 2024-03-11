import numpy as np
import torch
import torch.nn.functional as F


if __name__ == '__main__':
    # 创建一个示例序列（这里用随机数据代替）

    sequence = torch.rand(size=(2, 4, 4)).unsqueeze(0).unsqueeze(0)
    print(sequence)
    output_tensor = torch.nn.functional.interpolate(sequence, size=(3, 4, 4), mode='trilinear', align_corners=False)
    # 对第一个维度进行降采样
    # downsampled_sequence = down_sample(sequence, 40)
    print(output_tensor)

