import numpy as np
import torch
import torch.nn.functional as F



def down_sample(frames, target_data_len):
    data_len = len(frames)
    n_delete_frame = data_len - target_data_len
    gap = data_len / n_delete_frame
    temp = 0
    target_frames = torch.empty((target_data_len + 1, frames.size(-2), frames.size(-1)))
    i = 0
    for frame in frames:
        temp += 1
        if temp < gap:
            target_frames[i] = frame
            i += 1
        else:
            temp -= gap
    return target_frames[:target_data_len]

def up_sample(frames, target_data_len):
    data_len = len(frames)
    n_insert_frame = target_data_len - data_len
    gap = data_len / n_insert_frame
    temp = 0
    target_frames = torch.empty((target_data_len + 1, frames.size(-2), frames.size(-1)))
    i = 0
    for frame in frames:
        temp += 1
        if temp < gap:
            target_frames[i] = frame
            i += 1
        else:
            temp -= gap
    return target_frames[:target_data_len]

if __name__ == '__main__':
    # 创建一个示例序列（这里用随机数据代替）

    sequence = torch.rand(size=(2, 4, 4)).unsqueeze(0).unsqueeze(0)
    print(sequence)
    output_tensor = torch.nn.functional.interpolate(sequence, size=(3, 4, 4), mode='trilinear', align_corners=False)
    # 对第一个维度进行降采样
    # downsampled_sequence = down_sample(sequence, 40)
    print(output_tensor)

