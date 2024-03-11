import os

import torch.nn.functional

from data.di_gesture_dataset import *
import numpy as np

rai_data_path = '/root/autodl-tmp/dataset/mmWave_cross_domain_gesture_dataset'
heatmap_data_path = '/root/autodl-tmp/dataset/mmwave_2d_map/'
filenames = os.listdir(rai_data_root)


def unify_sequence(x, data_len):
    x = x.unsqueeze(0).unsqueeze(0)
    x = torch.nn.functional.interpolate(x, size=(data_len, x.size(-1)), mode='bilinear', align_corners=False)
    return torch.squeeze(x)


def trans_2_RI_AI(data_len=64):
    for i,name in tqdm(enumerate(filenames)):
        if i < 10600:
            continue
        rdi = np.load(os.path.join(rai_data_path, name))
        rdi = torch.from_numpy(rdi)
        ri = torch.mean(rdi, dim=-1)
        ai = torch.mean(rdi, dim=-2)
        ri = unify_sequence(ri, data_len)
        ai = unify_sequence(ai, data_len)
        ri = torch.transpose(ri, -2, -1)
        ai = torch.transpose(ai, -2, -1)
        np.save(os.path.join(heatmap_data_path, 'RI', name), ri.numpy())
        np.save(os.path.join(heatmap_data_path, 'AI', name), ai.numpy())


trans_2_RI_AI()
