import os
import time

import numpy as np
import matplotlib.pylab as plt
import torch
import torch.nn.functional as F
import visdom


visdom = visdom.Visdom(env='visual', port=6006)

root = '/root/autodl-tmp/dataset/mmWave_cross_domain_gesture_dataset'


filename = 'y_SlideLeft_e1_u11_p3_s2.npy'

data = np.load(os.path.join(root, filename))

new_data = np.empty((data.shape[0] + 10,32, 32))

new_data[0:4] = data[0]

new_data[-6:] = data[1]

new_data[4:-6] = data[:]

background = np.mean(new_data, axis=0)[None, :]

mask = new_data > background

dynamic_target = np.zeros(new_data.shape)

dynamic_target[mask] = 1

n_dynamic_target = np.sum(dynamic_target, axis=(-2, -1))

visdom.line(X=np.arange(new_data.shape[0])+1, Y=n_dynamic_target, win='dp')

def pool(x, dim=-1):
    x_avg = np.mean(x, axis=dim)[None, :]
    x_max = np.max(x, axis=dim)[None, :]
    x_std = np.std(x, axis=dim)[None, :]
    return np.concatenate((x_std, x_avg, x_max), axis=0)

for frame in data:
    visdom.heatmap(X=frame, win='rai')
    ri = pool(frame, dim=-1)
    di = pool(frame, dim=-2)
    visdom.heatmap(X=ri, win='ri', opts=dict(title='Example', width=800, height=600))
    visdom.heatmap(X=di, win='ai', opts=dict(title='Example', width=800, height=600))
