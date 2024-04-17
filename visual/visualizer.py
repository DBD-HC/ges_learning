import time

import numpy as np
import matplotlib.pylab as plt
import torch
import torch.nn.functional as F

from data.mcd_dataset import *


visdom = visdom.Visdom(env='visual', port=6006)




filename = 'y_Push_e4_u9_p1_s1.npy'

d = np.load(os.path.join(rai_data_root, filename))

d = torch.from_numpy(d)

x = np.ones(32)[:, None] * np.arange(32)[None, :]
y = np.arange(32)[:, None] * np.ones(32)[None, :]
x = x - 16

sin_map = x / np.sqrt(x**2 + y**2 + 1e-9)
rang_map = np.sqrt(x**2 + y**2)
rang_map = (rang_map - 16)/16

#y = np.ascontiguousarray(y[:, ::-1])
x = torch.from_numpy(sin_map)
y = torch.from_numpy(rang_map)

grid = torch.cat((x[:, :, None], y[:, :, None]), dim=-1)

new =  F.grid_sample(d[None, :], grid[None, :])
new = torch.squeeze(new)

for i, frame in enumerate(d):
    visdom.heatmap(X=frame, win='1')
    visdom.heatmap(X=new[i], win='2')
    time.sleep(1)




