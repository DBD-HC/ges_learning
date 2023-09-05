import numpy as np
import matplotlib.pylab as plt
import torch

from data.di_gesture_dataset import *


visdom = visdom.Visdom(env='raw-data', port=6006)




filename = 'y_Push_e4_u9_p3_s1.npy'

d = np.load(os.path.join(root, filename))


dt = np.random.random((10, 10, 5, 5))
dt_mean = np.mean(dt, axis= 1)
std1 = np.std(dt, axis= 1)
std2 = np.sqrt(np.mean(dt * dt, axis= 1) - dt_mean * dt_mean)

print(np.mean(dt * dt, axis= 1) > dt_mean * dt_mean)
print(std2)
def max_mean_std(data, dim=1):
    item_max = np.squeeze(np.max(data, axis=dim))[None, :]
    item_mean = np.squeeze(np.mean(data, axis=dim))[None, :]
    item_std = np.squeeze(np.std(data, axis=dim))[None, :]
    item = np.concatenate((item_max, item_mean, item_std), axis=0)
    return item


# for index, item in enumerate(d):
#     item_range = max_mean_std(item, dim=1)
#     item_angle = max_mean_std(item, dim=0)
#     visdom.heatmap(item, win=str(index % 10), opts=dict(title=filename + '_' + str(index)))
#     visdom.heatmap(item_angle, win='angle_h'+str(index % 10), opts=dict(title=filename + '_angle_' + str(index)))
#     visdom.heatmap(item_range, win='range_h' + str(index % 10), opts=dict(title=filename + '_range_' + str(index)))

g_d = max_mean_std(d, dim=0)

visdom.heatmap(g_d[0], win=filename+'track0', opts=dict(title=filename+'0'))
visdom.heatmap(g_d[1], win=filename+'track1', opts=dict(title=filename+'1'))
visdom.heatmap(g_d[2], win=filename+'track2', opts=dict(title=filename+'2'))


