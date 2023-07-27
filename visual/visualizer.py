import numpy as np
import matplotlib.pylab as plt
from data.di_gesture_dataset import *


visdom = visdom.Visdom(env='raw-data', port=6006)


def sigmoid(x):
    return 1/(1+np.exp(-x))  # sigmoid函数

thr = 0.7
men = 0.1
std = 2

x = np.arange(-6.0,6.0,0.1)  # 限定x的范围，给什么区间画出来的就是在哪个区间
y = x * sigmoid(((x - thr)-men)/std)  # 求y值

plt.plot(x,y)
visdom.line(X=x, Y=y, win='cfar',
             opts=dict(title='cfar'))


# filename = 'y_Push_e1_u10_p1_s1.npy'
#
# d = np.load(os.path.join(root, filename))
#
# for index, item in enumerate(d):
#     visdom.heatmap(item, win=str(index%10), opts=dict(title=filename + '_' + str(index)))
#     visdom.heatmap(np.mean(item, axis=0)[:,None], win='angle_h'+str(index % 10), opts=dict(title=filename + '_angle_' + str(index)))
#     visdom.heatmap(np.mean(item, axis=1)[:,None], win='range_h'+str(index % 10), opts=dict(title=filename + '_range_' + str(index)))
#     visdom.line(X=np.arange(32), Y=np.mean(item, axis=0), win='angle'+str(index%10),
#              opts=dict(title=filename + '_angle_' + str(index)))
#     visdom.line(X=np.arange(32), Y=np.mean(item, axis=1), win='range' + str(index % 10),
#                 opts=dict(title=filename + '_range_' + str(index)))




