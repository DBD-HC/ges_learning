from data.di_gesture_dataset import *


visdom = visdom.Visdom(env='raw-data', port=6006)


filename = 'y_Push_e1_u9_p1_s2.npy'

d = np.load(os.path.join(root, filename))

for index, item in enumerate(d):
    visdom.heatmap(item, win=str(index), opts=dict(title=filename + '_' + str(index)))




