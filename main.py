from pyheat import PyHeat
import numpy as np
import os
import visdom

# root = 'D:\\dataset\\Triple Sensor Data\\Button_Press'

# ph = PyHeat('di_gesture_dataset.py')
# ph.create_heatmap()
# ph.show_heatmap()
# print(np.argmax(array.reshape(len(array), -1), axis=-1), array[0].shape)

# d = np.load(os.path.join(root, '20220722105421.npy'))
# print(d)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

# print(z_score(arr2, factor=0.8))
n1 = 'y_Pull_e1_u3_p2_s3.npy'
n2 = 'y_Push_e1_u3_p2_s3.npy'

visdom = visdom.Visdom(env='pairs data', port=6006)

root = '/root/autodl-nas/mmWave_cross_domain_gesture_dataset'
s1 = np.load(os.path.join(root, n1))
s2 = np.load(os.path.join(root, n2))
for i, x in enumerate(s1):
    visdom.heatmap(x, win='1_'+ str(i),
                           opts=dict(title='Pull s = ' + str(i)))

for i, x in enumerate(s2):
    visdom.heatmap(x, win='2_'+ str(i),
                           opts=dict(title='Push s = ' + str(i)))

