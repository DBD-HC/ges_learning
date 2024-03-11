import os

import torch
from torch.utils.data import Dataset
from data.di_gesture_dataset import random_translation, random_scale_radiated_power
import numpy as np

from utils import random_geometric_features

cube_rai_data_path = '/root/autodl-tmp/dataset/cubelearn_arm_rai'
file_format = 'u{user}_g{ges}_s{sample}.npy'

cube_users = ['0', '1', '2', '3', '4', '5', '6', '7']

cube_gestures = ['SL', 'SD', 'FL', 'FD', 'FAL', 'FAD', 'PUSH', 'PULL', 'SWR', 'SWL', 'DL', 'DD']


def cube_combine(index, train_data, train_label):
    return CubeArmGesture(sum([x for i, x in enumerate(train_data) if i not in index], []),
                          sum([x for i, x in enumerate(train_label) if i not in index], []), transform=data_augmentation), \
           CubeArmGesture(sum([x for i, x in enumerate(train_data) if i in index], []),
                          sum([x for i, x in enumerate(train_label) if i in index], []))


def cube_split_data(domain=0, fold=0, user=(6, 7)):
    data_for_train = []
    label_for_train = []
    file_set = set(os.listdir(cube_rai_data_path))
    if domain == 0:
        print('in domain')
    else:
        data_for_train = [[], [], [], [], [], [], [], []]
        label_for_train = [[], [], [], [], [], [], [], []]
        for g_i, ges in enumerate(cube_gestures):
            for u_i, u in enumerate(cube_users):
                index = 0
                file_name = file_format.format(user=u, ges=ges, sample=index)
                while file_name in file_set:
                    data_for_train[u_i].append(file_name)
                    label_for_train[u_i].append(g_i)
                    index += 1
                    file_name = file_format.format(user=u, ges=ges, sample=index)
        return cube_combine(user, data_for_train, label_for_train)


def data_augmentation(d, label, position):
    # d = data_normalization(d)
    d, dis, angle = random_translation(d, position)
    d = random_scale_radiated_power(d, position, None, None)
    #d = random_geometric_features(d)
    # d = cfar(d)
    return d, label


class CubeArmGesture(Dataset):
    def __init__(self, filenames, labels, transform=None):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform
        self.len = len(self.filenames)

    def __getitem__(self, index):
        data = np.load(os.path.join(cube_rai_data_path, self.filenames[index]))
        label = self.labels[index]
        # res = self.transform(data, label)
        #data_var = np.var(data)
        #data_mean = np.mean(data)
        #data = (data - data_mean) / np.sqrt((data_var + 1e-9))

        if self.transform is not None:
            data, label = self.transform(data, label, None)
        else:
            data_var = np.var(data)
            data_mean = np.mean(data)
            data = (data - data_mean) / np.sqrt((data_var + 1e-9))
        return torch.from_numpy(data).type(torch.float32), None, torch.tensor(label), index

    def __len__(self):
        return self.len
