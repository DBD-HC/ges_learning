import itertools
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import scipy.io
import scipy.signal
import visdom

from data.data_splitter import DataSpliter
from utils import *

# import cupy as cp

# root = 'D:\\data\\mmWave_cross_domain_gesture_dataset'
# root = '/root/autodl-nas/mmWave_cross_domain_gesture_dataset'
root = '/root/autodl-tmp/dataset/complex_rai/rai_data'
# root = '/root/autodl-tmp/dataset/mmWave_cd_rdi/mmwave_rdi'
# root = '/root/autodl-tmp/dataset/mmwave_rai_lessrx'
# root = '/root/autodl-tmp/dataset/mmwave_rai_2rx'

st = {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}
st_act = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}


static_angle_range = [np.arange(-6, 7), np.arange(-20, 21)]
static_distance_range = [np.arange(-6, 7), np.arange(-20, 21)]

def random_translation(datas, position, p=0.5):

    if random.uniform(0, 1) > p:
        d_distance = random.choice(static_distance_range[0])
        d_angle = random.choice(static_angle_range[0])
    else:
        d_distance = random.choice(static_distance_range[1])
        d_angle = random.choice(static_angle_range[1])
    # datas = shift(datas, [0, d_distance, 0], cval=datas.min())
    simple_shift(datas, d_distance, d_angle)
    return datas, d_distance, d_angle


def random_scale_radiated_power(datas, position, distance, angle, p=0.5):
    #  if position < locations[3]:
    #         if abs(angle) > 10:
    #             scale_factor = random.uniform(0.2, 0.8)
    #         else:
    #             scale_factor = random.uniform(0.4, 1.2)
    #     elif position == locations[3]:
    #         if angle < 0:
    #             scale_factor = random.uniform(1, 1.2)
    #         else:
    #             scale_factor = random.uniform(0.8, 1)
    #     else:
    #         if angle > 0:
    #             scale_factor = random.uniform(1, 1.2)
    #         else:
    #             scale_factor = random.uniform(0.8, 1)
    scale_factor = random.uniform(0.8, 1.2)
    datas = scale_factor * data_normalization(datas)
    return datas


def data_augmentation(d, label, position):
    # d = data_normalization(d)
    d, dis, angle = random_translation(d, position)
    d = random_geometric_features(d)
    #d, label = random_reverse(d, label)
    d = random_data_len_adjust_2(d)
    # d = random_scale_radiated_power(d, position, dis, angle)
    d = data_normalization(d)
    # d = cfar(d)
    return d, label


class ComplexDataSplitter(DataSpliter):
    def __init__(self, data_path = root):
        self.gestures =  ['0', '1', '2', '3', '4', '5']
        self.envs =  ['0', '1', '2', '3']
        self.participants = ['0', '1', '2']
        self.locations = ['0', '1', '2', '3', '4']

        super(ComplexDataSplitter, self).__init__(data_path=data_path, domain_num=(3, len(self.envs), len(self.locations),len( self.participants)))
        self.file_format = 'rai_{ges}_{user}_{position}_{env}_s{sample}.npy'
        self.pre_domain = -1

    def get_domain_num(self, domain):
        return self.domain_num[domain]

    def clear_cache(self):
        self.train_data_filenames.clear()
        self.train_label_list.clear()
        self.test_data_filenames.clear()
        self.test_label_list.clear()

    def combine(self, index):
        return ComplexRAI(sum([x for i, x in enumerate(self.train_data_filenames) if i != index], []),
                              sum([x for i, x in enumerate(self.train_label_list) if i != index], []), data_augmentation), \
           ComplexRAI(self.train_data_filenames[index], self.train_label_list[index], data_normalization)

    def split_data(self, domain, k=None):
        if self.pre_domain != domain:
            self.clear_cache()
            self.pre_domain = domain

        domain_index = [0, 0, 0, 0]
        domain_num = self.domain_num[domain]
        if len(self.train_data_filenames) == 0:
            for i in range(domain_num):
                self.train_data_filenames.append([])
                self.train_label_list.append([])
            if domain == 0:
                for g_i, g in enumerate(self.gestures):
                    for e in self.envs:
                        for p in self.locations:
                            for u_i, u in enumerate(self.participants):
                                index = 1
                                filename = self.file_format.format(ges=g, user=u, position=p, env=e, sample=index)
                                temp_data = []
                                temp_label = []
                                while filename in self.filenames_set:
                                    temp_data.append(filename)
                                    temp_label.append(g_i)
                                    index += 1
                                    filename = self.file_format.format(ges=g, user=u, position=p, env=e, sample=index)
                                if len(temp_data) == 0:
                                    continue
                                size = len(temp_data) // len(self.train_data_filenames)
                                random.shuffle(temp_data)
                                j = 0
                                for i in range(0, len(temp_data), size):
                                    if j == len(self.train_data_filenames) - 1:
                                        self.train_data_filenames[j].extend(temp_data[i:])
                                        self.train_label_list[j].extend(temp_label[i:])
                                        break
                                    else:
                                        self.train_data_filenames[j].extend(temp_data[i:i + size])
                                        self.train_label_list[j].extend(temp_label[i:i + size])
                                    j += 1
            else:
                for g_i, g in enumerate(self.gestures):
                    for e_i, e in enumerate(self.envs):
                        domain_index[1] = e_i
                        for p_i, p in enumerate(self.locations):
                            domain_index[2] = p_i
                            for u_i, u in enumerate(self.participants):
                                domain_index[3] = u_i
                                index = 1
                                filename = self.file_format.format(ges=g, user=u, position=p, env=e, sample=index)
                                while filename in self.filenames_set:
                                    self.train_data_filenames[domain_index[domain]].append(filename)
                                    self.train_label_list[domain_index[domain]].append(g_i)
                                    index += 1
                                    filename = self.file_format.format(ges=g, user=u, position=p, env=e, sample=index)
        return self.combine(k)


def get_track(rai):
    rai_max = np.max(rai, axis=0)
    rai_mean = np.mean(rai, axis=0)
    rai_std = np.std(rai, axis=0)
    global_track = np.concatenate((rai_max[None, :], rai_mean[None, :], rai_std[None, :]), axis=0)
    return global_track

class ComplexRAI(Dataset):
    def __init__(self, file_names, labels, transform=None, max_frames=128, need_augmentation=True, data_root=None):
        self.len = len(file_names)
        self.max_frame = max_frames
        self.file_names = np.array(file_names)

        self.labels = np.array(labels)
        self.labels = np.where(self.labels < 6, labels, 6)
        if data_root is not None:
            self.data_root = data_root
        else:
            self.data_root = root
        if need_augmentation:
            self.transform = transform
        else:
            self.transform = data_normalization

    def __getitem__(self, index):
        d = np.load(os.path.join(self.data_root, self.file_names[index]))
        # d = file_cache[self.file_names[index]]
        label = self.labels[index]
        res = self.transform(d, self.labels[index], None)
        if isinstance(res, tuple):
            d = res[0]
            label = res[1]
        else:
            d = res
        #d = d[:-10]
        track = get_track(d)
        return torch.from_numpy(d).type(torch.float32), torch.from_numpy(track).type(torch.float32), torch.tensor(
            label), index

    def __len__(self):
        return self.len



if __name__ == '__main__':

    # clear_data()
    print('110')

