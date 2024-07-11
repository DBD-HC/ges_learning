import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from data.data_splitter import data_normalization
from data.rai_ges_dataset import random_translation
from utils import simple_shift, random_geometric_features, random_data_len_adjust_2

#static_angle_range = np.arange(-20, 21)
#static_distance_range = np.arange(-6, 7)
#data_len_adjust_gap = [-3, -4, -5]


def compare_replace(d, v):
    min_d = d[0]
    max_d = d[1]
    if d[0] > v:
        min_d = v
    if d[1] < v:
        max_d = v
    return min_d, max_d


def z_score(d):
    mean_d = np.mean(d, axis=(-2, -1), keepdims=True)
    var_d = np.var(d, axis=(-2, -1), keepdims=True)
    d = (d - mean_d) / np.sqrt(var_d + 1e-9)
    return d

target_static_angle_range = np.arange(-4, 5)
target_static_distance_range = np.arange(-4, 5)

def random_translation2(datas):

    return datas

def data_augmentation_target(d, data_type=None):
    d_distance = random.choice(target_static_distance_range)
    d_angle = random.choice(target_static_angle_range)
    simple_shift(d, d_distance, d_angle)
    d = random_geometric_features(d)
    d = random_data_len_adjust_2(d)
    return d


def get_track(rai):
    rai_max = np.max(rai, axis=0)
    rai_mean = np.mean(rai, axis=0)
    rai_std = np.std(rai, axis=0)
    global_track = np.concatenate((rai_max[None, :], rai_mean[None, :], rai_std[None, :]), axis=0)
    return global_track


class DANDataset(Dataset):
    def __init__(self, file_names=None, labels=None, data_root=None, transform=None):
        self.data_root = data_root
        # self.data_root = '/root/autodl-tmp/dataset/mmWave_cross_domain_gesture_dataset'

        self.file_names = file_names
        self.labels = labels
        self.transform = transform

        self.len = len(self.labels)

    def __len__(self):
        return self.len

    def get_data(self, index):
        d1 = np.load(os.path.join(self.data_root, self.file_names[index]))
        label1 = self.labels[index]
        d1_aug = np.zeros_like(d1)

        d1_aug[:] = d1[:]
        d1 = self.transform(d1, 0)
        d1_aug = self.transform(d1_aug, 0)
        #d1 = data_augmentation_source(d1, 0)
        #d1 = z_score(d1)
        #d1_aug = z_score(d1_aug)

        d1 = torch.from_numpy(d1).type(torch.float32)
        label1 = torch.tensor(label1)

        d1_aug = torch.from_numpy(d1_aug).type(torch.float32)

        return d1, d1_aug, label1

    def __getitem__(self, index):
        # label = torch.tensor([int(label1 == label2)], dtype=torch.float32)
        return self.get_data(index)
