import itertools
import math
import random

import torch
from torch.utils.data import Dataset
from itertools import chain
import os
import h5py
import numpy as np
import joblib

gestures = ['Clockwise', 'Counterclockwise', 'Pull', 'Push', 'SlideLeft', 'SlideRight']
negative_samples = ['liftleft', 'liftright', 'sit', 'stand', 'turn', 'walking', 'waving']
envs = ['e1', 'e2', 'e3', 'e4', 'e5', 'e6']
participants = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8', 'u9', 'u10', 'u11', 'u12', 'u13',
                'u14', 'u15', 'u16', 'u17', 'u18', 'u19', 'u20', 'u21', 'u22', 'u23', 'u24', 'u25']
locations = ['p1', 'p2', 'p3', 'p4', 'p5']

participant_domain = [['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7'],
                      ['u8', 'u9', 'u10', 'u11', 'u12', 'u13', 'u14', 'u15', 'u16', 'u17', 'u18', 'u19', 'u20', 'u21',
                       'u22', 'u23', 'u24', 'u25']]

root = 'D:\\dataset\\mmWave_cross_domain_gesture_dataset'

st = {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}


# file_example n_liftleft_e1_u1_p1_s1.npy
def get_samples(prefix):
    index = 1
    prefix = prefix + '_s{:d}.npy'
    file_name = prefix.format(index)
    samples = []
    while os.path.exists(os.path.join(root, file_name)):
        samples.append(file_name)
        index += 1
        file_name = prefix.format(index)
    return samples


train_data = []
train_label = []
test_data = []


def assign_samples_k_fold(act, samples):
    if len(samples) <= 0:
        return
    random.shuffle(samples)
    size = len(samples) // 5
    chunks = [samples[i:i + size] for i in range(0, len(samples), size)]
    for i, value in enumerate(chunks):
        train_data[i].extend(value)



def split_data(domain):
    file_format = '{type}_{act}_{env}_{user}'
    if domain == 'in_domain':
        train_data.extend([[], [], [], [], []])
        train_label.extend([[], [], [], [], []])
        for act in itertools.chain(gestures, negative_samples):
            if act in gestures:
                is_gesture = 'y'
            else:
                is_gesture = 'n'
            for e in envs:
                for u in participants:
                    prefix = file_format.format(type=is_gesture, act=act, env=e, user=u)
                    if act == 'walking':
                        samples = get_samples(prefix)
                        assign_samples_k_fold(act, samples)
                    else:
                        for p in locations:
                            samples = get_samples(prefix + '_' + p)
                            assign_samples_k_fold(act, samples)
    elif domain == 'person':
        pass


class di_gesture_dataset(Dataset):
    def __init__(self, file_names):
        self.len = len(file_names)
        self.cache_size = 50
        self.data = np.empty(self.cache_size)
        self.cache_index = (0, self.cache_size)
        for i in range(self.cache_size):
            item = np.load(os.path.join(root, file_names[i]))
            self.data[i] = item

    def __getitem__(self, index):
        if index <= self.cache_index[0] < self.cache_index[1]:
            return

    def __len__(self):
        pass


split_data('in_domain')

print(st)
