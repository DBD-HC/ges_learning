import math

import torch
from torch.utils.data import Dataset
from functools import lru_cache
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


def split_data(domain):
    train_data = []
    test_data = []
    file_list = os.listdir(root)
    num_walking = 50
    file_format = 'n_{act}_{env}_{user}'
    if domain == 'cross_domain':
        train_data.extend([[], [], [], []])
        for act in gestures, negative_samples:
            for e in envs:
                for u in participants:
                    prefix = file_format.format(act=act, env=e, user=u)
                    if act == 'walking':
                        samples = get_samples(prefix)
                    else:
                        for p in locations:


    pass


class di_gesture_dataset(Dataset):
    def __init__(self, domain):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
