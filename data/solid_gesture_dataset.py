import math

import torch
from torch.utils.data import Dataset
import random
import os
import h5py
import numpy as np
import joblib
from tqdm import tqdm
import visdom
from utils import *

gestures = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
multi_user_session = ['2', '3', '5', '6', '8', '9', '10', '11', '12', '13']
single_user_session = ['0', '1', '4', '7', '13', '14']
data_path = '/root/autodl-tmp/dataset/dsp/'
rdi_file_format = '{ges}_{session}_{sample}.npy'
visdom = visdom.Visdom(env='solid gesture', port=6006)
file_cache = {}
state = None
file_name_for_train = []
label_for_train = []


def combine(index):
    return SolidGesDataset(sum([x for i, x in enumerate(file_name_for_train) if i != index], []),
                           sum([x for i, x in enumerate(label_for_train) if i != index], []), data_normalization), \
           SolidGesDataset(file_name_for_train[index], label_for_train[index], data_normalization)


def split_dataset(type='cross_person', fold=0):
    file_list = os.listdir(data_path)
    if state != type:
        file_name_for_train.clear()
        label_for_train.clear()
        if type == 'cross_person':
            file_name_for_train.extend([[], [], [], [], [], [], [], [], [], []])
            label_for_train.extend([[], [], [], [], [], [], [], [], [], []])
            session_list = multi_user_session
        else:
            file_name_for_train.extend([[], [], [], [], [], []])
            label_for_train.extend([[], [], [], [], [], []])
            session_list = single_user_session
        for ges in gestures:
            for s_id, session in enumerate(session_list):
                sample_idx = 0
                filename = rdi_file_format.format(ges, session, sample_idx)
                while filename in file_list:
                    file_name_for_train[s_id].append(filename)
                    label_for_train[s_id].append(int(ges))
                    sample_idx += 1
    return combine(fold)


class SolidGesDataset(Dataset):
    def __init__(self, filenames, labels, transform=None):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform
        self.len = len(self.filenames)

    def __getitem__(self, index):
        data = np.load(os.path.join(data_path, self.filenames[index]))
        label = self.labels[index]
        # res = self.transform(data, label)
        data_var = np.var(data)
        data_mean = np.mean(data)
        data = (data - data_mean) / np.sqrt((data_var + 1e-9))
        return torch.from_numpy(data).type(torch.float32), torch.tensor(label)

    def __len__(self):
        return self.len
