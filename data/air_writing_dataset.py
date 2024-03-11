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

participants = ['participant1', 'participant2', 'participant3', 'participant4', 'participant9', 'participant10',
                'participant11', 'participant12']

gestures = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

gesture_index = np.arange(0, 800, 1)
train_data, val_data, test_data = [], [], []
train_label, val_label, test_label = [], [], []

data_path = '/root/autodl-tmp/dataset/air_writing'
rdai_path = '/root/autodl-fs/air-writing'
rai_file_format = 'rai_{user}_{ges}_{s}.npy'

samples_per_participant = 100
samples_per_gesture = 10



train_data_filenames = []
train_label_list = []
filenames_set = set(os.listdir(data_path))


def combine(index):
    return GestureDataset(sum([x for i, x in enumerate(train_data_filenames) if i != index], []),
                              sum([x for i, x in enumerate(train_label_list) if i != index], [])), \
           GestureDataset(train_data_filenames[index], train_label_list[index])


def split_data_air_writing(domain=0, fold=0, env_index=0, position_index=0, person_index=0):
    # file_format = 'rai_{ges}_{user}_{position}_{env}_s{sample}.npy'
    if domain == 0:
        train_data_filenames.extend([[], [], []])
        train_label_list.extend([[], [], []])
        size = len(filenames_set) // len(train_data_filenames)
        j = 0
        # temp = []
        #         filenames = os.listdir(root)
        #         for filename in filenames:
        #             chunk = filename.split('_')
        #             if chunk[2] == 0:
        #                 temp.append(filename)
        #         filenames = temp
        filenames = os.listdir(data_path)
        random.shuffle(filenames)
        for i in range(0, len(filenames_set), size):
            if j == len(train_data_filenames) -1:
                train_data_filenames[j].extend(filenames[i:])
            else:
                train_data_filenames[j].extend(filenames[i:i + size])
            j += 1
        # chunks = [filenames[i:i + size] for i in range(0, len(samples), size)]
    elif domain == 1:
        train_data_filenames.extend([[], [], [], [], [], [], [], []])
        train_label_list.extend([[], [], [], [], [], [], [], []])
    print()
    domain_index = [0, 0, 0, 0]
    val_domain_index = [fold, person_index, 0, 0]
    for g_i, g in enumerate(gestures):
                for u_i, u in enumerate(participants):
                    domain_index[1] = u_i
                    index = 5
                    filename = rai_file_format.format(ges=g, user=u, s=index)
                    while filename in filenames_set:
                        train_data_filenames[domain_index[domain]].append(filename)
                        train_label_list[domain_index[domain]].append(g_i)
                        index += 1
                        filename = rai_file_format.format(user=u, ges=g, s=index)
    return combine(val_domain_index[domain])


class GestureDataset(Dataset):
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
        data = (data - data_mean)/np.sqrt((data_var + 1e-9))
        return torch.from_numpy(data).type(torch.float32), torch.tensor(label)

    def __len__(self):
        return self.len


if __name__ == '__main__':
    get_RDAI_2()
    visdom = visdom.Visdom(env='air-writing', port=6006)
    train_datas, test_datas = split_RDAI()
    for it in tqdm(range(30)):
        d, l = train_datas.__getitem__(it)
        map = torch.sum(d, dim=0)
        visdom.heatmap(map, win=str(l) + '_range angel', opts=dict(title=str(l) + 'range angel'))
        # for fi, frame in enumerate(d):
        #     visdom.heatmap(frame, win=str(fi % 20) + '_range angel', opts = dict(title=str(fi) + 'range angel fft'))
