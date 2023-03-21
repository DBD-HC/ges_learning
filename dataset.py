import math

import torch
from torch.utils.data import Dataset
from functools import lru_cache
import os
import h5py
import numpy as np
import joblib

participants = ['participant1', 'participant2', 'participant3', 'participant4', 'participant9', 'participant10',
                'participant11', 'participant12']

gestures = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

gesture_index = np.arange(0, 800, 1)
train_data, val_data, test_data = [], [], []

data_path = '/root/autodl-nas/'

samples_per_participant = 100
samples_per_gesture = 10


def split_dataset(split_ratio=(0.6, 0.2, 0.2)):
    try:
        data = joblib.load(os.path.join(data_path, 'data_cache.p'))
        train_data.extend(data['train_data'])
        val_data.extend(data['validation_data'])
        test_data.extend(data['test_data'])
    except FileNotFoundError as e:
        print("未发现缓存数据集分割结果，重新分割")
        np.random.shuffle(gesture_index)
        train_len = int(len(gesture_index) * split_ratio[0])
        valid_len = int(len(gesture_index) * (split_ratio[0] + split_ratio[1]))
        train_data.extend(gesture_index[:train_len])
        val_data.extend(gesture_index[train_len:valid_len])
        test_data.extend(gesture_index[valid_len:])
        data = {'train_data': train_data,
                'validation_data': val_data,
                'test_data': test_data}
        joblib.dump(data, os.path.join(data_path, 'data_cache.p'))
    return Gesture_Dataset('train'), Gesture_Dataset('validation'), Gesture_Dataset('test')


# @lru_cache(maxsize=4)
def get_data(participant_index, gesture_type):
    with h5py.File(os.path.join(data_path, participants[participant_index], str(gesture_type), "RawData.mat"),
                   'r') as f:
        # 读取指定数据集，例如读取名为data的数据集
        raw_data = f['rawdata'][:]
        # 将数据集转换为numpy数组
        raw_data = np.array(raw_data)
        raw_data = raw_data.view('complex')
        raw_data = raw_data.reshape((10, 100, 128, 100, 2))
        raw_data = raw_data.transpose((0, 1, 2, 4, 3))

    return raw_data


class Gesture_Dataset(Dataset):
    def __init__(self, mode):
        self.mode = mode
        if mode == 'train':
            self.len = len(train_data)
            self.ges_index = train_data
        elif mode == 'validation':
            self.len = len(test_data)
            self.ges_index = val_data
        else:
            self.len = len(val_data)
            self.ges_index = test_data

    def __getitem__(self, index):
        participant_index = math.floor(self.ges_index[index] / samples_per_participant)
        gesture_type = math.floor((self.ges_index[index] % samples_per_participant) / samples_per_gesture)
        ges_index_in_class = (self.ges_index[index] % samples_per_participant) % samples_per_gesture
        raw_data = get_data(participant_index, gesture_type)
        raw_data = torch.from_numpy(raw_data[ges_index_in_class]).to(torch.complex64)

        return raw_data, torch.tensor(gesture_type)

    def __len__(self):
        return self.len
