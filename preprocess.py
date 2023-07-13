# This code will display a Doppler Micro-Doppler Spectogram that will help
# you familiarize with the data
# By choosing the Person, Gesture and the Repeat, you will see that certain
# data displayed in a Spectogram

import h5py
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from tqdm import tqdm
import joblib

gestures = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

train_participants = ['participant1', 'participant2', 'participant3', 'participant9', 'participant10', 'participant11']
test_participants = ['participant4', 'participant12']

data_path = '/root/autodl-nas/'

##data_path = 'D:\\data\\air_writing\\'
data_path = '/root/autodl-nas/'

train_data_raw = np.empty((600, 100, 128, 2, 100), dtype=complex)
train_target = np.zeros(600, dtype=int)
test_data_raw = np.empty((200, 100, 128, 2, 100), dtype=complex)
test_target = np.zeros(200, dtype=int)
index = 0





print('=====preprocess train raw radar data=====')
for x in tqdm(train_participants):
    for g in gestures:
        with h5py.File(os.path.join(data_path, x, g, "RawData.mat"), 'r') as f:
            # 读取指定数据集，例如读取名为data的数据集
            raw_data = f['rawdata'][:]
            # 将数据集转换为numpy数组
            raw_data = np.array(raw_data)
            raw_data = raw_data.view('complex')
            raw_data = raw_data.reshape((10, 100, 128, 100, 2))
            raw_data = raw_data.transpose((0, 1, 2, 4, 3))

            train_data_raw[index:index + 10] = raw_data
            train_target[index:index + 10] = int(g)
            index += 10

index = 0

joblib.dump({'radar_data': train_data_raw,
             'target': train_target},
            os.path.join(data_path, 'train_data.p'))


print('=====preprocess train raw radar data=====')
for x in tqdm(test_participants):
    for g in gestures:
        with h5py.File(os.path.join(data_path, x, g, "RawData.mat"), 'r') as f:
            # 读取指定数据集，例如读取名为data的数据集
            raw_data = f['rawdata'][:]
            # 将数据集转换为numpy数组
            raw_data = np.array(raw_data)
            raw_data = raw_data.view('complex')
            raw_data = raw_data.reshape((10, 100, 128, 100, 2))
            raw_data = raw_data.transpose((0, 1, 2, 4, 3))

            test_data_raw[index:index + 10] = raw_data
            test_target[index:index + 10] = int(g)
            index += 10



joblib.dump({'radar_data': test_data_raw,
             'target': test_target},
            os.path.join(data_path, 'test_data.p'))
