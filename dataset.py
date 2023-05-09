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

data_path = '/root/autodl-nas/'
rdai_path = '/root/autodl-fs/air-writing'
rdai_file_format = 'rdai_{user}_{ges}_{s}.npy'

samples_per_participant = 100
samples_per_gesture = 10

visdom = visdom.Visdom(env='air-writing', port=6006)



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


# Angle vector
angle_vector = np.arange(100) / 100 * np.pi - np.pi / 2
angle_vector = np.sin(angle_vector).reshape((-1, 1)) @ np.arange(2).reshape((1, -1))
angle_vector = np.exp(2j * np.pi * 0.5 * angle_vector)
alpha_spacial_cov = 0.01


def capon(range_doppler):
    # Generate range-azimuth heatmap using CAPON
    RA = np.empty((100, 100), dtype=complex)
    for range_idx in range(100):
        x = range_doppler[:, :, range_idx]
        r = x @ x.conj().T / 128
        r += alpha_spacial_cov * np.trace(r) / 2 * np.eye(2)
        r = np.linalg.inv(r)
        for angle_idx in range(100):
            a = angle_vector[angle_idx].reshape((-1, 1))
            RA[range_idx, angle_idx] = 1 / (a.conj().T @ r @ a)[0, 0]
    return RA


static_remove_thr = 2
scale_factor = 0.8


def get_RDAI_2():
    num_pat = len(participants)
    print('==========开始解析数据==========')
    if not os.path.exists(rdai_path):
        os.mkdir(rdai_path)
    for p_index in tqdm(range(6, num_pat)):
        for ges in range(10):
            raw_datas = get_data(p_index, ges)

            raw_datas = raw_datas.transpose((0, 1, 3, 2, 4))


            for s_index, sample in enumerate(raw_datas):
                print('person:{} ges:{} sample:{}'.format(participants[p_index], gestures[ges], str(s_index)))
                # (100, 2, 128, 100)
                ## range fft
                range_fft_raw = np.fft.fft(sample, axis=3)
                range_fft_raw = range_fft_raw - range_fft_raw.mean(axis=2).reshape((-1, 2, 1, 100))
                # (100, 2, 100)
                mean_amp = np.abs(range_fft_raw).mean(axis=2).sum(axis=1).mean(axis=1)
                # (100, 2)
                mean_amp = (mean_amp- mean_amp.min())/(mean_amp.max() - mean_amp.min())
                first_idx = next((i for i, x in enumerate(mean_amp) if x > 0.3), None)
                last_idx = next((i for i, x in enumerate(mean_amp[::-1]) if x > 0.3), None)
                if last_idx is not None:
                    last_idx = len(range_fft_raw) - 1 - last_idx
                RDAI = np.zeros((100, 50, 50))
                RDI = np.zeros((128, 100))
                for f_i ,frame in enumerate(range_fft_raw):
                    if not (f_i < first_idx or f_i > last_idx):
                        range_doppler_fft = np.fft.fft(frame, axis=2)
                        range_doppler_fft = np.abs(range_doppler_fft).sum(axis=2).mean(axis=0)
                        RDI[:, f_i] = range_doppler_fft
                visdom.heatmap(RDI, win=str(s_index) + '_range doppler',
                               opts=dict(title=str(s_index) + 'range doppler'))
                for f_i ,frame in enumerate(range_fft_raw):
                    if not (f_i < first_idx or f_i > last_idx):
                        try:
                            RA = capon(frame)
                        except BaseException as e:
                            print(e)
                        visdom.heatmap(np.abs(RA), win=str(s_index) + '_range angle',
                                       opts=dict(title=str(s_index) + 'range angle'))
                        RDAI[f_i] = abs(RA[0:50, 25:75])
                file_name = rdai_file_format.format(user=participants[p_index], ges=gestures[ges], s=str(s_index))
                RDAI = RDAI[first_idx: last_idx]
                np.save(os.path.join(rdai_path, file_name), RDAI)




def get_RDAI():
    num_pat = len(participants)
    print('==========开始解析数据==========')
    if not os.path.exists(rdai_path):
        os.mkdir(rdai_path)
    for p_index in tqdm(range(6, num_pat)):
        for ges in range(10):
            raw_datas = get_data(p_index, ges)
            raw_datas = raw_datas.transpose((0, 1, 3, 2, 4))
            for s_index, sample in enumerate(raw_datas):
                print('person:{} ges:{} sample:{}'.format(participants[p_index], gestures[ges], str(s_index)))
                range_doppler = np.zeros((2, 128, 100))
                mean_amp = np.zeros((100, 2, 128, 100))
                rdai = np.zeros((100, 50, 50))
                for frame_id, frame in enumerate(sample):
                    range_fft_raw = np.fft.fft(frame, axis=2)
                    range_fft_raw = range_fft_raw - range_fft_raw.mean(axis=1).reshape((2, 1, 100))
                    visdom.heatmap(np.abs(range_fft_raw[0]), win=str(frame_id) + '_range doppler', opts=dict(title=str(frame_id) + 'range doppler'))
                    for angle in range(2):
                        for r_index in range(100):
                            rd_fft_raw = np.fft.fft(range_fft_raw[angle, :, r_index])
                            rd_fft_raw = np.fft.fftshift(rd_fft_raw)
                            rd_fft_raw[64 - static_remove_thr:64 + static_remove_thr] = 0
                            range_doppler[angle, :, r_index] = rd_fft_raw[:]
                    try:
                        ra = capon(range_doppler)
                        rdai[frame_id] = np.abs(ra[0:50, 25:75])
                    except BaseException as e:
                        print(e)
                    # visdom.heatmap(np.abs(ra[0:50, 25:75]), win=str(p_index) + '_' + str('range fft'),
                    #                opts=dict(title='range doppler fft'))
                file_name = rdai_file_format.format(user=participants[p_index], ges=gestures[ges], s=str(s_index))
                np.save(os.path.join(rdai_path, file_name), rdai)


static_angle_range = [np.arange(-5, 6), np.arange(-1, 2)]
static_distance_range = [np.arange(-5, 6), np.arange(-2, 3)]


def random_translation(datas, p=0.5):
    d_distance = random.choice(static_distance_range[0])
    d_angle = random.choice(static_angle_range[0])
    simple_shift(datas, d_distance, d_angle)
    return datas, d_distance, d_angle


def random_reverse(datas, labels, p=0.5):
    cond = random.uniform(0, 1)
    if cond > p:
        datas = np.flip(datas, axis=0)
    return datas, labels


def random_scale_radiated_power(datas):
    scale_factor = random.uniform(0.8, 1.2)
    datas = scale_factor * data_normalization(datas)
    return datas


adjust_gap = np.arange(4, 6)


def data_augmentation(d, label):
    d, dis, angle = random_translation(d)
    d, label = random_reverse(d, label)
    d = random_geometric_features(d)
    d = random_scale_radiated_power(d)
    d = random_data_len_adjust_2(d)
    return d, label


file_cache = {}


def split_RDAI(p=0):
    if len(train_data) == 0:
        filenames_set = set(os.listdir(rdai_path))
        for user in participants:
            filenames = []
            labels = []
            for i, ges in enumerate(gestures):
                count = 0
                file_name = rdai_file_format.format(user=user, ges=ges, s=count)
                while file_name in filenames_set:
                    filenames.append(os.path.join(rdai_path, file_name))
                    labels.append(i)
                    count += 1
                    file_name = rdai_file_format.format(user=user, ges=ges, s=count)
            train_data.append(filenames)
            train_label.append(labels)
    if len(file_cache) == 0:
        print('=======初始化数据集缓存====')
        for names in train_data:
            for name in tqdm(names):
                d = np.load(name)
                file_cache[name] = d
        print('=======数据集缓存初始化完毕====')

    data_for_train = []
    label_for_train = []
    data_for_test = []
    label_for_test = []

    for i in range(len(participants)):
        if i != p:
            data_for_train.extend(train_data[i])
            label_for_train.extend(train_label[i])
        else:
            data_for_test.extend(train_data[i])
            label_for_test.extend(train_label[i])

    return Gesture_Dataset(data_for_train, label_for_train, transform=data_augmentation), \
           Gesture_Dataset(data_for_test, label_for_test, transform=data_normalization)


class Gesture_Dataset(Dataset):
    def __init__(self, filenames, labels, transform=None):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform
        self.len = len(self.filenames)

    def __getitem__(self, index):
        data = file_cache[self.filenames[index]]
        label = self.labels[index]
        res = self.transform(data, label)
        if isinstance(res, tuple):
            data = res[0]
            label = res[1]
        else:
            data = res
        data = np.sum(data, axis=0)
        return torch.from_numpy(data).type(torch.float32), torch.tensor(label)

    def __len__(self):
        return self.len


if __name__ == '__main__':
    get_RDAI_2()
    train_datas, test_datas = split_RDAI()
    for it in tqdm(range(30)):
        d, l = train_datas.__getitem__(it)
        map = torch.sum(d, dim=0)
        visdom.heatmap(map, win=str(l) + '_range angel', opts=dict(title=str(l) + 'range angel'))
        # for fi, frame in enumerate(d):
        #     visdom.heatmap(frame, win=str(fi % 20) + '_range angel', opts = dict(title=str(fi) + 'range angel fft'))
