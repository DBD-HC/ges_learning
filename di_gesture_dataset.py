import itertools
import math
import random

import torch
from torch.utils.data import Dataset
from itertools import chain
import os
from tqdm import tqdm
import numpy as np
import joblib

gestures = ['y_Clockwise', 'y_Counterclockwise', 'y_Pull', 'y_Push', 'y_SlideLeft', 'y_SlideRight']
negative_samples = ['n_liftleft', 'n_liftright', 'n_sit', 'n_stand', 'n_turn', 'n_walking', 'n_waving']
envs = ['e1', 'e2', 'e3', 'e4', 'e5', 'e6']
participants = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8', 'u9', 'u10', 'u11', 'u12', 'u13',
                'u14', 'u15', 'u16', 'u17', 'u18', 'u19', 'u20', 'u21', 'u22', 'u23', 'u24', 'u25']
locations = ['p1', 'p2', 'p3', 'p4', 'p5']

# root = 'D:\\dataset\\mmWave_cross_domain_gesture_dataset'
root = '/root/autodl-nas/mmWave_cross_domain_gesture_dataset'

st = {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}


# file_example n_liftleft_e1_u1_p1_s1.npy
def check_and_get(prefix):
    index = 1
    prefix = prefix + '_s{:d}.npy'
    file_name = prefix.format(index)
    samples = []
    while os.path.exists(os.path.join(root, file_name)):
        samples.append(file_name)
        index += 1
        file_name = prefix.format(index)
    return samples


train_data_filenames = []
train_label_list = []
test_data_filenames = []
test_label_list = []


def assign_samples_k_fold(ges_type, samples):
    if len(samples) <= 0:
        return
    random.shuffle(samples)
    size = len(samples) // 5
    chunks = [samples[i:i + size] for i in range(0, len(samples), size)]
    for i, value in enumerate(chunks):
        train_data_filenames[i].extend(value)
        train_label_list[i].extend([ges_type] * size)


def get_samples(file_format, act_type, act, env, participant):
    prefix = file_format.format(act=act, env=env, user=participant)
    samples = []
    sample_labels = []
    if act == 'walking':
        sub_samples = check_and_get(prefix)
        samples.extend(sub_samples)
        sample_labels.extend([act_type] * len(sub_samples))
    else:
        for p in locations:
            sub_samples = check_and_get(prefix + '_' + p)
            samples.extend(sub_samples)
            sample_labels.extend([act_type] * len(sub_samples))
    return samples, sample_labels


def combine(index):
    return di_gesture_dataset(sum([x for i, x in enumerate(train_data_filenames) if i != index], []),
                              sum([x for i, x in enumerate(train_label_list) if i != index], [])), \
           di_gesture_dataset(train_data_filenames[index], train_label_list[index])


def split_data(domain, fold=0, env_index=0, position_index=0):
    file_format = '{act}_{env}_{user}'
    if domain == 'in_domain':
        # if os.path.exists('train_data_filenames.npy'):
        #     train_data_filenames.extend(np.load('train_data_filenames.npy'))
        #     train_label_list.extend(np.load('train_label_list.npy'))
        if len(train_data_filenames) == 0:
            train_data_filenames.extend([[], [], [], [], []])
            train_label_list.extend([[], [], [], [], []])
            for i, act in enumerate(itertools.chain(gestures, negative_samples)):
                for e in envs:
                    for u in participants:
                        prefix = file_format.format(act=act, env=e, user=u)
                        if act == 'walking':
                            samples = check_and_get(prefix)
                            assign_samples_k_fold(i, samples)
                        else:
                            for p in locations:
                                samples = check_and_get(prefix + '_' + p)
                                assign_samples_k_fold(i, samples)
            # np.save('train_data_filenames.npy', np.array(train_data_filenames))
            # np.save('train_label_list.npy', np.array(train_label_list))
        tr, te =  combine(fold)
        return  tr,te
    elif domain == 'cross_person':
        if len(train_data_filenames) == 0:
            for i, act in enumerate(itertools.chain(gestures, negative_samples)):
                for e in envs:
                    for u in participants[:7]:
                        samples, labels = get_samples(file_format, i, act, e, u)
                        train_data_filenames.extend(samples)
                        train_label_list.extend(labels)
                    for u in participants[7:]:
                        samples, labels = get_samples(file_format, i, act, e, u)
                        test_data_filenames.extend(samples)
                        test_label_list.extend(labels)
        return di_gesture_dataset(train_data_filenames, train_label_list), \
               di_gesture_dataset(test_data_filenames, test_label_list)
    elif domain == 'cross_environment':
        if len(train_data_filenames) == 0:
            train_data_filenames.extend([[], [], [], [], [], []])
            train_label_list.extend([[], [], [], [], [], []])
            for i, act in enumerate(itertools.chain(gestures, negative_samples)):
                for u in participants:
                    for index, e in enumerate(envs):
                        samples, labels = get_samples(file_format, i, act, e, u)
                        train_data_filenames[index].extend(samples)
                        train_label_list[index].extend(labels)
        return combine(env_index)
    else:
        if len(train_data_filenames) == 0:
            train_data_filenames.extend([[], [], [], [], []])
            train_label_list.extend([[], [], [], [], []])
            file_format = file_format + '_{position}'
            for i, act in enumerate(itertools.chain(gestures, negative_samples)):
                if act == 'walking':
                    continue
                for index, e in enumerate(envs):
                    for u in participants:
                        for p in locations:
                            prefix = file_format.format(act=act, env=e, user=u, position=p)
                            samples = check_and_get(prefix)
                            train_data_filenames[index].extend(samples)
                            train_label_list[index].extend([i] * len(samples))
        return combine(position_index)


class di_gesture_dataset(Dataset):
    def __init__(self, file_names, labels, max_frames=128):
        self.len = len(file_names)
        self.max_frame = max_frames
        self.file_names = np.array(file_names)
        self.datas = np.zeros((self.len, self.max_frame, 32, 32))
        permutation = np.random.permutation(self.len)
        self.file_names = self.file_names[permutation]
        self.labels = np.array(labels)
        self.labels = self.labels[permutation]
        # print('=====load dataset======')
        # for i in tqdm(permutation):
        #    d = np.load(os.path.join(root, file_names[i]))
        #    self.datas[i, :len(d)] = d

    def __getitem__(self, index):
        # DRAI = np.zeros((self.max_frame, 32, 32))
        d = np.load(os.path.join(root, self.file_names[index]))
        # DRAI[:len(d)] = d
        return torch.from_numpy(d), torch.tensor(self.labels[index])

    def __len__(self):
        return self.len


if __name__ == '__main__':
    train_datas, test_datas = split_data('in_domain', 1)
    max_frame = 0
    for it in tqdm(range(train_datas.__len__())):
        data, label = train_datas.__getitem__(it)
        max_frame = max(max_frame, data.size()[0])
    for it in tqdm(range(test_datas.__len__())):
        data, label = test_datas.__getitem__(it)
        max_frame = max(max_frame, data.size()[0])

    print("max_frame{}".format(max_frame))
