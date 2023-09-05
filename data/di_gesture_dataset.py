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
from utils import *

# import cupy as cp

gestures = ['y_Clockwise', 'y_Counterclockwise', 'y_Pull', 'y_Push', 'y_SlideLeft', 'y_SlideRight']
negative_samples = ['n_liftleft', 'n_liftright', 'n_sit', 'n_stand', 'n_turn', 'n_waving', 'n_walking']
envs = ['e1', 'e2', 'e3', 'e4', 'e5', 'e6']
participants = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8', 'u9', 'u10', 'u11', 'u12', 'u13',
                'u14', 'u15', 'u16', 'u17', 'u18', 'u19', 'u20', 'u21', 'u22', 'u23', 'u24', 'u25']
locations = ['p1', 'p2', 'p3', 'p4', 'p5']

# root = 'D:\\data\\mmWave_cross_domain_gesture_dataset'
# root = '/root/autodl-nas/mmWave_cross_domain_gesture_dataset'
root = '/root/autodl-tmp/dataset/mmWave_cross_domain_gesture_dataset'

st = {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}
st_act = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}

train_data_filenames = []
train_label_list = []
test_data_filenames = []
test_label_list = []
file_cache = {}


filenames_set = set(os.listdir(root))
print('========加载文件缓存========')

# for fn in tqdm(filenames_set):
#     if fn.endswith('.npy'):
#         file_cache[fn] = np.load(os.path.join(root, fn))

print('========文件缓存加载完毕========')


# file_example n_liftleft_e1_u1_p1_s1.npy
def check_and_get(prefix):
    index = 1
    prefix = prefix + '_s{:d}.npy'
    file_name = prefix.format(index)
    samples = []
    while file_name in filenames_set:
        samples.append(file_name)
        index += 1
        file_name = prefix.format(index)
    return samples





def assign_samples_k_fold(ges_type, samples):
    if len(samples) <= 0:
        return
    random.shuffle(samples)
    size = len(samples) // 5
    chunks = [samples[i:i + size] for i in range(0, len(samples), size)]
    for i, value in enumerate(chunks):
        st_act[ges_type] += len(value)
        train_data_filenames[i].extend(value)
        train_label_list[i].extend([ges_type] * size)


def get_samples(file_format, act_type, act, env, participant):
    prefix = file_format.format(act=act, env=env, user=participant)
    samples = []
    sample_labels = []
    if act == 'n_walking':
        sub_samples = check_and_get(prefix)
        samples.extend(sub_samples)
        sample_labels.extend([act_type] * len(sub_samples))
    else:
        for p in locations:
            sub_samples = check_and_get(prefix + '_' + p)
            samples.extend(sub_samples)
            sample_labels.extend([act_type] * len(sub_samples))
    return samples, sample_labels


angle_scale_map = {
    0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31],
    1: [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30],
    2: [0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
        30],
    3: [0, 0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        29],
    4: [0, 0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27,
        29],
    5: [0, 0, 0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26,
        29],
    6: [0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 28],
    7: [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 28],
    8: [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 25, 28],
    9: [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 27],
    10: [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 23, 27],
    11: [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 23, 27],
    12: [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 23, 27],
    13: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 19, 22, 27],
    14: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 18, 21, 26],
    15: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 17, 21, 26],
    16: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 12, 14, 17, 21, 26],
    17: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 7, 8, 9, 11, 14, 17, 21, 26],
    18: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 7, 8, 10, 13, 17, 21, 26],
    19: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 7, 9, 12, 16, 21, 26],
    20: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 8, 11, 15, 20, 26],
}

# 距离平移范围
angle_range = {
    locations[0]: np.arange(-10, 11),
    locations[1]: np.arange(-10, 11),
    locations[2]: np.arange(-10, 11),
    locations[3]: np.arange(-4, 17),
    locations[4]: np.arange(-16, 5),
    'p6': np.arange(-10, 11),
}

distance_range = {
    locations[0]: np.arange(-4, 16),
    locations[1]: np.arange(-8, 9),
    locations[2]: np.arange(-15, 5),
    locations[3]: np.arange(-8, 9),
    locations[4]: np.arange(-8, 9),
    'p6': np.arange(-8, 9),
}

static_angle_range = [np.arange(-20, 21), np.arange(-6, 7)]
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


gesture_pairs = {
    0: 1,
    1: 0,
    2: 3,
    3: 2,
    4: 5,
    5: 4,
    6: 6
}


def random_reverse(datas, labels, p=0.5):
    cond = random.uniform(0, 1)
    if cond > p:
        datas = np.flip(datas, axis=0)
        labels = gesture_pairs[labels]
    return datas, labels


def cfar(datas):
    for i, frame in enumerate(datas):
        datas[i] = ca_cfar_2d(frame, thr_factor=threshold_factor)
    return datas

def data_augmentation(d, label, position):
    # d = data_normalization(d)
    d, dis, angle = random_translation(d, position)
    d = random_geometric_features(d)
    d, label = random_reverse(d, label)
    d = random_data_len_adjust(d)
    d = random_scale_radiated_power(d, position, dis, angle)
    # d = cfar(d)
    return d, label


def get_track_binary(datas):
    # 获取轨迹
    shape = datas.shape
    datas = datas.reshape(-1, shape[-1] * shape[-2])
    max_indexes = np.argmax(datas, axis=1)
    # max_indexes = np.where(datas - 0.99 * np.max(datas, axis=1)[:, None] > 0)
    x, y = max_indexes // shape[-1], max_indexes % shape[-1]
    # x, y = max_indexes[1] // shape[-1], max_indexes[1] % shape[-1]
    track = np.zeros((shape[-2], shape[-1]))
    track[x, y] = 255
    # shape = datas.shape
    # track = np.sum(datas, axis=0)
    track = data_normalization(track)
    return track


def get_track(datas):
    # d_range = datas.sum(axis=2)
    # x = np.argmax(d_range, axis=1)
    track = np.zeros((3, datas.shape[-2], datas.shape[-1]))
    #for i, frame in enumerate(datas):
    #    y = np.argmax(frame[x[i]])
    #    track[0, x[i], y] = frame[x[i], y]
    # track[0] = np.max(datas,axis=0)
    # track[1] = np.mean(datas, axis=0)
    # track[2] = np.std(datas, axis=0)
    #track[0] = data_normalization(track[0])
    return track


def combine(index):
    return di_gesture_dataset(sum([x for i, x in enumerate(train_data_filenames) if i != index], []),
                              sum([x for i, x in enumerate(train_label_list) if i != index], []), data_augmentation), \
           di_gesture_dataset(train_data_filenames[index], train_label_list[index], data_normalization)


pre_domain = ['sos']

def clear_cache():
    train_data_filenames.clear()
    train_label_list.clear()
    test_data_filenames.clear()
    test_label_list.clear()

def split_data(domain, fold=0, env_index=0, position_index=0):
    file_format = '{act}_{env}_{user}'

    if domain == 'in_domain':
        if len(train_data_filenames) == 0:
            train_data_filenames.extend([[], [], [], [], []])
            train_label_list.extend([[], [], [], [], []])
            for i, act in enumerate(itertools.chain(gestures, negative_samples)):
                for e in envs:
                    for u in participants:
                        prefix = file_format.format(act=act, env=e, user=u)
                        if act == 'n_walking':
                            samples = check_and_get(prefix)
                            assign_samples_k_fold(i, samples)
                        else:
                            for p in locations:
                                samples = check_and_get(prefix + '_' + p)
                                assign_samples_k_fold(i, samples)
        tr, te = combine(fold)
        print('ges and num {}'.format(st_act))
        return tr, te
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
        return di_gesture_dataset(train_data_filenames, train_label_list, data_augmentation), \
               di_gesture_dataset(test_data_filenames, test_label_list, data_normalization)
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
                if act == 'n_walking':
                    continue
                for e in envs:
                    for u in participants:
                        for index, p in enumerate(locations):
                            prefix = file_format.format(act=act, env=e, user=u, position=p)
                            samples = check_and_get(prefix)
                            train_data_filenames[index].extend(samples)
                            train_label_list[index].extend([i] * len(samples))
        return combine(position_index)


class di_gesture_dataset(Dataset):
    def __init__(self, file_names, labels, transform=None, max_frames=128):
        self.len = len(file_names)
        self.max_frame = max_frames
        self.file_names = np.array(file_names)

        self.labels = np.array(labels)
        self.labels = np.where(self.labels < 6, labels, 6)
        self.transform = transform
        self.positions = []
        for i, name in enumerate(self.file_names):
            inf = name.split('_')
            if inf[4] in locations:
                self.positions.append(inf[4])
            else:
                self.positions.append('p6')

    def __getitem__(self, index):
        d = np.load(os.path.join(root, self.file_names[index]))
        # d = file_cache[self.file_names[index]]
        label = self.labels[index]
        res = self.transform(d, self.labels[index], self.positions[index])
        if isinstance(res, tuple):
            d = res[0]
            label = res[1]
        else:
            d = res
        track = get_track(d)
        return torch.from_numpy(d).type(torch.float32), torch.from_numpy(track).type(torch.float32), torch.tensor(
            label), index

    def __len__(self):
        return self.len


if __name__ == '__main__':
    train_datas, test_datas = split_data('cross_environment', env_index=3)
    visdom = visdom.Visdom(env='raw-data', port=6006)
    max_frame = 0
    position_set = {}
    label_set = {}
    for it in tqdm(range(10)):
        # for it in tqdm(range(train_datas.__len__())):
        data, track, label, _ = train_datas.__getitem__(it)

        # pos = test_datas.positions[it]
        # l = label.item()
        # if (pos, l) not in position_set:
        #     position_set[(pos, l)] = 1
        # else:
        #     position_set[(pos, l)] += 1
        # if position_set[(pos, l)] < 2:
        #     visdom.heatmap(data[3], win=pos + str(position_set[(pos, l)]) + str(l),
        #                    opts=dict(title='RAM position = ' + pos + 'label = ' + str(l)))
        max_frame = max(max_frame, data.size()[0])
    # for it in tqdm(range(100)):
    permutation = np.random.permutation(test_datas.__len__())
    for it in tqdm(permutation):
        data, track, label, index = test_datas.__getitem__(it)
        visdom.heatmap(track[0], win=str(label.item()),
                           opts=dict(title='label = ' + test_datas.file_names[index]))

        max_frame = max(max_frame, data.size()[0])

    print("max_frame{}".format(max_frame))
