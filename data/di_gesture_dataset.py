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

from data.data_splitter import DataSpliter
from utils import *

# import cupy as cp



# root = 'D:\\data\\mmWave_cross_domain_gesture_dataset'
# root = '/root/autodl-nas/mmWave_cross_domain_gesture_dataset'
rai_data_root = '/root/autodl-tmp/dataset/mmWave_cross_domain_gesture_dataset'
rdi_data_root = '/root/autodl-tmp/dataset/mmWave_cd_rdi/mmwave_rdi'
# di_data_root = '/root/autodl-tmp/dataset/mmwave_di'
# root = '/root/autodl-tmp/dataset/mmWave_cd_rdi/mmwave_rdi'
# root = '/root/autodl-tmp/dataset/mmwave_rai_lessrx'
# root = '/root/autodl-tmp/dataset/mmwave_rai_2rx'

st = {'e1': 0, 'e2': 0, 'e3': 0, 'e4': 0, 'e5': 0, 'e6': 0}
st_act = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}

print('========加载文件缓存========')

# for fn in tqdm(filenames_set):
#     if fn.endswith('.npy'):
#         file_cache[fn] = np.load(os.path.join(root, fn))

print('========文件缓存加载完毕========')


# file_example n_liftleft_e1_u1_p1_s1.npy


static_angle_range = [np.arange(-6, 7), np.arange(-6, 7)]
static_distance_range = [np.arange(-6, 7), np.arange(-6, 7)]


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
    d = random_data_len_adjust_2(d)
    # d = random_scale_radiated_power(d, position, dis, angle)
    d = data_normalization(d)
    # d = cfar(d)
    return d, label


pre_domain = [0]




class DIDataSplitter(DataSpliter):
    def __init__(self, data_path = rai_data_root):

        self.gestures = ['y_Clockwise', 'y_Counterclockwise', 'y_Pull', 'y_Push', 'y_SlideLeft', 'y_SlideRight']
        self.negative_samples = ['n_liftleft', 'n_liftright', 'n_sit', 'n_stand', 'n_turn', 'n_waving', 'n_walking']
        self.acts = self.gestures + self.negative_samples
        self.envs = ['e1', 'e2', 'e3', 'e4', 'e5', 'e6']
        self.participants = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8', 'u9', 'u10', 'u11', 'u12', 'u13',
                        'u14', 'u15', 'u16', 'u17', 'u18', 'u19', 'u20', 'u21', 'u22', 'u23', 'u24', 'u25']
        self.locations = ['p1', 'p2', 'p3', 'p4', 'p5']

        super(DIDataSplitter, self).__init__(data_path=data_path, domain_num=(5, 5, len(self.envs), len(self.locations)))
        self.file_format = '{act}_{env}_{user}_{loc}_s{s}.npy'
        self.file_format_walking = '{act}_{env}_{user}_s{s}.npy'
        self.pre_domain = -1

    def get_domain_num(self, domain):
        return self.domain_num[domain]

    def clear_cache(self):
        self.train_data_filenames.clear()
        self.train_label_list.clear()
        self.test_data_filenames.clear()
        self.test_label_list.clear()

    def combine(self, index):
        return di_gesture_dataset(sum([x for i, x in enumerate(self.train_data_filenames) if i != index], []),
                                  sum([x for i, x in enumerate(self.train_label_list) if i != index], []),
                                  data_augmentation), \
               di_gesture_dataset(self.train_data_filenames[index], self.train_label_list[index], data_normalization)

    def assign_k_fold(self, samples, sample_labels, fold):
        s_len = len(samples)
        if s_len <= 0:
            return
        sample_label_zip = list(zip(samples, sample_labels))
        random.shuffle(sample_label_zip)
        samples, sample_labels = zip(*sample_label_zip)
        size = s_len // fold
        j = 0
        for i in range(0, s_len, size):
            if j == len(self.train_data_filenames) - 1:
                self.train_data_filenames[j].extend(samples[i:])
                self.train_label_list[j].extend(sample_labels[i:])
                break
            else:
                self.train_data_filenames[j].extend(samples[i:i + size])
                self.train_label_list[j].extend(sample_labels[i:i + size])
            j += 1

    def check_and_get(self, file_name_prefix, sample_label):
        sample_index = 1
        file_name = file_name_prefix.format(sample_index)
        samples = []
        sample_labels = []
        while file_name in self.filenames_set:
            samples.append(file_name)
            sample_labels.append(sample_label)
            sample_index += 1
            file_name = file_name_prefix.format(sample_index)
        return samples, sample_labels

    def get_samples(self, act, act_i, env, user):
        # {act}_{env}_{user}_{loc}_s{s}.npy
        if act == self.negative_samples[-1]:
            file_name_prefix = self.file_format_walking.format(act=act, env=env, user=user, s='{}')
            samples, sample_labels = self.check_and_get(file_name_prefix, act_i)
        else:
            samples = []
            sample_labels = []
            for loc_i, loc in enumerate(self.locations):
                file_name_prefix = self.file_format.format(act=act, env=env, user=user, loc=loc, s='{}')
                temp_samples, temp_sample_labels = self.check_and_get(file_name_prefix, act_i)
                samples.extend(temp_samples)
                sample_labels.extend(temp_sample_labels)
        return samples, sample_labels

    def split_data(self, domain, index=None):
        if self.pre_domain != domain:
            self.clear_cache()
            self.pre_domain = domain

        # in_domain
        if domain == 0:
            if len(self.train_data_filenames) == 0:
                self.train_data_filenames.extend([[], [], [], [], []])
                self.train_label_list.extend([[], [], [], [], []])
                for i, act in enumerate(self.acts):
                    for e in self.envs:
                        for u in self.participants:
                            if act == self.negative_samples[-1]:
                                file_name_prefix = self.file_format_walking.format(act=act, env=e, user=u, s='{}')
                                samples, sample_labels = self.check_and_get(file_name_prefix, i)
                                self.assign_k_fold(samples, sample_labels, self.domain_num[domain])
                            else:
                                for p in self.locations:
                                    file_name_prefix = self.file_format.format(act=act, env=e, user=u, loc=p, s='{}')
                                    samples, sample_labels = self.check_and_get(file_name_prefix, i)
                                    self.assign_k_fold(samples, sample_labels, self.domain_num[domain])
            tr, te = self.combine(index)
            print('ges and num {}'.format(st_act))
            return tr, te
        # cross_person
        elif domain == 1:
            if len(self.train_data_filenames) == 0:
                for i, act in enumerate(self.acts):
                    for e in self.envs:
                        for u_i, u in enumerate(self.participants):
                            samples, labels = self.get_samples(act=act, act_i=i, env=e, user=u)
                            if u_i < 3:
                                self.train_data_filenames.extend(samples)
                                self.train_label_list.extend(labels)
                            else:
                                self.test_data_filenames.extend(samples)
                                self.test_label_list.extend(labels)
            return di_gesture_dataset(self.train_data_filenames, self.train_label_list, data_augmentation), \
                   di_gesture_dataset(self.test_data_filenames, self.test_label_list, data_normalization)
        # cross_environment
        elif domain == 2:
            if len(self.train_data_filenames) == 0:
                self.train_data_filenames.extend([[], [], [], [], [], []])
                self.train_label_list.extend([[], [], [], [], [], []])
                for act_i, act in enumerate(itertools.chain(self.gestures, self.negative_samples)):
                    for u in self.participants:
                        for e_i, e in enumerate(self.envs):
                            samples, labels = self.get_samples(act, act_i, e, u)
                            self.train_data_filenames[e_i].extend(samples)
                            self.train_label_list[e_i].extend(labels)
            return self.combine(index)
        # cross location
        else:
            if len(self.train_data_filenames) == 0:
                self.train_data_filenames.extend([[], [], [], [], []])
                self.train_label_list.extend([[], [], [], [], []])
                for i, act in enumerate(self.acts):
                    if act == self.negative_samples[-1]:
                        continue
                    for e in self.envs:
                        for u in self.participants:
                            for p_i, p in enumerate(self.locations):
                                file_name_prefix = self.file_format.format(act=act, env=e, user=u, loc=p, s='{}')
                                samples, labels = self.check_and_get(file_name_prefix, i)
                                self.train_data_filenames[p_i].extend(samples)
                                self.train_label_list[p_i].extend(labels)
            return self.combine(index)

'''
def split_data(domain, fold=0, env_index=0, position_index=0, person_index=7):
    file_format = '{act}_{env}_{user}'
    if pre_domain[0] != domain:
        clear_cache()
        pre_domain[0] = domain
    # in_domain
    if domain == 0:
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
    # cross_person
    elif domain == 1:
        if len(train_data_filenames) == 0:
            for i, act in enumerate(itertools.chain(gestures, negative_samples)):
                for e in envs:
                    for u in participants[:person_index]:
                        samples, labels = get_samples(file_format, i, act, e, u)
                        train_data_filenames.extend(samples)
                        train_label_list.extend(labels)
                    for u in participants[person_index:]:
                        samples, labels = get_samples(file_format, i, act, e, u)
                        test_data_filenames.extend(samples)
                        test_label_list.extend(labels)
        return di_gesture_dataset(train_data_filenames, train_label_list, data_augmentation), \
               di_gesture_dataset(test_data_filenames, test_label_list, data_normalization)
    # cross_environment
    elif domain == 2:
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
    # cross location
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
'''



def get_track(rai):
    rai_max = np.max(rai, axis=0)
    rai_mean = np.mean(rai, axis=0)
    rai_std = np.std(rai, axis=0)
    global_track = np.concatenate((rai_max[None, :], rai_mean[None, :], rai_std[None, :]), axis=0)
    return global_track

def compress_rdi(d):
    if d.ndim > 3:
        d = np.sqrt(d[0::2] ** 2 + d[1::2] ** 2)
        d = np.mean(d, axis=0)
    return d,

class di_gesture_dataset(Dataset):
    def __init__(self, file_names, labels, transform=None, max_frames=128, need_augmentation=True, data_root=None, data_type=0):
        self.len = len(file_names)
        self.max_frame = max_frames
        self.file_names = np.array(file_names)

        self.labels = np.array(labels)
        self.labels = np.where(self.labels < 6, labels, 6)
        if data_root is not None:
            self.data_root = data_root
        elif data_type == 0:
            self.data_root = rai_data_root
        else:
            self.data_root = rdi_data_root
        if need_augmentation:
            self.transform = transform
        else:
            self.transform = data_normalization

    def __getitem__(self, index):
        d = np.load(os.path.join(self.data_root, self.file_names[index]))
        # compress_rdi(d)
        # d = file_cache[self.file_names[index]]
        label = self.labels[index]
        res = self.transform(d, self.labels[index], None)
        if isinstance(res, tuple):
            d = res[0]
            label = res[1]
        else:
            d = res

        return torch.from_numpy(d).type(torch.float32), torch.from_numpy(get_track(d)).type(torch.float32) if d.ndim == 3 else None, torch.tensor(
            label), index

    def __len__(self):
        return self.len


def plot_data_len_stat():
    statis = np.zeros((100, 7))
    ges_index_map = {'Clockwise': 0, 'Counterclockwise': 1, 'Pull': 2, 'Push': 3, 'SlideLeft': 4, 'SlideRight': 5,
                     'NG': 6}
    for file_name in filenames_set:
        inf = file_name.split('_')
        sample_type = inf[0]
        if sample_type == 'n':
            ges_type = 'NG'
        else:
            ges_type = inf[1]
        ges_index = ges_index_map[ges_type]
        sample = np.load(os.path.join(rai_data_root, file_name))
        statis[len(sample), ges_index] += 1
    print(statis)
    visdom.bar(
        X=statis[:, None],
        opts=dict(
            stacked=True,
            ylabel='Number of Samples',
            xlabel='Number of Frames',
            legend=['Clockwise', 'Counterclockwise', 'Pull', 'Push', 'SlideLeft', 'SlideRight', 'NG']
        )
    )


def plot_user_sample_stat():
    statis = np.zeros((25, 7))
    ges_index_map = {'Clockwise': 0, 'Counterclockwise': 1, 'Pull': 2, 'Push': 3, 'SlideLeft': 4, 'SlideRight': 5,
                     'NG': 6}
    for file_name in filenames_set:
        inf = file_name.split('_')
        user_index = int(inf[3][1:]) - 1

        sample_type = inf[0]
        if sample_type == 'n':
            ges_type = 'NG'
        else:
            ges_type = inf[1]
        ges_index = ges_index_map[ges_type]
        statis[user_index, ges_index] += 1
    print(statis)
    visdom.bar(
        X=statis[:, None],
        win='user sample num',
        opts=dict(
            stacked=True,
            ylabel='Number of Samples',
            xlabel='Number of Frames',
            legend=['Clockwise', 'Counterclockwise', 'Pull', 'Push', 'SlideLeft', 'SlideRight', 'NG']
        )
    )


def plot_domain_stat_2():
    statis = np.zeros((6, 7))
    ges_index_map = {'Clockwise': 0, 'Counterclockwise': 1, 'Pull': 2, 'Push': 3, 'SlideLeft': 4, 'SlideRight': 5,
                     'NG': 6}
    for file_name in filenames_set:
        inf = file_name.split('_')
        env_index = int(inf[2][1:]) - 1

        sample_type = inf[0]
        if sample_type == 'n':
            ges_type = 'NG'
        else:
            ges_type = inf[1]
        ges_index = ges_index_map[ges_type]
        statis[env_index, ges_index] += 1
    print(statis)
    visdom.bar(
        X=statis[:, None],
        win='user sample num',
        opts=dict(
            stacked=True,
            ylabel='Number of Samples',
            xlabel='Number of Frames',
            legend=['Clockwise', 'Counterclockwise', 'Pull', 'Push', 'SlideLeft', 'SlideRight', 'NG']
        )
    )


def check():
    rai = sorted(os.listdir('/root/autodl-tmp/dataset/mmWave_cross_domain_gesture_dataset'))
    rdi = sorted(os.listdir('/root/autodl-tmp/dataset/mmwave_di'))
    for i, v in enumerate(rai):
        if v != rdi[i]:
            print(v)


def rename():
    rd_path = '/root/autodl-tmp/dataset/mmwave_di'
    file_format = 'n_walking_e6_u21_s{sample}.npy'

    for i in range(10):
        origin = file_format.format(sample=61 + i)
        new_name = file_format.format(sample=41 + i)
        os.rename(os.path.join(rd_path, origin), os.path.join(rd_path, new_name))


if __name__ == '__main__':
    check()
    visdom = visdom.Visdom(env='statis', port=6006)
    plot_domain_stat_2()

    # drs = domain_reduction_split(('e1', 'e2'))
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
