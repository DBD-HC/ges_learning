import itertools
import os.path
import time

import numpy as np
import torch
import visdom
from torch.utils.data import Dataset
from data.data_splitter import DataSpliter, data_type_map, data_normalization
from utils import *

rai_data_root = '/root/autodl-tmp/dataset/mmWave_cross_domain_gesture_dataset'
complex_rdi_data_root = '/root/autodl-tmp/dataset/mmWave_cd_rdi/complex_rdi/mmwave_rdi'
single_rdi_data_root = '/root/autodl-tmp/dataset/mmWave_cd_rdi/single_rdi'
time_range_angle_data_root = '/root/autodl-tmp/dataset/mmWave_cd_tf'
time_range_doppler_data_root = '/root/autodl-tmp/dataset/mmWave_cd_trd'

def generate_time_range_doppler():
    filenames = os.listdir(single_rdi_data_root)
    for filename in tqdm(filenames):
        time_r = []
        time_d = []
        file = np.load(os.path.join(single_rdi_data_root, filename))
        for f in file:
            time_r.append(np.mean(f, axis=-1))
            time_d.append(np.mean(f, axis=-2))
        time_r = np.array(time_r)[None, :]
        time_d = np.array(time_d)[None, :]
        time_rd = np.concatenate((time_r, time_d), axis=0)
        np.save(os.path.join(time_range_doppler_data_root, filename), time_rd)

static_angle_range = np.arange(-20, 21)
static_distance_range = np.arange(-6, 7)


def random_translation(datas):
    d_distance = random.choice(static_distance_range)
    d_angle = random.choice(static_angle_range)
    simple_shift(datas, d_distance, d_angle)
    return datas

# def cropped_rdi_augmentation(d):


def data_augmentation(d, data_type):
    # rai
    if data_type == data_type_map['RANGE_ANGLE_IMAGE']:
        d = random_translation(d)
        d = random_geometric_features(d)
        d = random_data_len_adjust_2(d)
    elif data_type == data_type_map['CROPPED_RANGE_DOPPLER_IMAGER']:
        #d = crop_complex_rdi(d)
        d = random_rdi_speed(d)
    d = data_normalization(d, data_type)
    return d



pre_domain = [0]


class MCDDataSplitter(DataSpliter):
    def __init__(self, data_path=rai_data_root, is_multi_negative=False):

        self.gestures = ['y_Clockwise', 'y_Counterclockwise', 'y_Pull', 'y_Push', 'y_SlideLeft', 'y_SlideRight']
        self.negative_samples = ['n_liftleft', 'n_liftright', 'n_sit', 'n_stand', 'n_turn', 'n_waving', 'n_walking']
        self.acts = self.gestures + self.negative_samples
        self.envs = ['e1', 'e2', 'e3', 'e4', 'e5', 'e6']
        self.participants = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8', 'u9', 'u10', 'u11', 'u12', 'u13',
                             'u14', 'u15', 'u16', 'u17', 'u18', 'u19', 'u20', 'u21', 'u22', 'u23', 'u24', 'u25']
        self.locations = ['p1', 'p2', 'p3', 'p4', 'p5']
        self.is_multi_negative=is_multi_negative

        super(MCDDataSplitter, self).__init__(data_path=data_path,
                                              domain_num=(5, 25, len(self.envs), len(self.locations),0))
        self.file_format = '{act}_{env}_{user}_{loc}_s{s}.npy'
        self.file_format_walking = '{act}_{env}_{user}_s{s}.npy'
        # [user idx, env idx, loc idx]
        self.od_for_train = [[0, 1, 2, 3, 4], [1] , [2]]
        self.pre_domain = -1

    def get_domain_num(self, domain):
        return self.domain_num[domain]

    def get_class_num(self):
        # Gesture categories along with a negative sample class
        if self.is_multi_negative:
            return  len(self.gestures) + len(self.negative_samples)
        return len(self.gestures) + 1

    def clear_cache(self):
        self.train_data_filenames.clear()
        self.train_label_list.clear()
        self.test_data_filenames.clear()
        self.test_label_list.clear()


    def get_dataset(self, idx=None, data_type=0):
        datas = super().get_dataset(idx)
        return MCDDataset(datas[0], datas[1], data_normalization, data_type=self.data_type)

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

    def split_data(self, domain, train_index=None, val_index=None, test_index=None, need_val=False, need_test=False,
                   need_augmentation=True, is_reduction=False):
        if self.pre_domain != domain:
            self.clear_cache()
            self.pre_domain = domain
        if len(self.train_data_filenames) == 0:
            for i in range(self.get_domain_num(domain)):
                self.train_data_filenames.append([])
                self.train_label_list.append([])

            domain_index = [0, 0, 0, 0]
            # in_domain
            if domain == 0:

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
            elif domain == 2 or domain == 1:

                for act_i, act in enumerate(itertools.chain(self.gestures, self.negative_samples)):
                    for u_i, u in enumerate(self.participants):
                        domain_index[1] = u_i
                        for e_i, e in enumerate(self.envs):
                            domain_index[2] = e_i
                            samples, labels = self.get_samples(act, act_i, e, u)
                            self.train_data_filenames[domain_index[domain]].extend(samples)
                            self.train_label_list[domain_index[domain]].extend(labels)
            elif domain == 3:
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
            else:
                label_mapping = {}
                for act_i, act in enumerate(self.acts):
                    label_mapping[act[2:]] = act_i
                for filename in self.filenames_set:
                    if not filename.endswith('.npy'):
                        continue
                    values = filename.split('_')
                    act = values[1]
                    # Excluding the ‘walking’ samples
                    if act == self.negative_samples[-1][2:]:
                        continue
                    cur_user_idx = int(values[3][1:])
                    cur_env_idx = int(values[2][1:])
                    cur_loc_idx = int(values[4][1:])
                    if cur_user_idx in self.od_for_train[0] \
                        and cur_env_idx in self.od_for_train[1] \
                        and cur_loc_idx in self.od_for_train[-1]:
                        self.train_data_filenames.append(filename)
                        self.train_label_list.append(label_mapping[values[1]])
                    else:
                        self.test_data_filenames.append(filename)
                        self.test_label_list.append(label_mapping[values[1]])

        datas = super().combine(train_index, val_index, test_index, is_reduction, need_val, need_test)
        return MCDDataset(datas[0], datas[1], data_augmentation if need_augmentation else data_normalization, data_type=self.data_type, is_multi_negative=self.is_multi_negative), \
               MCDDataset(datas[2], datas[3], data_normalization, data_type=self.data_type, is_multi_negative=self.is_multi_negative), \
               MCDDataset(datas[4], datas[5], data_normalization, data_type=self.data_type, is_multi_negative=self.is_multi_negative)


def get_track(rai):
    rai_max = np.max(rai, axis=0)
    rai_mean = np.mean(rai, axis=0)
    rai_std = np.std(rai, axis=0)
    global_track = np.concatenate((rai_max[None, :], rai_mean[None, :], rai_std[None, :]), axis=0)
    return global_track


def compress_rdi(d):
    if d.ndim > 3:
        d = np.sqrt(d[:, 0::2] ** 2 + d[:, 1::2] ** 2)
        d = np.mean(d, axis=1)
    return d

def pre_processing(d, data_type):
    if data_type == data_type_map['CROPPED_RANGE_DOPPLER_IMAGER']:
        d = crop_complex_rdi(d)
    return d

class MCDDataset(Dataset):
    def __init__(self, file_names, labels, transform=None, pre_processing=pre_processing, data_root=None, data_type=0, is_multi_negative=False):
        if file_names is None:
            return
        self.len = len(file_names)
        self.file_names = np.array(file_names)
        self.labels = np.array(labels)
        if not is_multi_negative:
            self.labels = np.where(self.labels < 6, labels, 6)
        self.data_type = data_type
        if data_root is not None:
            self.data_root = data_root
        # rai
        elif data_type == data_type_map['RANGE_ANGLE_IMAGE']:
            self.data_root = rai_data_root
        # time_range_angle
        elif data_type == data_type_map['TIME_RANGE_ANGLE_IMAGE']:
            self.data_root = time_range_angle_data_root
        elif data_type == data_type_map['TIME_RANGE_DOPPLER_IMAGE']:
            self.data_root = time_range_doppler_data_root
        # complex rdi
        elif data_type == data_type_map['COMPLEX_RANGE_DOPPLER'] \
                or data_type == data_type_map['CROPPED_RANGE_DOPPLER_IMAGER']:
            self.data_root = complex_rdi_data_root
        # single rdi
        else:
            self.data_root = single_rdi_data_root

        self.transform = transform
        self.pre_processing = pre_processing


    def __getitem__(self, index):
        d = np.load(os.path.join(self.data_root, self.file_names[index]))
        d = self.pre_processing(d, self.data_type)
        label = self.labels[index]
        label = torch.tensor(label)
        d = self.transform(d, self.data_type)
        if self.data_type == data_type_map['RANGE_ANGLE_IMAGE'] or self.data_type == data_type_map['SINGLE_RANGE_DOPPLER']:
            track = torch.from_numpy(get_track(d)).type(torch.float32)
            d = torch.from_numpy(d).type(torch.float32)
            return  d, track, label

        return torch.from_numpy(d).type(torch.float32), label

    def __len__(self):
        return self.len


def check():
    rai = sorted(os.listdir('/root/autodl-tmp/dataset/mmWave_cross_domain_gesture_dataset'))
    rdi = sorted(os.listdir('/root/autodl-tmp/dataset/mmwave_rdi'))
    for i, v in enumerate(rai):
        if v != rdi[i]:
            print(v)


def rename():
    rd_path = '/root/autodl-tmp/dataset/mmwave_rdi'
    file_format = 'n_walking_e6_u21_s{sample}.npy'

    for i in range(10):
        origin = file_format.format(sample=61 + i)
        new_name = file_format.format(sample=41 + i)
        os.rename(os.path.join(complex_rdi_data_root, origin), os.path.join(complex_rdi_data_root, new_name))


    # generate_single_rdi(complex_rdi_data_root, single_rdi_data_root)
# generate_time_range_doppler()