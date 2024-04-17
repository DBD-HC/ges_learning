import math
import os

import numpy as np
import random
from scipy import interpolate
import torch
from torchvision.transforms import transforms
from tqdm import tqdm

data_len_adjust_gap = [-5, -4, -3, 3, 4, 5]


def get_geometric_transform_mat(rotate_angle, rotate_center, scale_factor, scale_center):
    cos_b = np.cos(rotate_angle)
    sin_b = np.sin(rotate_angle)
    rotate_mat1 = np.array([[cos_b, -sin_b, rotate_center[0] * (1 - cos_b) + rotate_center[1] * sin_b],
                            [sin_b, cos_b, rotate_center[1] * (1 - cos_b) - rotate_center[1] * sin_b],
                            [0, 0, 1]])
    scaling_mat = np.array([[scale_factor[0], 0, scale_center[0] * (1 - scale_factor[0])],
                            [0, scale_factor[1], scale_center[1] * (1 - scale_factor[1])],
                            [0, 0, 1]])
    return rotate_mat1 @ scaling_mat


def random_geometric_features(datas):
    shape = datas.shape
    datas = datas.reshape(-1, shape[-1] * shape[-2])
    max_indexes = np.argmax(datas, axis=1)
    x, y = max_indexes // shape[-1], max_indexes % shape[-1]
    point_mat = np.stack([x, y, [1] * x.size], 0)
    distances = x ** 2 + y ** 2
    rotating_index = np.argmax(distances)
    rotate_angle = random.uniform(-np.pi / 12, np.pi / 12)
    scale_center = (x.sum() / x.size, y.sum() / y.size)
    scale_factor = (random.uniform(0.8, 1.2), random.uniform(0.8, 1.2))
    mat = get_geometric_transform_mat(rotate_angle, (x[rotating_index], y[rotating_index]), scale_factor, scale_center)
    point_mat_new = mat @ point_mat
    delta_xy = np.around(point_mat_new - point_mat).astype(int)
    datas = datas.reshape(-1, shape[-1], shape[-2])
    datas = simple_shift_list(datas, delta_xy[0], delta_xy[1])
    return datas

def z_score(d):
    var = np.var(d)
    mean = np.mean(d)
    d = (d - mean) / np.sqrt(var + 1e-9)
    return d


def random_data_len_adjust_2(datas, p=0.5, gap=None):
    if random.uniform(0, 1) > p:
        return datas
    if gap is None:
        gap = random.choice(data_len_adjust_gap)
    frame_num = len(datas)
    shape = datas.shape
    datas = datas.reshape(shape[0], -1)
    if gap < 0:
        remove_indexes = np.arange(np.abs(gap)-1, frame_num, np.abs(gap))
        datas = np.delete(datas, remove_indexes, axis=0)
    else:
        insert_index = frame_num // gap
        if frame_num % gap == 0:
            insert_index = insert_index - 1
        new_data = np.zeros((frame_num + insert_index, datas.shape[-1]))
        i = 0
        count = gap
        for frame in datas:
            if count == 0:
                new_data[i] = (frame + new_data[i - 1]) / 2
                i = i + 1
                count = gap
            new_data[i] = frame
            i = i + 1
            count = count - 1
        datas = new_data
    if len(shape) == 3:
        datas = datas.reshape(-1, shape[-2], shape[-1])
    else:
        datas = datas.reshape(-1, shape[-3], shape[-2], shape[-1])

    return datas


def random_data_len_adjust(datas, p=0.5):
    frame_num = len(datas)
    shape = datas.shape
    datas = datas.reshape(-1, shape[-1] * shape[-2])
    if random.uniform(0, 1) > p:
        adjusted_frame_num = random.choice(np.arange(frame_num - frame_num // 3, frame_num - frame_num // 5 + 1))
        remove_indexes = random.sample(range(0, frame_num), frame_num - adjusted_frame_num)
        datas = np.delete(datas, remove_indexes, axis=0)
    else:
        adjusted_frame_num = random.choice(np.arange(frame_num + frame_num // 5, frame_num + frame_num // 3 + 1))
        # adjusted_frame_num = random.choice(
        #     [i for i in range(frame_num - (frame_num >> 1),
        #                       128 if frame_num > 64 else frame_num << 1)])
        gap = adjusted_frame_num - frame_num
        new_datas = np.zeros((adjusted_frame_num, datas.shape[-1]))
        gap = frame_num // gap
        n_i = 0
        new_datas[0] = datas[0]
        for i in range(1, frame_num):
            if i % gap == 0 and n_i + 1 < adjusted_frame_num:
                new_datas[n_i] = (datas[i - 1] + datas[i]) / 2
                n_i += 1
                new_datas[n_i] = datas[i]
            elif n_i < adjusted_frame_num:
                new_datas[n_i] = datas[i]
            n_i += 1
        if n_i < adjusted_frame_num:
            new_datas[n_i] = datas[-1]
        datas = new_datas
    datas = datas.reshape(-1, shape[-1], shape[-2])
    return datas


def simple_shift_list(datas, d_distance, d_angle):
    for i in range(len(d_distance)):
        if d_distance[i] > 0:
            datas[i, d_distance[i]:, :] = datas[i, :-d_distance[i], :]
            datas[i, :d_distance[i], :] = datas[i].min()
        elif d_distance[i] < 0:
            datas[i, :d_distance[i], :] = datas[i, -d_distance[i]:, :]
            datas[i, d_distance[i]:, :] = datas[i].min()
        if d_angle[i] > 0:
            datas[i, :, d_angle[i]:] = datas[i, :, :-d_angle[i]]
            datas[i, :, :d_angle[i]] = datas[i].min()
        elif d_angle[i] < 0:
            datas[i, :, :d_angle[i]] = datas[i, :, -d_angle[i]:]
            datas[i, :, d_angle[i]:] = datas[i].min()
    return datas


def simple_shift(datas, d_distance, d_angle):
    if d_distance > 0:
        datas[:, d_distance:, :] = datas[:, :-d_distance, :]
        datas[:, :d_distance, :] = datas.min(axis=(1, 2))[:, None, None]
    elif d_distance < 0:
        datas[:, :d_distance, :] = datas[:, -d_distance:, :]
        datas[:, d_distance:, :] = datas.min(axis=(1, 2))[:, None, None]
    if d_angle > 0:
        datas[:, :, d_angle:] = datas[:, :, :-d_angle]
        datas[:, :, :d_angle] = datas.min(axis=(1, 2))[:, None, None]
    elif d_angle < 0:
        datas[:, :, :d_angle] = datas[:, :, -d_angle:]
        datas[:, :, d_angle:] = datas.min(axis=(1, 2))[:, None, None]

    return datas


def crop(rais):
    mean_rai = np.mean(rais, axis=(0, -1))
    mean_rai[:7] = 0
    max_index = np.argmax(mean_rai)
    rais = simple_shift(rais, rais.shape[-2]//2 - max_index, 0)
    return rais

def crop_rdi(d):
    d = np.sqrt(d[:, 0::2] ** 2 + d[:, 1::2] ** 2)
    d = np.mean(d, axis=1)
    d = crop(d)
    d = d - np.mean(d, axis=0)[None, :]
    d[d < 0] = 0
    # static removal
    d[:, :, 15:18] = 0
    return d

def resample_time(rdis, ratio):
    rdis =  np.transpose(rdis, (2, 1, 0))
    x = np.linspace(0, 1, rdis.shape[2])  # 比率为1.5
    y = np.linspace(0, 1, rdis.shape[1])
    new_x = np.linspace(0, 1, int(rdis.shape[2] * ratio))
    new_rdis = np.empty((rdis.shape[0], rdis.shape[1], len(new_x)))
    for i, rdi in enumerate(rdis):
        f = interpolate.interp2d(x, y, rdi, kind='linear')
        new_rdis[i] = f(new_x, y)
    new_rdis =  np.transpose(new_rdis, (2, 1, 0))
    return new_rdis

def resample(rdis, ratio):
    new_rdis = np.zeros_like(rdis)
    # 定义x和y的坐标
    x = np.linspace(0, 1, rdis.shape[2])  # 比率为1.5
    y = np.linspace(0, 1, rdis.shape[1])
    # 定义新的x和y的坐标，按比率重新采样
    new_x = np.linspace(0, 1, int(rdis.shape[2] * ratio))
    for i, rdi in enumerate(rdis):
        # 创建插值函数
        f = interpolate.interp2d(x, y, rdi, kind='linear')

        # 对新的x和y坐标进行插值
        len_x = len(new_x)

        if len_x > rdi.shape[1]:
            center_x = int(len_x / 2)
            left_s = rdi.shape[1] // 2
            right_s = rdi.shape[1] - left_s
            new_rdis[i] = f(new_x, y)[:, center_x - left_s:center_x + right_s]
        else:
            center_x = int(rdi.shape[1] / 2)
            left_s = len_x // 2
            right_s = len_x - left_s
            new_rdis[i, :, center_x - left_s:center_x + right_s] = f(new_x, y)[:, :]

    new_rdis = resample_time(new_rdis, ratio)

    return new_rdis


def get_after_conv_size(size, kernel_size, layer, dilation=1, padding=0, stride=1, reduction=1):
    size = (size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    size = math.ceil(size / reduction)
    if layer == 1:
        return size
    else:
        return get_after_conv_size(size, kernel_size, layer - 1, dilation, padding, stride, reduction)

def get_track(rai):
    rai_max = np.max(rai, axis=0)
    rai_mean = np.mean(rai, axis=0)
    rai_std = np.std(rai, axis=0)
    global_track = np.concatenate((rai_max[None, :], rai_mean[None, :], rai_std[None, :]), axis=0)
    return global_track


td_transform = transforms.Compose([
    transforms.Resize((224, 224)),
])

def rdi_complex_to_single(rai_path, target_path):
    d = np.load(rai_path)
    d = np.sqrt(d[:, 0::2] ** 2 + d[:, 1::2] ** 2)
    d = np.mean(d, axis=1)
    d = d - np.mean(d, axis=0)[None, :]
    d[d < 0] = 0
    d[:, :, 15:18] = 0
    np.save(target_path, d.astype(np.float32))

def generate_single_rdi(rdi_root, target_root):
    rai_filenames = os.listdir(rdi_root)
    for filename in tqdm(rai_filenames):
        if filename.endswith('.npy'):
            rdi_complex_to_single(os.path.join(rdi_root, filename), os.path.join(target_root, filename))

def rais_to_time_frequency(rai_path, target_path):
    rai = np.load(rai_path)
    rai = z_score(rai)
    time_angle = np.mean(rai, axis=-2)[None, :]
    time_range = np.mean(rai, axis=-1)[None, :]
    input_data = np.concatenate((time_range, time_angle), axis=0)
    input_data = torch.from_numpy(input_data)
    input_data = td_transform(input_data)
    input_data = input_data.numpy()
    np.save(target_path, input_data.astype(np.float32))

def generate_time_doppler(rai_root, target_root):
    rai_filenames = os.listdir(rai_root)
    for filename in tqdm(rai_filenames):
        if filename.endswith('.npy'):
            rais_to_time_frequency(os.path.join(rai_root, filename), os.path.join(target_root, filename))