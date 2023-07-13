import math

import numpy as np
import random

data_len_adjust_gap = np.arange(2, 5)


# rotating matrix
# cosβ| − sinβ|rx(1 − cosβ) + ry*sinβ
# sinβ|   cosβ|ry(1 − cosβ) − rx*sinβ
#    0|      0| 1
# scaling matrix
#   γx|      0|sx(1 − γx)
#    0|     γy|sy(1 − γy)
#    0|      0| 1
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


def data_normalization(d, *args):
    mean = np.mean(d)
    var = np.var(d)
    d = (d - mean) / np.sqrt(var + 1e-9)
    return d


def random_data_len_adjust_2(datas, p=0.5, gap=None):
    if gap is None:
        gap = random.choice(data_len_adjust_gap)
    frame_num = len(datas)
    shape = datas.shape
    datas = datas.reshape(shape[0], -1)
    if random.uniform(0, 1) > p:
        remove_indexes = np.arange(0, frame_num, gap)
        datas = np.delete(datas, remove_indexes, axis=0)
    else:
        j = 0
        k = 0
        insert_indexes = np.arange(1, frame_num - 1, gap)
        new_datas = np.zeros((frame_num + len(insert_indexes), datas.shape[-1]))
        for i in range(frame_num):
            if k >= len(insert_indexes) or insert_indexes[k] != i:
                new_datas[j] = datas[i]
                j += 1
            else:
                new_datas[j] = (datas[i] + datas[i - 1]) / 2
                new_datas[j + 1] = datas[i]
                j += 2
                k += 1
        datas = new_datas
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
            datas[i, :d_distance[i], :] = datas.min()
        elif d_distance[i] < 0:
            datas[i, :d_distance[i], :] = datas[i, -d_distance[i]:, :]
            datas[i, d_distance[i]:, :] = datas.min()
        if d_angle[i] > 0:
            datas[i, :, d_angle[i]:] = datas[i, :, :-d_angle[i]]
            datas[i, :, :d_angle[i]] = datas.min()
        elif d_angle[i] < 0:
            datas[i, :, :d_angle[i]] = datas[i, :, -d_angle[i]:]
            datas[i, :, d_angle[i]:] = datas.min()
    return datas


def simple_shift(datas, d_distance, d_angle):
    if d_distance > 0:
        datas[:, d_distance:, :] = datas[:, :-d_distance, :]
        datas[:, :d_distance, :] = datas.min()
    elif d_distance < 0:
        datas[:, :d_distance, :] = datas[:, -d_distance:, :]
        datas[:, d_distance:, :] = datas.min()
    if d_angle > 0:
        datas[:, :, d_angle:] = datas[:, :, :-d_angle]
        datas[:, :, :d_angle] = datas.min()
    elif d_angle < 0:
        datas[:, :, :d_angle] = datas[:, :, -d_angle:]
        datas[:, :, d_angle:] = datas.min()

    return datas


def get_after_conv_size(size, kernel_size, layer, dilation=1, padding=0, stride=1, reduction=1):
    size = (size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    size = math.ceil(size / reduction)
    if layer == 1:
        return size
    else:
        return get_after_conv_size(size, kernel_size, layer - 1, dilation, padding, stride, reduction)
