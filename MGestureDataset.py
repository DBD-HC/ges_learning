import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import joblib
from utils import *
import scipy.io as sio

# root = '/root/autodl-fs/m-gesture/short_RangeDoppler'
root = 'D:\\dataset\\M-GestureReleaseData\\long_raw'
gestures = ['knock', 'lswipe', 'rswipe', 'rotate', 'unex']
mp4_file_name_format = 'short_RD_{user}_{ges}_ ({s}).mp4'
file_name_format = 'short_RD_{user}_{ges}_{s}.joblib'
long_raw_name_format = 'long_raw_{user}_{ges}_{s}.mat'
users = ['011', '014', '025', '026', '032', '036', '066', '071', '083', '089']


# 定义一个下采样的函数
def downsample(img, w=32, h=32):
    return cv2.resize(img, (w, h))


def data_transform():
    for d in tqdm(range(120, 135)):
        filenames_set = set(os.listdir(os.path.join(root, str(d))))
        count = [1, 1, 1, 1, 1, 1]
        for x in filenames_set:
            if not x.endswith('.mp4'):
                continue
            inf = x.split('_')
            ges = inf[3]
            if ges not in gestures:
                if ges == 'antiknock':
                    ges = 'anticlock'
                else:
                    print(x)
                    return 0
            ges_index = gestures.index(ges)
            file_name = os.path.join(root, str(d), file_name_format.format(user=d, ges=ges, s=count[ges_index]))
            count[ges_index] += 1
            # 打开视频文件
            cap = cv2.VideoCapture(os.path.join(root, str(d), x))

            # 获取视频的宽度和高度
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 计算视频的总帧数
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 创建一个空的numpy数组，用于存储所有帧的数据，并定义维度
            RDI = np.zeros((frame_count, 128, 128, 3), dtype='uint8')

            # 逐帧读取视频，将每一帧存储到numpy数组中
            for i in range(frame_count):
                ret, frame = cap.read()
                if ret:
                    RDI[i] = downsample(frame, 128, 128)
                else:
                    break
            print(file_name)
            try:
                joblib.dump(RDI, file_name)
            except OSError as e:
                print(e)

            print('saved')
            cap.release()


train_data = []
train_label = []

file_cache = {}


def split_data(u=0):
    if len(train_data) == 0:
        for user in range(120, 135):
            filenames_set = set(os.listdir(os.path.join(root, str(user))))
            filenames = []
            labels = []
            for i, ges in enumerate(gestures):
                count = 1
                file_name = file_name_format.format(user=user, ges=ges, s=count)
                while file_name in filenames_set and count <= 10:
                    filenames.append(os.path.join(root, str(user), file_name))
                    labels.append(i)
                    count += 1
                    file_name = file_name_format.format(user=user, ges=ges, s=count)
            train_data.append(filenames)
            train_label.append(labels)
    if len(file_cache) == 0:
        print('=======初始化数据集缓存====')
        for names in train_data:
            for name in tqdm(names):
                d = joblib.load(name)
                d = np.transpose(d, (0, 3, 1, 2))
                file_cache[name] = d
        print('=======数据集缓存初始化完毕====')

    data_for_train = []
    label_for_train = []
    data_for_test = []
    label_for_test = []

    for i in range(15):
        if i != u:
            data_for_train.extend(train_data[i])
            label_for_train.extend(train_label[i])
        else:
            data_for_test.extend(train_data[i])
            label_for_test.extend(train_label[i])

    return MGestureDataset(data_for_train, label_for_train, transform=data_augmentation), MGestureDataset(data_for_test,
                                                                                                          label_for_test)


static_angle_range = [np.arange(-4, 5), np.arange(-1, 2)]
static_distance_range = [np.arange(-2, 3), np.arange(-2, 3)]


def random_translation(datas, p=0.5):
    d_distance = random.choice(static_distance_range[0])
    d_angle = random.choice(static_angle_range[0])
    d_distance = 0
    shape = datas.shape
    datas = datas.reshape(-1, shape[-2], shape[-1])
    simple_shift(datas, d_distance, d_angle)
    return datas.reshape(shape), d_distance, d_angle


def random_track_adjust(d, p=0.5):
    shape = d.shape
    d = d.reshape(-1, shape[-2], shape[-1])
    x = random.choices([-1, 0, 1], k=shape[0])
    y = random.choices([-1, 0, 1], k=shape[0])
    x = np.repeat(x, shape[1])
    y = np.repeat(y, shape[1])
    d = simple_shift_list(d, x, y)
    d = d.reshape(-1, shape[-3], shape[-2], shape[-1])

    return d


gesture_pairs = {
    0: 0,
    1: 2,
    2: 1,
    3: 4,
    4: 3,
    5: 5
}


def random_reverse(datas, labels, p=0.5):
    cond = random.uniform(0, 1)
    if cond > p:
        datas = np.flip(datas, axis=0)
        labels = gesture_pairs[labels]
    return datas, labels


adjust_gap = np.arange(4, 6)


def data_augmentation(d, label):
    # d = data_normalization(d)
    shape = d.shape
    d = d.reshape(-1, shape[-2], shape[-1])
    d, dis, angle = random_translation(d)
    d = d.reshape(-1, shape[-3], shape[-2], shape[-1])
    # d, label = random_reverse(d, label)
    # d = random_track_adjust(d)
    # gap = random.choice(adjust_gap)
    # d = random_data_len_adjust_2(d, gap)
    return d, label


class MGestureDataset(Dataset):
    def __init__(self, file_names, labels, transform=None):
        self.len = len(file_names)
        self.file_names = np.array(file_names)
        # permutation = np.random.permutation(self.len)
        # self.file_names = self.file_names[permutation]
        self.labels = np.array(labels)
        # self.labels = self.labels[permutation]
        self.transform = transform

    def __getitem__(self, index):
        d = file_cache[self.file_names[index]]
        label = self.labels[index]
        # d = np.load(os.path.join(root, self.file_names[index]))
        if self.transform is not None:
            res = self.transform(d, self.labels[index])
            d = res[0]
            label = res[1]

        return torch.from_numpy(d).type(torch.float32), torch.tensor(label), index

    def __len__(self):
        return self.len


if __name__ == '__main__':
    data_transform()
