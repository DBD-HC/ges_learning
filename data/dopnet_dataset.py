import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import os

root = '/root/autodl-fs'
root2 = '/root/autodl-nas/dopnet'
trainDir = 'trainData'
testDir = 'testData'

gestures = [0, 1, 2, 3]
persons = ['A', 'B', 'C', 'D', 'E', 'F']
gesture_map = {
    'Wave': 0,
    'Pinch': 1,
    'Swipe': 2,
    'Click': 3
}

samples = {
    'A': [56, 98, 64, 105],
    'B': [112, 116, 72, 105],
    'C': [85, 132, 80, 137],
    'D': [70, 112, 71, 93],
    'E': [56, 98, 91, 144],
    'F': [87, 140, 101, 208]
}

st = {
    'A': [0, 0, 0, 0],
    'B': [0, 0, 0, 0],
    'C': [0, 0, 0, 0],
    'D': [0, 0, 0, 0],
    'E': [0, 0, 0, 0],
    'F': [0, 0, 0, 0]
}

file_format = 'p{person}_g{ges}_s{sample}.npy'


def split_raw_train_data(person='A'):
    raw_data = sio.loadmat(os.path.join(root, "Data_Per_PersonData_Training_Person_{}.mat".format(person)))
    for ges in gestures:
        print('开始分割原始数据 person:{} gesture:{}'.format(person, ges))
        for i in tqdm(range(samples[person][ges])):
            sample = raw_data["Data_Training"]["Doppler_Signals"][0][0][0][ges][i][0]
            sample = 20 * np.log10(abs(sample) / np.amax(abs(sample)))
            sample = sample.transpose(1, 0)
            file_name = os.path.join(root2, trainDir,
                                     file_format.format(person=person, ges=ges, sample=i))
            np.save(file_name, sample)


def split_raw_test_data():
    raw_data = sio.loadmat(os.path.join(root, "Data_For_Test_Random.mat"))
    raw_data = raw_data['raw_data']
    print("开始分割原始测试数据集")
    for sample_inform in tqdm(raw_data):
        sample = sample_inform[0][0][0]
        sample = 20 * np.log10(abs(sample) / np.amax(abs(sample)))
        sample = sample.transpose(1, 0)
        inform = sample_inform[1][0]
        # inform example Swipe 6 Person E
        inform = str(inform).split()
        person = inform[-1]
        ges = gesture_map[inform[0]]
        # st[int(ges)][1] = max(st[int(ges)][1], int(inform[1]))
        file_name = os.path.join(root2, testDir,
                                 file_format.format(person=person, ges=ges, sample=st[person][int(ges)]))
        np.save(file_name, sample)
        st[person][int(ges)] += 1

    print(st)




train_data_filenames = []
train_label_list = []
test_data_filenames = []
test_label_list = []


def get_samples(person, dir, ges):
    index = 0
    file_name = file_format.format(person=person, ges=ges, sample=index)
    samples = []
    while os.path.exists(os.path.join(root2, dir, file_name)):
        samples.append(file_name)
        index += 1
        file_name = file_format.format(person=person, ges=ges, sample=index)
    return samples

def data_augmentation(d):


def split_dataset():
    for person in persons:
        for ges in gestures:
            sub_test_s = get_samples(person, testDir, ges)
            sub_train_s = get_samples(person, trainDir, ges)
            train_data_filenames.extend(sub_train_s)
            test_data_filenames.extend(sub_test_s)
            train_label_list.extend([ges] * len(sub_train_s))
            test_label_list.extend([ges] * len(sub_test_s))

    return DopnetDataset(train_data_filenames, train_label_list, trainDir), DopnetDataset(test_data_filenames,
                                                                                          test_label_list, testDir)
file_cache = {}

class DopnetDataset(Dataset):
    def __init__(self, file_names, labels, dir, transform=None, max_frames=128):
        self.len = len(file_names)
        self.max_frame = max_frames
        self.file_names = np.array(file_names)
        self.dir = dir
        self.labels = np.array(labels)
        self.transform = transform
        for name in enumerate(self.file_names):
            d = np.load(os.path.join(root, self.dir, name))
            file_cache[name] = d


    def __getitem__(self, index):
        d = file_cache[self.file_names[index]]
        if self.transform is not None:
            d = self.transform(d)
        return torch.from_numpy(d), torch.tensor(self.labels[index])

    def __len__(self):
        return self.len


if __name__ == '__main__':
    train_datas, test_datas = split_dataset()
    max_frame = 0
    for it in tqdm(range(10)):
    # for it in tqdm(range(train_datas.__len__())):
        data, label = train_datas.__getitem__(it)
        max_frame = max(max_frame, data.size()[0])
    for it in tqdm(range(10)):
    # for it in tqdm(range(test_datas.__len__())):
        data, label = test_datas.__getitem__(it)
        max_frame = max(max_frame, data.size()[0])

    print("max_frame{}".format(max_frame))
    #for p in persons:
    #    print('======person'+p)
    #    split_raw_train_data(p)
    #split_raw_test_data()
