import os
import random

from utils import z_score

data_type_map = {
    'RANGE_ANGLE_IMAGE':0,
    'TIME_RANGE_ANGLE_IMAGE':1,
    'COMPLEX_RANGE_DOPPLER':2,
    'SINGLE_RANGE_DOPPLER':3,
    'CROPPED_RANGE_DOPPLER_IMAGER':4,
    'TIME_RANGE_DOPPLER_IMAGE':5
}

def data_normalization(d, data_type):
    if data_type == data_type_map['TIME_RANGE_ANGLE_IMAGE'] or data_type == data_type_map['TIME_RANGE_DOPPLER_IMAGE'] :
        for channel_i in range(len(d)):
            d[channel_i] = z_score(d[channel_i])
        return d
    d = z_score(d)
    return d

class DataSpliter:
    def __init__(self, data_path, domain_num=None):
        self.train_data_filenames = []
        self.train_label_list = []
        self.test_data_filenames = []
        self.test_label_list = []
        print('========加载文件缓存========')
        self.filenames_set = set(os.listdir(data_path))
        print('========文件缓存加载完毕========')
        self.domain_num = domain_num
        self.data_type = data_type_map['RANGE_ANGLE_IMAGE']

    def set_data_type(self, data_type='RANGE_ANGLE_IMAGE'):
        self.data_type = data_type_map[data_type]

    def get_domain_num(self, domain):
        return self.domain_num[domain]

    def get_class_num(self):
        pass

    def clear_cache(self):
        pass

    def combine_reduction(self, index):
        return sum([x for i, x in enumerate(self.train_data_filenames) if i <= index], []), \
               sum([x for i, x in enumerate(self.train_label_list) if i <= index], []), \
               sum([x for i, x in enumerate(self.train_data_filenames) if i > index], []), \
               sum([x for i, x in enumerate(self.train_label_list) if i > index], [])

    def combine(self, train_index=None, val_index=None, test_index=None, is_reduction=False, need_val=False,
                need_test=False):
        if is_reduction:
            return self.combine_reduction(index)
        if train_index is None and test_index is None and val_index is None and len(self.train_data_filenames) > 0:
            sample_label_zip = list(zip(self.train_data_filenames, self.train_label_list))
            random.shuffle(sample_label_zip)
            temp_data, temp_label = zip(*sample_label_zip)
            train_data_len = int(len(temp_data) * 0.8)
            train_datas = temp_data[:train_data_len]
            train_labels = temp_label[:train_data_len]
            val_datas = temp_data[train_data_len:]
            val_labels = temp_label[train_data_len:]
            test_datas = self.test_data_filenames
            test_labels = self.test_label_list
        else:
            if test_index is not None:
                need_test = True
            temp_label = sum([x for i, x in enumerate(self.train_label_list) if i in train_index], [])
            temp_data = sum([x for i, x in enumerate(self.train_data_filenames) if i in train_index], [])
            test_datas = None
            test_labels = None
            if need_val:
                if val_index is None:
                    train_datas = []
                    train_labels = []
                    val_datas = []
                    val_labels = []
                    test_datas = []
                    test_labels = []
                    for i in train_index:
                        sample_label_zip = list(zip(self.train_data_filenames[i], self.train_label_list[i]))
                        random.shuffle(sample_label_zip)
                        temp_data, temp_label = zip(*sample_label_zip)
                        if need_test and test_index is None:
                            train_data_len = int(len(temp_data) * 0.6)
                            val_data_len = int(len(temp_data) * 0.8)
                            train_datas.extend(temp_data[:train_data_len])
                            train_labels.extend(temp_label[:train_data_len])
                            val_datas.extend(temp_data[train_data_len:val_data_len])
                            val_labels.extend(temp_label[train_data_len:val_data_len])
                            test_datas.extend(temp_data[val_data_len:])
                            test_labels.extend(temp_label[val_data_len:])
                        else:
                            train_data_len = int(len(temp_data) * 0.8)
                            train_datas.extend(temp_data[:train_data_len])
                            train_labels.extend(temp_label[:train_data_len])
                            val_datas.extend(temp_data[train_data_len:])
                            val_labels.extend(temp_label[train_data_len:])
                else:
                    train_datas = temp_data
                    train_labels = temp_label
                    val_labels = sum([x for i, x in enumerate(self.train_label_list) if i in val_index], [])
                    val_datas = sum([x for i, x in enumerate(self.train_data_filenames) if i in val_index], [])

            else:
                train_datas = temp_data
                train_labels = temp_label
                val_datas = None
                val_labels = None

        if test_index is not None:
            test_datas = sum([x for i, x in enumerate(self.train_data_filenames) if i in test_index], [])
            test_labels = sum([x for i, x in enumerate(self.train_label_list) if i in test_index], [])

        return train_datas, train_labels, test_datas, test_labels, val_datas, val_labels

    def get_dataset(self, idx=None):
        return sum([x for i, x in enumerate(self.train_data_filenames) if i in idx], []), \
               sum([x for i, x in enumerate(self.train_label_list) if i in idx], [])

    def split_data(self, domain, train_index=None, val_index=None, test_index=None, need_val=False, need_test=False,
                   need_augmentation=False, is_reduction=False):
        pass
