from torch.utils.data import Dataset
from data.data_splitter import DataSpliter, data_type_map, data_normalization
from utils import *

rai_root = '/root/autodl-tmp/dataset/rai_ges/rai_data'
time_range_angle_root = '/root/autodl-tmp/dataset/rai_ges/tf_data'

static_angle_range = np.arange(-20, 21)
static_distance_range = np.arange(-6, 7)

def random_translation(datas):
    d_distance = random.choice(static_distance_range)
    d_angle = random.choice(static_distance_range)
    simple_shift(datas, d_distance, d_angle)
    return datas


def data_augmentation(d, data_type):
    # rai
    if data_type == data_type_map['RANGE_ANGLE_IMAGE']:
        d = random_translation(d)
        d = random_geometric_features(d)
        d = random_data_len_adjust_2(d)
    elif data_type == data_type_map['CROPPED_RANGE_DOPPLER_IMAGER'] or data_type == data_type_map['CROPPED_RANGE_ANGLE_IMAGER']:
        #d = crop_rai(d)    
        d = random_rdi_speed(d)

    d = data_normalization(d, data_type)
    return d

def pre_processing(d, data_type):
    if data_type == data_type_map['CROPPED_RANGE_DOPPLER_IMAGER'] or data_type == data_type_map['CROPPED_RANGE_ANGLE_IMAGER']:
        d = crop_rai(d)  
    return d


class RAIGesDataSplitter(DataSpliter):
    def __init__(self, data_path=rai_root):
        self.gestures = ['0', '1', '2', '3', '4', '5']
        self.envs = ['0', '1', '2', '3']
        self.participants = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.locations = ['0', '1', '2', '3', '4']

        super(RAIGesDataSplitter, self).__init__(data_path=data_path, domain_num=(
        3, len(self.envs), len(self.locations), len(self.participants), 0))
        self.file_format = 'rai_{ges}_{user}_{position}_{env}_s{sample}.npy'
        self.pre_domain = -1
        self.od_for_train = [[0], [1] , [0, 6, 7]]
        self.pre_processing = pre_processing

    def get_domain_num(self, domain):
        return self.domain_num[domain]

    def get_class_num(self):
        return len(self.gestures)

    def clear_cache(self):
        self.train_data_filenames.clear()
        self.train_label_list.clear()
        self.test_data_filenames.clear()
        self.test_label_list.clear()

    def get_dataset(self, idx=None):
        datas = super().get_dataset(idx)
        return RAIGes(datas[0], datas[1], data_normalization, data_type=self.data_type)

    def split_data(self, domain, train_index=None, val_index=None, test_index=None, need_val=False, need_test=False,
                   need_augmentation=True, is_reduction=False):
        if self.pre_domain != domain:
            self.clear_cache()
            self.pre_domain = domain

        domain_index = [0, 0, 0, 0]
        domain_num = self.domain_num[domain]
        if len(self.train_data_filenames) == 0:
            for i in range(domain_num):
                self.train_data_filenames.append([])
                self.train_label_list.append([])
            if domain == 0:
                for g_i, g in enumerate(self.gestures):
                    for e in self.envs:
                        for p in self.locations:
                            for u_i, u in enumerate(self.participants):
                                index = 1
                                filename = self.file_format.format(ges=g, user=u, position=p, env=e, sample=index)
                                temp_data = []
                                temp_label = []
                                while filename in self.filenames_set:
                                    temp_data.append(filename)
                                    temp_label.append(g_i)
                                    index += 1
                                    filename = self.file_format.format(ges=g, user=u, position=p, env=e, sample=index)
                                if len(temp_data) == 0:
                                    continue
                                size = len(temp_data) // len(self.train_data_filenames)
                                random.shuffle(temp_data)
                                j = 0
                                for i in range(0, len(temp_data), size):
                                    if j == len(self.train_data_filenames) - 1:
                                        self.train_data_filenames[j].extend(temp_data[i:])
                                        self.train_label_list[j].extend(temp_label[i:])
                                        break
                                    else:
                                        self.train_data_filenames[j].extend(temp_data[i:i + size])
                                        self.train_label_list[j].extend(temp_label[i:i + size])
                                    j += 1
            elif domain == 1 or domain == 2 or domain == 3:
                for g_i, g in enumerate(self.gestures):
                    for e_i, e in enumerate(self.envs):
                        domain_index[1] = e_i
                        for p_i, p in enumerate(self.locations):
                            domain_index[2] = p_i
                            for u_i, u in enumerate(self.participants):
                                domain_index[3] = u_i
                                index = 1
                                filename = self.file_format.format(ges=g, user=u, position=p, env=e, sample=index)
                                while filename in self.filenames_set:
                                    self.train_data_filenames[domain_index[domain]].append(filename)
                                    self.train_label_list[domain_index[domain]].append(g_i)
                                    index += 1
                                    filename = self.file_format.format(ges=g, user=u, position=p, env=e, sample=index)
            else:
                # 'rai_{ges}_{user}_{position}_{env}_s{sample}.npy'
                for filename in self.filenames_set:
                    if not filename.endswith('.npy'):
                        continue
                    values = filename.split('_')
                    act = int(values[1])
                    cur_user_idx = int(values[2])
                    cur_env_idx = int(values[4])
                    cur_loc_idx = int(values[3])
                    if cur_env_idx in self.od_for_train[0] \
                        and cur_loc_idx in self.od_for_train[1] \
                        and cur_user_idx in self.od_for_train[-1]:
                        self.train_data_filenames.append(filename)
                        self.train_label_list.append(act)
                    else:
                        self.test_data_filenames.append(filename)
                        self.test_label_list.append(act)

        datas = super().combine(train_index, val_index, test_index, is_reduction, need_val, need_test)
        return RAIGes(datas[0], datas[1], data_augmentation if need_augmentation else data_normalization, data_type=self.data_type), \
               RAIGes(datas[2], datas[3], data_normalization, data_type=self.data_type), \
               RAIGes(datas[4], datas[5], data_normalization, data_type=self.data_type)


class RAIGes(Dataset):
    def __init__(self, file_names, labels, transform=None, data_root=None, data_type=0):
        if file_names is None:
            return
        self.len = len(file_names)
        self.file_names = np.array(file_names)
        self.labels = np.array(labels)
        self.data_type = data_type
        if data_root is not None:
            self.data_root = data_root
        elif data_type == data_type_map['RANGE_ANGLE_IMAGE'] or data_type == data_type_map['CROPPED_RANGE_ANGLE_IMAGER']:
            self.data_root = rai_root
        else:
            self.data_root = time_range_angle_root

        self.transform = transform
        self.pre_processing = pre_processing


    def __getitem__(self, index):
        d = np.load(os.path.join(self.data_root, self.file_names[index]))
        label = self.labels[index]
        d = self.pre_processing(d, self.data_type)
        d = self.transform(d, self.data_type)

        label = torch.tensor(label)
        if self.data_type == data_type_map['RANGE_ANGLE_IMAGE']:
            track = torch.from_numpy(get_track(d)).type(torch.float32)
            d = torch.from_numpy(d).type(torch.float32)
            return d, track, label
        return torch.from_numpy(d).type(torch.float32), label

    def __len__(self):
        return self.len


if __name__ == '__main__':
    complex_DataSplitter = RAIGesDataSplitter()
    complex_DataSplitter.split_data(1)
    # clear_data()
    print('110')
