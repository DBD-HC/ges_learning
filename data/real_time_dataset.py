import os
import random

from data.data_splitter import data_normalization
from data.rai_ges_dataset import RAIGes, data_augmentation

raiges_root = '/root/autodl-tmp/dataset/rai_ges/rai_data'
mcd_root = '/root/autodl-tmp/dataset/mmWave_cross_domain_gesture_dataset'

train_set_data = []
test_set_data = []
train_set_label = []
test_set_label = []


di_negative_gesture = ['n_liftleft', 'n_liftright', 'n_sit', 'n_stand', 'n_turn', 'n_waving', 'n_walking']
di_envs = ['e1', 'e2', 'e3', 'e4', 'e5', 'e6']
di_participants = ['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8', 'u9', 'u10', 'u11', 'u12', 'u13',
                        'u14', 'u15', 'u16', 'u17', 'u18', 'u19', 'u20', 'u21', 'u22', 'u23', 'u24', 'u25']
di_locations = ['p1', 'p2', 'p3', 'p4', 'p5']
di_file_format = '{act}_{env}_{user}_{loc}_s{s}.npy'
di_file_format_walking = '{act}_{env}_{user}_s{s}.npy'



cp_gestures =  ['0', '1', '2', '3', '4', '5']
cp_envs =  ['0', '1', '2', '3']
cp_participants = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
cp_locations = ['0', '1', '2', '3', '4']
cp_file_format = 'rai_{ges}_{user}_{position}_{env}_s{sample}.npy'

di_set = set(os.listdir(mcd_root))
cp_set = set(os.listdir(raiges_root))

def check_and_get(data_path ,file_name_prefix, sample_label):
    sample_index = 1
    file_name = file_name_prefix.format(s=sample_index)
    samples = []
    sample_labels = []
    while file_name in di_set or file_name in cp_set:
        samples.append(os.path.join(data_path, file_name))
        sample_labels.append(sample_label)
        sample_index += 1
        file_name = file_name_prefix.format(s = sample_index)
    return samples, sample_labels

def get_real_time_data():
    if len(train_set_data) == 0:
        samples = [[], [], [], [], [], [], []]
        sample_labels = [[], [], [], [], [], [], []]
        ## get_negative
        for env_i, env in enumerate(di_envs):
            for p_i, p in enumerate(di_participants):
                for act_i, act in enumerate(di_negative_gesture):
                    for loc_i, loc in enumerate(di_locations):
                        if act == di_negative_gesture[-1]:
                            # '{act}_{env}_{user}_s{s}.npy'
                            file_name_prefix = di_file_format_walking.format(act=act, env=env, user=p, s='{s}')
                        else:
                            # '{act}_{env}_{user}_{loc}_s{s}.npy'
                            file_name_prefix = di_file_format.format(act=act, env=env, user=p, loc=loc, s='{s}')
                        temp_samples, temp_labels = check_and_get(mcd_root, file_name_prefix, act_i)
                        samples[-1].extend(temp_samples)
                        sample_labels[-1].extend([6] * len(temp_samples))
                        if act == di_negative_gesture[-1]:
                            break

        for env_i, env in enumerate(cp_envs):
            for p_i, p in enumerate(cp_participants):
                for act_i, act in enumerate(cp_gestures):
                    for loc_i, loc in enumerate(cp_locations):
                        # rai_{ges}_{user}_{position}_{env}_s{sample}.npy'
                        file_name_prefix = cp_file_format.format(ges=act, user=p, position=loc, env=env, sample="{s}")
                        temp_samples, temp_labels = check_and_get(raiges_root, file_name_prefix, act_i)
                        samples[act_i].extend(temp_samples)
                        sample_labels[act_i].extend(temp_labels)

        for i, item in enumerate(samples):
            random.shuffle(item)
            data_len = len(item)
            gap = data_len // 5
            train_set_data.extend(item[:gap * 4])
            train_set_label.extend(sample_labels[i][:gap * 4])
            test_set_data.extend(item[gap * 4:])
            test_set_label.extend(sample_labels[i][gap * 4:])

    return RAIGes(train_set_data, train_set_label, data_augmentation), \
           RAIGes(test_set_data, test_set_label, data_normalization)





