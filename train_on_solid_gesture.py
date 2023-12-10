from train_on_di_gesture import train, plot_result
import numpy as np
import torch
import torchmetrics

from data.solid_gesture_dataset import *
import torch.optim as optim
from model.network import *
import matplotlib.pyplot as plt
import visdom
from torch.nn.utils.rnn import pad_sequence
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, average_precision_score
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC, MulticlassAveragePrecision


def get_lr(epoch=200):
    lr_list = np.zeros(epoch)
    lr_list[:epoch] = 0.0001
    return lr_list


def cross_user(user_range=(0, 10), epoch=100):
    print('cross env')
    res_history = []
    lr_main = get_lr(epoch)
    for e in range(user_range[0], user_range[1]):
        train_set, test_set = split_dataset('cross_person', e)
        net = RAIRadarGestureClassifier(cfar=False, track=True, spatial_channels=(4, 8, 16), ra_conv=True, heads=4,
                                        track_channels=(4, 8, 16), track_out_size=64, hidden_size=(128, 128),
                                        ra_feat_size=32, attention=True, cfar_expand_channels=8, in_channel=1)
        res = train(train_set, test_set, total_epoch=epoch, net=net, lr_list=lr_main)
        res_history.append(res)
        plot_result('solid_cross_user_{}'.format(e))
        print('=====================solid_cross_user {} for test acc_history:{}================='.format(e + 1, res_history))
        np.save('solid_user.npy', np.array(res_history))


cross_user()