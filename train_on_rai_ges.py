import os

import torch
import visdom

from data.rai_ges_dataset import RAIGesDataSplitter
from log_helper import LogHelper
from train import TIME_FREQUENCY_IMAGE, RANGE_ANGLE_IMAGE, new_cross_domain, new_train_transferring, new_cube_k_fold, \
    train_for_real_time

if __name__ == '__main__':
    loger = LogHelper()
    visdom_port = 6006
    vis = visdom.Visdom(env='model result', port=visdom_port)
    random_seed = 1998

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    complex_DataSplitter = RAIGesDataSplitter()

    new_cube_k_fold(augmentation=True, epoch=200, start_epoch=0, domain=0, data_type=RANGE_ANGLE_IMAGE, batch_size=128,
                   model_type=0, train_manager=None, data_spliter=complex_DataSplitter)
    new_cube_k_fold(augmentation=True, epoch=200, start_epoch=0, domain=0, data_type=TIME_FREQUENCY_IMAGE, batch_size=128,
                   model_type=1, train_manager=None, data_spliter=complex_DataSplitter)

 '''
    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=3,
                     train_index=[0, 6, 7],
                     test_index=[1, 2, 3, 4, 5, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=200, data_spliter=complex_DataSplitter)

    new_cross_domain(augmentation=True, model_type=1, data_type=RANGE_ANGLE_IMAGE, domain=3,
                     train_index=[0, 6, 7],
                     test_index=[1, 2, 3, 4, 5, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=200, data_spliter=complex_DataSplitter)

    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=3,
                     train_index=[0, 6, 7],
                     test_index=[1, 2, 3, 4, 5, 8, 9],
                     val_time=5,
                     multistream=False, diff=True,
                     attention=True, epoch=200, data_spliter=complex_DataSplitter)

    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=3,
                     train_index=[0, 6, 7],
                     test_index=[1, 2, 3, 4, 5, 8, 9],
                     val_time=5,
                     multistream=True, diff=False,
                     attention=True, epoch=200, data_spliter=complex_DataSplitter)

    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=3,
                     train_index=[0, 6, 7],
                     test_index=[1, 2, 3, 4, 5, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=False, epoch=200, data_spliter=complex_DataSplitter)

 
    new_train_transferring(augmentation=True, domain=1, model_type=0, data_type=RANGE_ANGLE_IMAGE,
                           train_index=[0],
                           test_index=[1, 2, 3],
                           need_test=True,
                           pre_train_lr=0.001, pre_train_epoch=50, train_epoch=100)
    new_train_transferring(augmentation=True, domain=3, model_type=0, data_type=RANGE_ANGLE_IMAGE,
                           train_index=[0, 6, 7],
                           test_index=[1, 2, 3, 4, 5, 8, 9],
                           need_test=False,
                           pre_train_lr=0.001, pre_train_epoch=50, train_epoch=100)

    train_for_real_time(model_type=0)
    train_for_real_time(model_type=1)
    train_for_real_time(model_type=2)

    new_cube_k_fold(augmentation=True, epoch=200, start_epoch=0, domain=0, data_type=RANGE_ANGLE_IMAGE, batch_size=128,
                   model_type=2, train_manager=None, data_spliter=complex_DataSplitter)
    new_cube_k_fold(augmentation=True, epoch=200, start_epoch=0, domain=0, data_type=TIME_FREQUENCY_IMAGE, batch_size=128,
                   model_type=6, train_manager=None, data_spliter=complex_DataSplitter)


    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=2,
                     train_index=[1],
                     test_index=[0, 2, 3, 4],
                     val_time=1,
                     need_test=True,
                     multistream=True, diff=True, need_save_result=False,
                     attention=True, epoch=200, data_spliter=complex_DataSplitter)


    new_train_transferring(augmentation=True, domain=2, model_type=0, data_type=RANGE_ANGLE_IMAGE,
                           train_index=[1],
                           test_index=[0, 2, 3, 4],
                           need_test=True,
                           pre_train_lr=0.001, pre_train_epoch=50, train_epoch=100)

    train_for_real_time(model_type=0)
    train_for_real_time(model_type=1)
    train_for_real_time(model_type=2)
    '''
    '''
    new_train_transferring(augmentation=True, domain=2, model_type=0, data_type=RANGE_ANGLE_IMAGE,
                           train_index=[1],
                           test_index=[0, 2, 3, 4],
                           need_test=True,
                           pre_train_lr=0.001, pre_train_epoch=50, train_epoch=100)

    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=1,
                     train_index=[0],
                     test_index=[1, 2, 3],
                     val_time=5,
                     need_test=True,
                     multistream=True, diff=True,
                     attention=True, epoch=200, data_spliter=complex_DataSplitter)

    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=2,
                     train_index=[1],
                     test_index=[0, 2, 3, 4],
                     val_time=5,
                     need_test=True,
                     multistream=True, diff=True,
                     attention=True, epoch=200, data_spliter=complex_DataSplitter)

    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=3,
                     train_index=[0, 6, 7],
                     test_index=[1, 2, 3, 4, 5, 8, 9],
                     val_time=5,
                     need_test=True,
                     multistream=True, diff=True,
                     attention=True, epoch=200, data_spliter=complex_DataSplitter)
    # ==========
    new_cross_domain(augmentation=True, model_type=1, data_type=RANGE_ANGLE_IMAGE, domain=1,
                     train_index=[0],
                     test_index=[1, 2, 3],
                     val_time=5,
                     need_test=True,
                     multistream=True, diff=True,
                     attention=True, epoch=200, data_spliter=complex_DataSplitter)

    new_cross_domain(augmentation=True, model_type=1, data_type=RANGE_ANGLE_IMAGE, domain=2,
                     train_index=[1],
                     test_index=[0, 2, 3, 4],
                     val_time=5,
                     need_test=True,
                     multistream=True, diff=True,
                     attention=True, epoch=200, data_spliter=complex_DataSplitter)

    new_cross_domain(augmentation=True, model_type=1, data_type=RANGE_ANGLE_IMAGE, domain=3,
                     train_index=[0, 6, 7],
                     test_index=[1, 2, 3, 4, 5, 8, 9],
                     val_time=5,
                     need_test=True,
                     multistream=True, diff=True,
                     attention=True, epoch=200, data_spliter=complex_DataSplitter)

    # ==========
    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=1,
                     train_index=[0],
                     test_index=[1, 2, 3],
                     val_time=5,
                     need_test=True,
                     multistream=False, diff=True,
                     attention=True, epoch=200, data_spliter=complex_DataSplitter)

    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=2,
                     train_index=[1],
                     test_index=[0, 2, 3, 4],
                     val_time=5,
                     need_test=True,
                     multistream=False, diff=True,
                     attention=True, epoch=200, data_spliter=complex_DataSplitter)

    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=3,
                     train_index=[0, 6, 7],
                     test_index=[1, 2, 3, 4, 5, 8, 9],
                     val_time=5,
                     need_test=True,
                     multistream=False, diff=True,
                     attention=True, epoch=200, data_spliter=complex_DataSplitter)
    # ==========
    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=1,
                     train_index=[0],
                     test_index=[1, 2, 3],
                     val_time=5,
                     need_test=True,
                     multistream=True, diff=False,
                     attention=True, epoch=200, data_spliter=complex_DataSplitter)

    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=2,
                     train_index=[1],
                     test_index=[0, 2, 3, 4],
                     val_time=5,
                     need_test=True,
                     multistream=True, diff=False,
                     attention=True, epoch=200, data_spliter=complex_DataSplitter)

    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=3,
                     train_index=[0, 6, 7],
                     test_index=[1, 2, 3, 4, 5, 8, 9],
                     val_time=5,
                     need_test=True,
                     multistream=True, diff=False,
                     attention=True, epoch=200, data_spliter=complex_DataSplitter)
    # ==========
    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=1,
                     train_index=[0],
                     test_index=[1, 2, 3],
                     val_time=5,
                     need_test=True,
                     multistream=True, diff=True,
                     attention=False, epoch=200, data_spliter=complex_DataSplitter)

    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=2,
                     train_index=[1],
                     test_index=[0, 2, 3, 4],
                     val_time=5,
                     need_test=True,
                     multistream=True, diff=True,
                     attention=False, epoch=200, data_spliter=complex_DataSplitter)

    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=3,
                     train_index=[0, 6, 7],
                     test_index=[1, 2, 3, 4, 5, 8, 9],
                     val_time=5,
                     need_test=True,
                     multistream=True, diff=True,
                     attention=False, epoch=200, data_spliter=complex_DataSplitter)

    '''
    #new_cube_k_fold(augmentation=True, epoch=200, start_epoch=0, domain=0, data_type=RANGE_ANGLE_IMAGE, batch_size=128,
    #                model_type=2, train_manager=None, data_spliter=complex_DataSplitter)
    #new_cube_k_fold(augmentation=True, epoch=200, start_epoch=0, domain=0, data_type=TIME_FREQUENCY_IMAGE, batch_size=128,
    #                model_type=6, train_manager=None, data_spliter=complex_DataSplitter)
    '''
    new_train_transferring(augmentation=True, domain=1, model_type=0, data_type=RANGE_ANGLE_IMAGE,
                           train_index=[0],
                           test_index=[1, 2, 3],
                           need_test=True,
                           pre_train_lr=0.001, pre_train_epoch=100, train_epoch=100)
    new_train_transferring(augmentation=True, domain=2, model_type=0, data_type=RANGE_ANGLE_IMAGE,
                           train_index=[1],
                           test_index=[0, 2, 3, 4],
                           need_test=True,
                           pre_train_lr=0.001, pre_train_epoch=100, train_epoch=100)
    new_train_transferring(augmentation=True, domain=3, model_type=0, data_type=RANGE_ANGLE_IMAGE,
                           train_index=[0, 6, 7],
                           test_index=[1, 2, 3, 4, 5, 8, 9],
                           need_test=False,
                           pre_train_lr=0.001, pre_train_epoch=100, train_epoch=100)

    # deep_rai
    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=2,
                 train_index=[1],
                 test_index=[0, 2, 3, 4],
                 val_time=5,
                 need_test=True,
                 multistream=True, diff=True,
                 attention=True, epoch=200, data_spliter=complex_DataSplitter)

    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=3,
             # train_index=[0, 1, 2, 3, 4],
             # test_index=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
             train_index=[0, 6, 7],
             test_index=[1, 2, 3, 4, 5, 8, 9],
             # val_index=[5, 6],
             # test_index=[3, 4, 5, 6, 7, 8, 9],
             val_time=5,
             multistream=True, diff=True,
             attention=True, epoch=200, data_spliter=complex_DataSplitter)

    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=1,
                 train_index=[0],
                 test_index=[1, 2, 3],
                 val_time=5,
                 need_test=True,
                 multistream=True, diff=True,
                 attention=True, epoch=200, data_spliter=complex_DataSplitter)

    # deep_rai_small
    new_cross_domain(augmentation=True, model_type=1, data_type=RANGE_ANGLE_IMAGE, domain=2,
                 train_index=[1],
                 test_index=[0, 2, 3, 4],
                 val_time=5,
                 need_test=True,
                 multistream=True, diff=True,
                 attention=True, epoch=200, data_spliter=complex_DataSplitter)

    new_cross_domain(augmentation=True, model_type=1, data_type=RANGE_ANGLE_IMAGE, domain=3,
             # train_index=[0, 1, 2, 3, 4],
             # test_index=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
             train_index=[0, 6, 7],
             test_index=[1, 2, 3, 4, 5, 8, 9],
             # val_index=[5, 6],
             # test_index=[3, 4, 5, 6, 7, 8, 9],
             val_time=5,
             multistream=True, diff=True,
             attention=True, epoch=200, data_spliter=complex_DataSplitter)

    new_cross_domain(augmentation=True, model_type=1, data_type=RANGE_ANGLE_IMAGE, domain=1,
                 train_index=[0],
                 test_index=[1, 2, 3],
                 val_time=5,
                 need_test=True,
                 multistream=True, diff=True,
                 attention=True, epoch=200, data_spliter=complex_DataSplitter)

    # deep_rai single stream
    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=2,
                 train_index=[1],
                 test_index=[0, 2, 3, 4],
                 val_time=5,
                 need_test=True,
                 multistream=False, diff=True,
                 attention=True, epoch=200, data_spliter=complex_DataSplitter)

    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=3,
             # train_index=[0, 1, 2, 3, 4],
             # test_index=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
             train_index=[0, 6, 7],
             test_index=[1, 2, 3, 4, 5, 8, 9],
             # val_index=[5, 6],
             # test_index=[3, 4, 5, 6, 7, 8, 9],
             val_time=5,
             multistream=False, diff=True,
             attention=True, epoch=200, data_spliter=complex_DataSplitter)

    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=1,
                 train_index=[0],
                 test_index=[1, 2, 3],
                 val_time=5,
                 need_test=True,
                 multistream=False, diff=True,
                 attention=True, epoch=200, data_spliter=complex_DataSplitter)

    # deep_rai no diff
    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=2,
                 train_index=[1],
                 test_index=[0, 2, 3, 4],
                 val_time=5,
                 need_test=True,
                 multistream=True, diff=False,
                 attention=True, epoch=200, data_spliter=complex_DataSplitter)

    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=3,
             # train_index=[0, 1, 2, 3, 4],
             # test_index=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
             train_index=[0, 6, 7],
             test_index=[1, 2, 3, 4, 5, 8, 9],
             # val_index=[5, 6],
             # test_index=[3, 4, 5, 6, 7, 8, 9],
             val_time=5,
             multistream=True, diff=False,
             attention=True, epoch=200, data_spliter=complex_DataSplitter)

    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=1,
                 train_index=[0],
                 test_index=[1, 2, 3],
                 val_time=5,
                 need_test=True,
                 multistream=True, diff=False,
                 attention=True, epoch=200, data_spliter=complex_DataSplitter)

    # deep_rai no attention
    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=2,
                     train_index=[1],
                     test_index=[0, 2, 3, 4],
                     val_time=5,
                     need_test=True,
                     multistream=True, diff=True,
                     attention=False, epoch=200, data_spliter=complex_DataSplitter)

    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=3,
                     # train_index=[0, 1, 2, 3, 4],
                     # test_index=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                     train_index=[0, 6, 7],
                     test_index=[1, 2, 3, 4, 5, 8, 9],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=False, epoch=200, data_spliter=complex_DataSplitter)

    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=1,
                     train_index=[0],
                     test_index=[1, 2, 3],
                     val_time=5,
                     need_test=True,
                     multistream=True, diff=True,
                     attention=False, epoch=200, data_spliter=complex_DataSplitter) '''
