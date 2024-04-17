import os

import torch
import visdom

from data.mcd_dataset import MCDDataSplitter
from log_helper import LogHelper
from train import TIME_FREQUENCY_IMAGE, RANGE_ANGLE_IMAGE, new_cross_domain, SINGLE_RANGE_DOPPLER, \
    CROPPED_RANGE_DOPPLER_IMAGER, COMPLEX_RANGE_DOPPLER, new_cube_k_fold

if __name__ == '__main__':
    loger = LogHelper()
    visdom_port = 6006
    vis = visdom.Visdom(env='model result', port=visdom_port)
    random_seed = 1998

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    di_DataSplitter = MCDDataSplitter()

    new_cube_k_fold(augmentation=True, epoch=200, start_epoch=0, domain=0, data_type=RANGE_ANGLE_IMAGE, batch_size=128,
                    model_type=0, train_manager=None, data_spliter=di_DataSplitter)
    new_cube_k_fold(augmentation=True, epoch=200, start_epoch=0, domain=0, data_type=RANGE_ANGLE_IMAGE, batch_size=128,
                    model_type=1, train_manager=None, data_spliter=di_DataSplitter)
    '''
    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=1,
                     train_index=[0, 1, 2, 3, 4],
                     test_index=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=200, data_spliter=di_DataSplitter)

    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=2,
                     train_index=[0],
                     test_index=[1, 2, 3, 4, 5],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=200, data_spliter=di_DataSplitter)
    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=2,
                     train_index=[5],
                     test_index=[0, 1, 2, 3, 4],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=200, data_spliter=di_DataSplitter)
    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=3,
                     train_index=[1],
                     test_index=[0, 2, 3, 4],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     need_test=True,
                     multistream=True, diff=True,
                     attention=True, epoch=200, data_spliter=di_DataSplitter)
    # ===========
    new_cross_domain(augmentation=True, model_type=1, data_type=RANGE_ANGLE_IMAGE, domain=1,
                     train_index=[0, 1, 2, 3, 4],
                     test_index=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=200, data_spliter=di_DataSplitter)

    new_cross_domain(augmentation=True, model_type=1, data_type=RANGE_ANGLE_IMAGE, domain=2,
                     train_index=[0],
                     test_index=[1, 2, 3, 4, 5],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=200, data_spliter=di_DataSplitter)
    new_cross_domain(augmentation=True, model_type=1, data_type=RANGE_ANGLE_IMAGE, domain=2,
                     train_index=[5],
                     test_index=[0, 1, 2, 3, 4],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=200, data_spliter=di_DataSplitter)
    new_cross_domain(augmentation=True, model_type=1, data_type=RANGE_ANGLE_IMAGE, domain=3,
                     train_index=[1],
                     test_index=[0, 2, 3, 4],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     need_test=True,
                     multistream=True, diff=True,
                     attention=True, epoch=200, data_spliter=di_DataSplitter)
    # =======

    new_cross_domain(augmentation=False, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=1,
                     train_index=[0, 1, 2, 3, 4],
                     test_index=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=100, data_spliter=di_DataSplitter)

    new_cross_domain(augmentation=False, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=2,
                     train_index=[0],
                     test_index=[1, 2, 3, 4, 5],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=100, data_spliter=di_DataSplitter)
    new_cross_domain(augmentation=False, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=2,
                     train_index=[5],
                     test_index=[0, 1, 2, 3, 4],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=100, data_spliter=di_DataSplitter)
    new_cross_domain(augmentation=False, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=3,
                     train_index=[1],
                     test_index=[0, 2, 3, 4],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     need_test=True,
                     multistream=True, diff=True,
                     attention=True, epoch=100, data_spliter=di_DataSplitter)

    # =======
    new_cross_domain(augmentation=False, model_type=0, data_type=SINGLE_RANGE_DOPPLER, domain=1,
                     train_index=[0, 1, 2, 3, 4],
                     test_index=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=100, data_spliter=di_DataSplitter)

    new_cross_domain(augmentation=False, model_type=0, data_type=SINGLE_RANGE_DOPPLER, domain=2,
                     train_index=[0],
                     test_index=[1, 2, 3, 4, 5],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=100, data_spliter=di_DataSplitter)
    new_cross_domain(augmentation=False, model_type=0, data_type=SINGLE_RANGE_DOPPLER, domain=2,
                     train_index=[5],
                     test_index=[0, 1, 2, 3, 4],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=100, data_spliter=di_DataSplitter)
    new_cross_domain(augmentation=False, model_type=0, data_type=SINGLE_RANGE_DOPPLER, domain=3,
                     train_index=[1],
                     test_index=[0, 2, 3, 4],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     need_test=True,
                     multistream=True, diff=True,
                     attention=True, epoch=100, data_spliter=di_DataSplitter)

    # deep_rai
    new_cross_domain(augmentation=True, model_type=2, data_type=RANGE_ANGLE_IMAGE, domain=3,
                     train_index=[1],
                     test_index=[0, 2, 3, 4],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     need_test=True,
                     multistream=True, diff=True,
                     attention=True, epoch=200, data_spliter=di_DataSplitter)
    new_cross_domain(augmentation=True, model_type=2, data_type=RANGE_ANGLE_IMAGE, domain=2,
                     train_index=[5],
                     test_index=[0, 1, 2, 3, 4],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=200, data_spliter=di_DataSplitter)
    new_cross_domain(augmentation=True, model_type=2, data_type=RANGE_ANGLE_IMAGE, domain=2,
                     train_index=[0],
                     test_index=[1, 2, 3, 4, 5],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=200, data_spliter=di_DataSplitter)
    new_cross_domain(augmentation=True, model_type=2, data_type=RANGE_ANGLE_IMAGE, domain=1,
                     train_index=[0, 1, 2, 3, 4],
                     test_index=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=200, data_spliter=di_DataSplitter)
    
    # radar net
    new_cross_domain(augmentation=False, model_type=3, data_type=COMPLEX_RANGE_DOPPLER, domain=3,
                     train_index=[1],
                     test_index=[0, 2, 3, 4],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     need_test=True,
                     multistream=True, diff=True,
                     attention=True, epoch=100, data_spliter=di_DataSplitter)
    new_cross_domain(augmentation=False, model_type=3, data_type=COMPLEX_RANGE_DOPPLER, domain=2,
                     train_index=[5],
                     test_index=[0, 1, 2, 3, 4],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=100, data_spliter=di_DataSplitter)
    new_cross_domain(augmentation=False, model_type=3, data_type=COMPLEX_RANGE_DOPPLER, domain=2,
                     train_index=[0],
                     test_index=[1, 2, 3, 4, 5],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=100, data_spliter=di_DataSplitter)
    new_cross_domain(augmentation=False, model_type=3, data_type=COMPLEX_RANGE_DOPPLER, domain=1,
                     train_index=[0, 1, 2, 3, 4],
                     test_index=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=100, data_spliter=di_DataSplitter)

    new_cube_k_fold(augmentation=True, epoch=200, start_epoch=0, domain=0, data_type=RANGE_ANGLE_IMAGE, batch_size=128,
                    model_type=0, train_manager=None, data_spliter=di_DataSplitter)
    new_cube_k_fold(augmentation=True, epoch=200, start_epoch=0, domain=0, data_type=RANGE_ANGLE_IMAGE, batch_size=128,
                    model_type=1, train_manager=None, data_spliter=di_DataSplitter)
    
    new_cube_k_fold(augmentation=True, epoch=200, start_epoch=0, domain=0, data_type=CROPPED_RANGE_DOPPLER_IMAGER, batch_size=128,
                    model_type=4, train_manager=None, data_spliter=di_DataSplitter)
    new_cube_k_fold(augmentation=False, epoch=100, start_epoch=0, domain=0, data_type=COMPLEX_RANGE_DOPPLER, batch_size=128,
                    model_type=3, train_manager=None, data_spliter=di_DataSplitter)
    new_cube_k_fold(augmentation=True, epoch=200, start_epoch=0, domain=0, data_type=TIME_FREQUENCY_IMAGE, batch_size=128,
                    model_type=6, train_manager=None, data_spliter=di_DataSplitter)
    
    # deep_rai_small
    new_cross_domain(augmentation=True, model_type=1, data_type=RANGE_ANGLE_IMAGE, domain=2,
                     train_index=[0],
                     test_index=[1, 2, 3, 4, 5],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=200, data_spliter=di_DataSplitter)
    new_cross_domain(augmentation=True, model_type=1, data_type=RANGE_ANGLE_IMAGE, domain=2,
                     train_index=[5],
                     test_index=[0, 1, 2, 3, 4],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=200, data_spliter=di_DataSplitter)
    new_cross_domain(augmentation=True, model_type=1, data_type=RANGE_ANGLE_IMAGE, domain=3,
                     train_index=[1],
                     test_index=[0, 2, 3, 4],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     need_test=True,
                     multistream=True, diff=True,
                     attention=True, epoch=200, data_spliter=di_DataSplitter)
    new_cross_domain(augmentation=True, model_type=1, data_type=RANGE_ANGLE_IMAGE, domain=1,
                     train_index=[0, 1, 2, 3, 4],
                     test_index=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=200, data_spliter=di_DataSplitter)
    # deep_rai_rdi
    new_cross_domain(augmentation=False, model_type=0, data_type=SINGLE_RANGE_DOPPLER, domain=2,
                     train_index=[0],
                     test_index=[1, 2, 3, 4, 5],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=100, data_spliter=di_DataSplitter)
    new_cross_domain(augmentation=False, model_type=0, data_type=SINGLE_RANGE_DOPPLER, domain=2,
                     train_index=[5],
                     test_index=[0, 1, 2, 3, 4],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=100, data_spliter=di_DataSplitter)
    new_cross_domain(augmentation=False, model_type=0, data_type=SINGLE_RANGE_DOPPLER, domain=3,
                     train_index=[1],
                     test_index=[0, 2, 3, 4],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     need_test=True,
                     multistream=True, diff=True,
                     attention=True, epoch=100, data_spliter=di_DataSplitter)
    new_cross_domain(augmentation=False, model_type=1, data_type=SINGLE_RANGE_DOPPLER, domain=1,
                     train_index=[0, 1, 2, 3, 4],
                     test_index=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=100, data_spliter=di_DataSplitter)
    # deep_rai_no_aug
    new_cross_domain(augmentation=False, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=2,
                     train_index=[0],
                     test_index=[1, 2, 3, 4, 5],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=100, data_spliter=di_DataSplitter)
    new_cross_domain(augmentation=False, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=2,
                     train_index=[5],
                     test_index=[0, 1, 2, 3, 4],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=100, data_spliter=di_DataSplitter)
    new_cross_domain(augmentation=False, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=3,
                     train_index=[1],
                     test_index=[0, 2, 3, 4],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     need_test=True,
                     multistream=True, diff=True,
                     attention=True, epoch=100, data_spliter=di_DataSplitter)
    new_cross_domain(augmentation=False, model_type=1, data_type=RANGE_ANGLE_IMAGE, domain=1,
                     train_index=[0, 1, 2, 3, 4],
                     test_index=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=100, data_spliter=di_DataSplitter)

    new_cross_domain(augmentation=True, model_type=4, data_type=CROPPED_RANGE_ANGLE_IMAGER, domain=3,
                     train_index=[1],
                     test_index=[0, 2, 3, 4],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     need_test=True,
                     multistream=True, diff=True,
                     attention=True, epoch=200, data_spliter=di_DataSplitter)

    new_cross_domain(augmentation=True, model_type=4, data_type=CROPPED_RANGE_ANGLE_IMAGER, domain=1,
                     train_index=[0, 1, 2, 3, 4],
                     test_index=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=200, data_spliter=di_DataSplitter)
    new_cross_domain(augmentation=True, model_type=4, data_type=CROPPED_RANGE_ANGLE_IMAGER, domain=2,
                     train_index=[0],
                     test_index=[1, 2, 3, 4, 5],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=200, data_spliter=di_DataSplitter)
    new_cross_domain(augmentation=True, model_type=4, data_type=CROPPED_RANGE_ANGLE_IMAGER, domain=2,
                     train_index=[5],
                     test_index=[0, 1, 2, 3, 4],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=200, data_spliter=di_DataSplitter)

    new_cross_domain(augmentation=True, model_type=1, data_type=RANGE_ANGLE_IMAGE, domain=2,
                     train_index=[5],
                     test_index=[0, 1, 2, 3, 4],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=200, data_spliter=di_DataSplitter)

    new_cross_domain(augmentation=False, model_type=3, data_type=COMPLEX_RANGE_DOPPLER, domain=1,
                     train_index=[0, 1, 2, 3, 4],
                     test_index=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=100, data_spliter=di_DataSplitter)
    
    new_cross_domain(augmentation=False, model_type=3, data_type=COMPLEX_RANGE_DOPPLER, domain=2,
                     train_index=[5],
                     test_index=[0, 1, 2, 3, 4],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=100, data_spliter=di_DataSplitter)

    new_cross_domain(augmentation=False, model_type=3, data_type=COMPLEX_RANGE_DOPPLER, domain=2,
                     train_index=[0],
                     test_index=[1, 2, 3, 4, 5],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=100, data_spliter=di_DataSplitter)
    '''