from data.rai_ges_dataset import RAIGesDataSplitter
from train import TIME_FREQUENCY_IMAGE, RANGE_ANGLE_IMAGE, cross_domain, train_transferring, k_fold, \
    train_for_real_time

if __name__ == '__main__':
    complex_DataSplitter = RAIGesDataSplitter()
    # model_type:model_name:data_type
    # 0:deep_rai:RANGE_ANGLE_IMAGE
    # 1:deep_rai_small:RANGE_ANGLE_IMAGE
    # 2:di_gesture:RANGE_ANGLE_IMAGE
    # 3:radar_net:COMPLEX_RANGE_DOPPLER
    # 4:rf_dual:CROPPED_RANGE_DOPPLER_IMAGER
    # 5:res_net:TIME_FREQUENCY_IMAGE
    # 6:mobile_net_v3:TIME_FREQUENCY_IMAGE

    # domain NO.: domain
    # 0:in domain
    # 1:environment
    # 2:location
    # 3:user

    # in domain
    k_fold(augmentation=True, epoch=200, start_epoch=0, domain=0, data_type=RANGE_ANGLE_IMAGE, batch_size=128,
           model_type=0, train_manager=None, data_spliter=complex_DataSplitter)
    # cross user
    cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=3,
                 train_index=[0, 6, 7],
                 test_index=[1, 2, 3, 4, 5, 8, 9],
                 val_time=5,
                 epoch=200, data_spliter=complex_DataSplitter)
    # cross environment
    cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=1,
                 train_index=[0],
                 test_index=[1, 2, 3],
                 val_time=5,
                 need_test=True,
                 epoch=200, data_spliter=complex_DataSplitter)
    # cross location
    cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=2,
                 train_index=[1],
                 test_index=[0, 2, 3, 4],
                 val_time=5,
                 need_test=True,
                 epoch=200, data_spliter=complex_DataSplitter)
    # transfer learning cross environment
    train_transferring(augmentation=True, domain=1, model_type=0, data_type=RANGE_ANGLE_IMAGE,
                       train_index=[0],
                       test_index=[1, 2, 3],
                       need_test=True,
                       pre_train_lr=0.001, pre_train_epoch=50, train_epoch=100)
    # transfer learning cross location
    train_transferring(augmentation=True, domain=2, model_type=0, data_type=RANGE_ANGLE_IMAGE,
                       train_index=[1],
                       test_index=[0, 2, 3, 4],
                       need_test=True,
                       pre_train_lr=0.001, pre_train_epoch=50, train_epoch=100)
    # transfer learning cross user
    train_transferring(augmentation=True, domain=3, model_type=0, data_type=RANGE_ANGLE_IMAGE,
                       train_index=[0, 6, 7],
                       test_index=[1, 2, 3, 4, 5, 8, 9],
                       pre_train_lr=0.001, pre_train_epoch=50, train_epoch=100)

    train_for_real_time(model_type=0)
    train_for_real_time(model_type=1)
    train_for_real_time(model_type=2)
