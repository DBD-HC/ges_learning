from data.mcd_dataset import MCDDataSplitter
from train import TIME_FREQUENCY_IMAGE, RANGE_ANGLE_IMAGE, cross_domain, SINGLE_RANGE_DOPPLER, \
    CROPPED_RANGE_DOPPLER_IMAGER, COMPLEX_RANGE_DOPPLER, k_fold, train_for_real_time

if __name__ == '__main__':
    di_DataSplitter = MCDDataSplitter()

    k_fold(augmentation=True, epoch=200, start_epoch=0, domain=0, data_type=RANGE_ANGLE_IMAGE, batch_size=128,
           model_type=0, train_manager=None, data_spliter=di_DataSplitter)
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
    # 1:user
    # 2:environment
    # 3:location

    # cross user
    cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=1,
                 train_index=[0, 1, 2, 3, 4],
                 test_index=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                 # val_index=[5, 6],
                 # test_index=[3, 4, 5, 6, 7, 8, 9],
                 val_time=5,
                 multistream=True, diff=True,
                 attention=True, epoch=200, data_spliter=di_DataSplitter)
    # cross environment
    cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=2,
                     train_index=[0],
                     test_index=[1, 2, 3, 4, 5],
                     val_time=5,
                     epoch=200, data_spliter=di_DataSplitter)
    # cross environment
    cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=2,
                     train_index=[5],
                     test_index=[0, 1, 2, 3, 4],
                     val_time=5,
                     epoch=200, data_spliter=di_DataSplitter)

    # cross location
    cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=3,
                 train_index=[1],
                 test_index=[0, 2, 3, 4],
                 val_time=5,
                 need_test=True,
                 epoch=200, data_spliter=di_DataSplitter)