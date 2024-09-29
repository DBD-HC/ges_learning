from data.rai_ges_dataset import RAIGesDataSplitter
from train import TIME_RANGE_ANGLE_IMAGE, RANGE_ANGLE_IMAGE, CROPPED_RANGE_ANGLE_IMAGER, cross_domain, train_transferring, k_fold, \
    train_for_real_time


def train_split(augmentation=True, epoch=200, dataset_splitter=None, model_type=0, train_index=[0, 1, 2, 3, 4], test_index=[5, 6, 7], domain=1, need_test=False, data_type=RANGE_ANGLE_IMAGE, multistream=True, diff=True,attention=True,):
    for i, v in enumerate(train_index):
        temp_train = [v]
        temp_test = test_index + [t_v for t_i, t_v in enumerate(train_index) if t_v not in temp_train]
        cross_domain(augmentation=augmentation, model_type=model_type, data_type=data_type, domain=domain,
                     train_index=temp_train,
                     test_index=temp_test,
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=1,
                     need_test=need_test,
                     multistream=multistream, diff=diff,
                     attention=attention, epoch=epoch, data_spliter=dataset_splitter)

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
    '''
    train_split(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=3,
                 train_index=[0, 1, 2],
                 test_index=[3, 4, 5, 6, 7, 8, 9],
                 # val_index=[5, 6],
                 # test_index=[3, 4, 5, 6, 7, 8, 9],
                 multistream=True, diff=True,
                 attention=True, epoch=200, dataset_splitter=complex_DataSplitter)
    train_split(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=3,
                 train_index=[0, 1, 2],
                 test_index=[3, 4, 5, 6, 7, 8, 9],
                 # val_index=[5, 6],
                 # test_index=[3, 4, 5, 6, 7, 8, 9],
                 multistream=False, diff=True,
                 attention=True, epoch=200, dataset_splitter=complex_DataSplitter)
    train_split(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=3,
                 train_index=[0, 1, 2],
                 test_index=[3, 4, 5, 6, 7, 8, 9],
                 # val_index=[5, 6],
                 # test_index=[3, 4, 5, 6, 7, 8, 9],
                 multistream=True, diff=False,
                 attention=True, epoch=200, dataset_splitter=complex_DataSplitter)


    train_split(augmentation=True, model_type=2, data_type=RANGE_ANGLE_IMAGE, domain=3,
                 train_index=[0, 1, 2],
                 test_index=[3, 4, 5, 6, 7, 8, 9],
                 # val_index=[5, 6],
                 # test_index=[3, 4, 5, 6, 7, 8, 9],
                 multistream=True, diff=True,
                 attention=True, epoch=200, dataset_splitter=complex_DataSplitter)

    train_split(augmentation=False, model_type=4, data_type=CROPPED_RANGE_ANGLE_IMAGER, domain=3,
                 train_index=[0, 1, 2],
                 test_index=[3, 4, 5, 6, 7, 8, 9],
                 # val_index=[5, 6],
                 # test_index=[3, 4, 5, 6, 7, 8, 9],
                 multistream=True, diff=True,
                 attention=True, epoch=100, dataset_splitter=complex_DataSplitter)
    train_split(augmentation=False, model_type=6, data_type=TIME_RANGE_ANGLE_IMAGE, domain=3,
                 train_index=[0, 1, 2],
                 test_index=[3, 4, 5, 6, 7, 8, 9],
                 # val_index=[5, 6],
                 # test_index=[3, 4, 5, 6, 7, 8, 9],
                 multistream=True, diff=True,
                 attention=True, epoch=100, dataset_splitter=complex_DataSplitter)
    # locations
    cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=2,
                 train_index=[1],
                 test_index=[0, 2, 3, 4],
                 val_time=5,
                 need_test=True,
                 epoch=200, data_spliter=complex_DataSplitter)
    cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=2,
                 train_index=[1],
                 test_index=[0, 2, 3, 4],
                 val_time=5,
                 need_test=True, multistream=False, diff=True,
                 epoch=200, data_spliter=complex_DataSplitter)
    cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=2,
                 train_index=[1],
                 test_index=[0, 2, 3, 4],
                 val_time=5,
                 need_test=True, multistream=True, diff=False,
                 epoch=200, data_spliter=complex_DataSplitter)
    cross_domain(augmentation=True, model_type=2, data_type=RANGE_ANGLE_IMAGE, domain=2,
                 train_index=[1],
                 test_index=[0, 2, 3, 4],
                 val_time=5,
                 need_test=True,
                 epoch=200, data_spliter=complex_DataSplitter)
    cross_domain(augmentation=False, model_type=4, data_type=CROPPED_RANGE_ANGLE_IMAGER, domain=2,
                 train_index=[1],
                 test_index=[0, 2, 3, 4],
                 val_time=5,
                 need_test=True,
                 epoch=100, data_spliter=complex_DataSplitter)
    '''
    cross_domain(augmentation=False, model_type=6, data_type=TIME_RANGE_ANGLE_IMAGE, domain=2,
                 train_index=[1],
                 test_index=[0, 2, 3, 4],
                 val_time=5,
                 need_test=True,
                 epoch=100, data_spliter=complex_DataSplitter)
    cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=1,
                 train_index=[0],
                 test_index=[1, 2, 3],
                 val_time=5,
                 need_test=True,
                 epoch=200, data_spliter=complex_DataSplitter)
    cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=1,
                 train_index=[0],
                 test_index=[1, 2, 3],
                 val_time=5,
                 need_test=True, multistream=False,
                 epoch=200, data_spliter=complex_DataSplitter)
    cross_domain(augmentation=True, model_type=2, data_type=RANGE_ANGLE_IMAGE, domain=1,
                 train_index=[0],
                 test_index=[1, 2, 3],
                 val_time=5,
                 need_test=True, diff=False,
                 epoch=200, data_spliter=complex_DataSplitter)
    cross_domain(augmentation=True, model_type=2, data_type=RANGE_ANGLE_IMAGE, domain=1,
                 train_index=[0],
                 test_index=[1, 2, 3],
                 val_time=5,
                 need_test=True,
                 epoch=200, data_spliter=complex_DataSplitter)
    cross_domain(augmentation=False, model_type=4, data_type=CROPPED_RANGE_ANGLE_IMAGER, domain=1,
                 train_index=[0],
                 test_index=[1, 2, 3],
                 val_time=5,
                 need_test=True,
                 epoch=100, data_spliter=complex_DataSplitter)
    cross_domain(augmentation=False, model_type=6, data_type=TIME_RANGE_ANGLE_IMAGE, domain=1,
                 train_index=[0],
                 test_index=[1, 2, 3],
                 val_time=5,
                 need_test=True,
                 epoch=100, data_spliter=complex_DataSplitter)

    '''
    # cross user
    cross_domain(augmentation=True, model_type=4, data_type=CROPPED_RANGE_ANGLE_IMAGER, domain=3,
                 train_index=[0, 6, 7],
                 test_index=[1, 2, 3, 4, 5, 8, 9],
                 val_time=5,
                 epoch=200, data_spliter=complex_DataSplitter)
    
    # cross environment
    cross_domain(augmentation=True, model_type=4, data_type=CROPPED_RANGE_ANGLE_IMAGER, domain=1,
                 train_index=[0],
                 test_index=[1, 2, 3],
                 val_time=5,
                 need_test=True,
                 epoch=200, data_spliter=complex_DataSplitter)
    # cross location
    cross_domain(augmentation=True, model_type=4, data_type=CROPPED_RANGE_ANGLE_IMAGER, domain=2,
                 train_index=[1],
                 test_index=[0, 2, 3, 4],
                 val_time=5,
                 need_test=True,
                 epoch=200, data_spliter=complex_DataSplitter)

    # in domain
    k_fold(augmentation=True, epoch=200, start_epoch=0, domain=0, data_type=CROPPED_RANGE_ANGLE_IMAGER, batch_size=128,
           model_type=4, train_manager=None, data_spliter=complex_DataSplitter)
    

    
    # one domain
    cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=4,
                 train_index=None,
                 test_index=None,
                 val_time=5, need_test=True,
                 epoch=200, data_spliter=complex_DataSplitter)
    # cross user
    cross_domain(augmentation=True, model_type=2, data_type=RANGE_ANGLE_IMAGE, domain=3,
                 train_index=[0, 6, 7],
                 test_index=[1, 2, 3, 4, 5, 8, 9],
                 val_time=5,
                 epoch=200, data_spliter=complex_DataSplitter)
    # cross environment
    cross_domain(augmentation=True, model_type=2, data_type=RANGE_ANGLE_IMAGE, domain=1,
                 train_index=[0],
                 test_index=[1, 2, 3],
                 val_time=5,
                 need_test=True,
                 epoch=200, data_spliter=complex_DataSplitter)
    # cross location
    cross_domain(augmentation=True, model_type=2, data_type=RANGE_ANGLE_IMAGE, domain=2,
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

    '''
