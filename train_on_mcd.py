from data.mcd_dataset import MCDDataSplitter
from train import TIME_RANGE_ANGLE_IMAGE, TIME_RANGE_DOPPLER_IMAGE, RANGE_ANGLE_IMAGE, cross_domain, SINGLE_RANGE_DOPPLER, \
    CROPPED_RANGE_DOPPLER_IMAGER, COMPLEX_RANGE_DOPPLER, k_fold, train_for_real_time

def train_split(augmentation=True, epoch=200, dataset_splitter=None, model_type=0, train_index=[0, 1, 2, 3, 4], test_index=[5, 6, 7], domain=1, need_test=False, data_type=RANGE_ANGLE_IMAGE, multistream=True, diff=True,attention=True,):
    for i in train_index:
        temp_train = [train_index[i]]
        temp_test = test_index + [v for i,v in enumerate(train_index) if i not in temp_train]
        cross_domain(augmentation=augmentation, model_type=model_type, data_type=data_type, domain=domain,
                     train_index=temp_train,
                     test_index=temp_test,
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=1,
                     need_test=need_test,
                     multistream=True, diff=True,
                     attention=True, epoch=epoch, data_spliter=dataset_splitter)

if __name__ == '__main__':
    di_DataSplitter = MCDDataSplitter()


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
    # 4:only one domain
    '''
    train_split(augmentation=True, model_type=2, data_type=RANGE_ANGLE_IMAGE, domain=1,
                 train_index=[0, 1, 2, 3, 4],
                 test_index=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                 # val_index=[5, 6],
                 # test_index=[3, 4, 5, 6, 7, 8, 9],
                 multistream=True, diff=True,
                 attention=True, epoch=200, dataset_splitter=di_DataSplitter)
    train_split(augmentation=False, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=1,
                 train_index=[0, 1, 2, 3, 4],
                 test_index=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                 # val_index=[5, 6],
                 # test_index=[3, 4, 5, 6, 7, 8, 9],
                 multistream=True, diff=True,
                 attention=True, epoch=100, dataset_splitter=di_DataSplitter)
    train_split(augmentation=False, model_type=0, data_type=SINGLE_RANGE_DOPPLER, domain=1,
                 train_index=[0, 1, 2, 3, 4],
                 test_index=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                 # val_index=[5, 6],
                 # test_index=[3, 4, 5, 6, 7, 8, 9],
                 multistream=True, diff=True,
                 attention=True, epoch=100, dataset_splitter=di_DataSplitter)

    train_split(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=1,
                 train_index=[0, 1, 2, 3, 4],
                 test_index=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                 # val_index=[5, 6],
                 # test_index=[3, 4, 5, 6, 7, 8, 9],
                 multistream=True, diff=True,
                 attention=True, epoch=200, dataset_splitter=di_DataSplitter)
    train_split(augmentation=False, model_type=0, data_type=SINGLE_RANGE_DOPPLER, domain=1,
                 train_index=[0, 1, 2, 3, 4],
                 test_index=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                 # val_index=[5, 6],
                 # test_index=[3, 4, 5, 6, 7, 8, 9],
                 multistream=True, diff=True,
                 attention=True, epoch=100, dataset_splitter=di_DataSplitter)
    train_split(augmentation=False, model_type=3, data_type=COMPLEX_RANGE_DOPPLER, domain=1,
                 train_index=[0, 1, 2, 3, 4],
                 test_index=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                 # val_index=[5, 6],
                 # test_index=[3, 4, 5, 6, 7, 8, 9],
                 multistream=True, diff=True,
                 attention=True, epoch=100, dataset_splitter=di_DataSplitter)

    train_split(augmentation=False, model_type=4, data_type=CROPPED_RANGE_DOPPLER_IMAGER, domain=1,
                 train_index=[0, 1, 2, 3, 4],
                 test_index=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                 # val_index=[5, 6],
                 # test_index=[3, 4, 5, 6, 7, 8, 9],
                 multistream=True, diff=True,
                 attention=True, epoch=100, dataset_splitter=di_DataSplitter)
    train_split(augmentation=False, model_type=6, data_type=TIME_RANGE_ANGLE_IMAGE, domain=1,
                 train_index=[0, 1, 2, 3, 4],
                 test_index=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                 # val_index=[5, 6],
                 # test_index=[3, 4, 5, 6, 7, 8, 9],
                 multistream=True, diff=True,
                 attention=True, epoch=100, dataset_splitter=di_DataSplitter)

    # cross locations
    cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=3,
                 train_index=[1],
                 test_index=[0, 2, 3, 4],
                 val_time=5,
                 need_test=True,
                 epoch=200, data_spliter=di_DataSplitter)
    cross_domain(augmentation=False, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=3,
                 train_index=[1],
                 test_index=[0, 2, 3, 4],
                 val_time=5,
                 need_test=True,
                 epoch=100, data_spliter=di_DataSplitter)
    cross_domain(augmentation=False, model_type=0, data_type=SINGLE_RANGE_DOPPLER, domain=3,
                 train_index=[1],
                 test_index=[0, 2, 3, 4],
                 val_time=5,
                 need_test=True,
                 epoch=100, data_spliter=di_DataSplitter)
    cross_domain(augmentation=True, model_type=2, data_type=RANGE_ANGLE_IMAGE, domain=3,
                 train_index=[1],
                 test_index=[0, 2, 3, 4],
                 val_time=5,
                 need_test=True,
                 epoch=200, data_spliter=di_DataSplitter)

    cross_domain(augmentation=True, model_type=3, data_type=COMPLEX_RANGE_DOPPLER, domain=3,
                 train_index=[1],
                 test_index=[0, 2, 3, 4],
                 val_time=5,
                 need_test=True,
                 epoch=200, data_spliter=di_DataSplitter)
    cross_domain(augmentation=False, model_type=4, data_type=CROPPED_RANGE_DOPPLER_IMAGER, domain=3,
                 train_index=[1],
                 test_index=[0, 2, 3, 4],
                 val_time=5,
                 need_test=True,
                 epoch=100, data_spliter=di_DataSplitter)
    cross_domain(augmentation=False, model_type=6, data_type=TIME_RANGE_ANGLE_IMAGE, domain=3,
                 train_index=[1],
                 test_index=[0, 2, 3, 4],
                 val_time=5,
                 need_test=True,
                 epoch=100, data_spliter=di_DataSplitter)

    cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=2,
                 train_index=[5],
                 test_index=[0, 1, 2, 3, 4],
                 val_time=5,
                 epoch=200, data_spliter=di_DataSplitter)
    # cross environment
    cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=2,
                 train_index=[0],
                 test_index=[1, 2, 3, 4, 5],
                 val_time=5,
                 epoch=200, data_spliter=di_DataSplitter)
    cross_domain(augmentation=False, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=2,
                 train_index=[5],
                 test_index=[0, 1, 2, 3, 4],
                 val_time=5,
                 epoch=100, data_spliter=di_DataSplitter)
    # cross environment
    cross_domain(augmentation=False, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=2,
                 train_index=[0],
                 test_index=[1, 2, 3, 4, 5],
                 val_time=5,
                 epoch=100, data_spliter=di_DataSplitter)
    cross_domain(augmentation=False, model_type=0, data_type=SINGLE_RANGE_DOPPLER, domain=2,
                 train_index=[5],
                 test_index=[0, 1, 2, 3, 4],
                 val_time=5,
                 epoch=100, data_spliter=di_DataSplitter)
    # cross environment
    cross_domain(augmentation=False, model_type=0, data_type=SINGLE_RANGE_DOPPLER, domain=2,
                 train_index=[0],
                 test_index=[1, 2, 3, 4, 5],
                 val_time=5,
                 epoch=100, data_spliter=di_DataSplitter)

    cross_domain(augmentation=True, model_type=2, data_type=RANGE_ANGLE_IMAGE, domain=2,
                 train_index=[5],
                 test_index=[0, 1, 2, 3, 4],
                 val_time=5,
                 epoch=200, data_spliter=di_DataSplitter)
    # cross environment
    cross_domain(augmentation=True, model_type=2, data_type=RANGE_ANGLE_IMAGE, domain=2,
                 train_index=[0],
                 test_index=[1, 2, 3, 4, 5],
                 val_time=5,
                 epoch=200, data_spliter=di_DataSplitter)
    # cross environment
    cross_domain(augmentation=False, model_type=4, data_type=CROPPED_RANGE_DOPPLER_IMAGER, domain=2,
                 train_index=[0],
                 test_index=[1, 2, 3, 4, 5],
                 val_time=5,
                 epoch=100, data_spliter=di_DataSplitter)
    cross_domain(augmentation=False, model_type=4, data_type=CROPPED_RANGE_DOPPLER_IMAGER, domain=2,
                 train_index=[5],
                 test_index=[0, 1, 2, 3, 4],
                 val_time=5,
                 epoch=100, data_spliter=di_DataSplitter)

    cross_domain(augmentation=False, model_type=6, data_type=TIME_RANGE_ANGLE_IMAGE, domain=2,
                 train_index=[5],
                 test_index=[0, 1, 2, 3, 4],
                 val_time=5,
                 epoch=100, data_spliter=di_DataSplitter)
    # cross environment
    cross_domain(augmentation=False, model_type=6, data_type=TIME_RANGE_ANGLE_IMAGE, domain=2,
                 train_index=[0],
                 test_index=[1, 2, 3, 4, 5],
                 val_time=5,
                 epoch=100, data_spliter=di_DataSplitter)
    '''
    cross_domain(augmentation=False, model_type=3, data_type=COMPLEX_RANGE_DOPPLER, domain=2,
                 train_index=[5],
                 test_index=[0, 1, 2, 3, 4],
                 val_time=5,
                 epoch=100, data_spliter=di_DataSplitter)
    # cross environment
    cross_domain(augmentation=False, model_type=3, data_type=COMPLEX_RANGE_DOPPLER, domain=2,
                 train_index=[0],
                 test_index=[1, 2, 3, 4, 5],
                 val_time=5,
                 epoch=100, data_spliter=di_DataSplitter)



    '''
    # cross environment
    cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=2,
                 train_index=[5],
                 test_index=[0, 1, 2, 3, 4],
                 val_time=5,
                 epoch=200, data_spliter=di_DataSplitter)
    # cross environment
    cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=2,
                 train_index=[0],
                 test_index=[1, 2, 3, 4, 5],
                 val_time=5,
                 epoch=200, data_spliter=di_DataSplitter)

    cross_domain(augmentation=True, model_type=2, data_type=RANGE_ANGLE_IMAGE, domain=4,
                 train_index=None,
                 test_index=None,
                 val_time=5, need_test=True,
                 epoch=200, data_spliter=di_DataSplitter)

    # cross location
    cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=3,
                 train_index=[1],
                 test_index=[0, 2, 3, 4],
                 val_time=5,
                 need_test=True,
                 epoch=200, data_spliter=di_DataSplitter)


    # cross user
    cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=1,
                 train_index=[0, 1, 2, 3, 4],
                 test_index=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                 # val_index=[5, 6],
                 # test_index=[3, 4, 5, 6, 7, 8, 9],
                 val_time=5,
                 multistream=True, diff=True,
                 attention=True, epoch=200, data_spliter=di_DataSplitter)
    '''