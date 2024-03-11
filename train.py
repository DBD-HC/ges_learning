import subprocess

import psutil
import torch
import torch.optim as optim
from torchvision.transforms import transforms

from data.complex_rai_dataset import ComplexDataSplitter
from log_helper import LogHelper

from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC, MulticlassAveragePrecision
from torchmetrics.functional import accuracy
import os
from sklearn.metrics import confusion_matrix

from model.compare_methods import DiGesture, RadarNet, DeepSolid, Resnet50Classifier, MobilenetV350Classifier
from data.di_gesture_dataset import *
from model.network import *
from data.cubelern_arm_dataset import *
from model.vae import VAE


def seed_worker(worker_id):
    seed = worker_id + random_seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


# 定义随机种子固定的函数
def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_confusion_matrix(best_ture_label, best_predict_label, title):
    cm = confusion_matrix(best_ture_label, best_predict_label)
    vis.heatmap(
        X=cm,
        opts=dict(
            columnnames=ges_label,
            rownames=ges_label,
            xlabel='Predicted',
            ylabel='True',
            text=True,
            colormap='Greys',
            title='Confusion Matrix ' + title
        )
    )


def unpack_run_model(model, pack):
    datas = pack[0].to(device)
    labels = pack[1].to(device)
    if len(pack) > 2:
        datalens = pack[3].to(device)
        if isinstance(model, RAIRadarGestureClassifier):
            tracks = pack[2].to(device)
            outputs = model(datas, tracks, datalens)
        else:
            outputs = model(datas, datalens)
    else:
        outputs = model(datas)

    return outputs, labels


def get_correct_num(outputs, labels):
    prediction = torch.argmax(outputs, 1)
    return (prediction == labels).sum().float().item()


def train(model, dataloader, epoch, lr, optimizer):
    running_loss = 0.0
    total_sample = 0.0
    correct_sample = 0.0
    model.train()
    for i, pack in enumerate(dataloader):
        optimizer.zero_grad()
        outputs, labels = unpack_run_model(model, pack)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # calculation of accuracy
        total_sample += labels.size(0)
        correct_sample += get_correct_num(outputs, labels)
        if i % 5 == 4:  # print every 5 mini-batches
            print('Training [%d, %5d] loss: %.3f, accuracy: %.5f' % (
                epoch + 1, i + 1, loss / (i + 1), correct_sample / total_sample))
    train_acc = correct_sample / total_sample
    print(
        '[Train] lr:%.6f, all_samples: %.5f, correct_samples: %.5f,  loss: %.5f, accuracy: %.5f' % (lr,
                                                                                                    total_sample,
                                                                                                    correct_sample,
                                                                                                    running_loss / len(
                                                                                                        dataloader),
                                                                                                    train_acc))
    return train_acc, running_loss / len(dataloader)


def validate(model, dataloader):
    model.eval()
    running_loss = 0.0
    total_sample = 0.0
    correct_sample = 0.0
    ture_label = []
    predict_label = []
    with torch.no_grad():
        for i, pack in enumerate(dataloader):
            outputs, labels = unpack_run_model(model, pack)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            # calculation of accuracy
            total_sample += labels.size(0)
            correct_sample += get_correct_num(outputs, labels)
            prediction = torch.argmax(outputs, 1)
            predict_label.extend([x.item() for x in prediction])
            ture_label.extend([x.item() for x in labels])
            multiclassAUROC.update(outputs, labels)
            multiclassAveragePrecision.update(outputs, labels)
    val_acc = correct_sample / total_sample
    val_auc = multiclassAUROC.compute()

    val_micro_acc = accuracy(torch.tensor(predict_label), torch.tensor(ture_label), num_classes=len(ges_label),
                             average='macro', task="multiclass")
    val_ap = multiclassAveragePrecision.compute()
    multiclassAveragePrecision.reset()
    multiclassAUROC.reset()
    print('[Validation] all validation: %.5f, correct validation: %.5f' % (total_sample, correct_sample))
    print('[Validation] val loss: %.5f, auc: %.5f, ap: %.5f, macro accuracy %.5f, accuracy: %.5f' % (
        running_loss / len(dataloader), val_auc, val_ap, val_micro_acc, val_acc))

    return val_acc, running_loss / len(dataloader), val_auc.item(), val_ap.item(), predict_label, ture_label


def dynamic_sequence_collate_fn(datas_and_labels):
    datas_and_labels = sorted(datas_and_labels, key=lambda x: x[0].size()[0], reverse=True)
    datas = [item[0] for item in datas_and_labels]
    tracks = torch.stack([item[1] for item in datas_and_labels])
    labels = torch.stack([item[2] for item in datas_and_labels])
    indexes = [item[3] for item in datas_and_labels]
    data_lengths = [len(x) for x in datas]
    datas = pad_sequence(datas, batch_first=True, padding_value=0)
    # datas = trans2xy(datas)
    return datas, labels, tracks, torch.tensor(data_lengths), indexes


def static_unify_sequence(x, data_len, h, w):
    x = x.unsqueeze(0).unsqueeze(0)
    x = F.interpolate(x, size=(data_len, h, w), mode='trilinear', align_corners=False)
    return torch.squeeze(x)


def static_sequence_collate_fn(datas_and_labels, sequence_len=50, h=224, w=224):
    datas_and_labels = sorted(datas_and_labels, key=lambda x: x[0].size()[0], reverse=True)
    datas = torch.stack([static_unify_sequence(item[0], sequence_len, h, w) for item in datas_and_labels])
    labels = torch.stack([item[2] for item in datas_and_labels])
    return datas, labels


def radar_net_unify_sequence(x):
    x = x.unsqueeze(0)
    x = F.interpolate(x, size=(x.size(-3), 24, 16), mode='trilinear', align_corners=False)
    x = torch.normal(mean=1, std=0.025, size=(1,)) * x
    return torch.squeeze(x)


def radar_net_collate_fn(datas_and_labels):
    datas_and_labels = sorted(datas_and_labels, key=lambda x: x[0].size()[0], reverse=True)
    datas = [radar_net_unify_sequence(item[0]) for item in datas_and_labels]
    labels = torch.stack([item[2] for item in datas_and_labels])
    indexes = [item[3] for item in datas_and_labels]
    data_lengths = [len(x) for x in datas]
    datas = pad_sequence(datas, batch_first=True, padding_value=0)
    return datas, labels, None, torch.tensor(data_lengths), indexes


td_transform = transforms.Compose([
    transforms.Resize((224, 224)),
])

def get_time_doppler(rai):
    time_angle = torch.mean(rai, dim=-2)[None, :]
    time_range = torch.mean(rai, dim=-1)[None, :]
    input_data = torch.cat((time_range, time_angle), dim=0)
    input_data = td_transform(input_data)
    return input_data

def time_doppler_collate_fn(datas_and_labels):
    datas = torch.stack([get_time_doppler(item[0]) for item in datas_and_labels])
    labels = torch.stack([item[2] for item in datas_and_labels])
    return datas, labels


def compute_Precision(ture_label, pred_label):
    precision_list = []
    pred_label = np.array(pred_label)
    ture_label = np.array(ture_label)
    for ges_type in range(len(ges_label)):
        mask = pred_label == ges_type
        total_predict = len(pred_label[mask])
        mask2 = ture_label[mask] == ges_type
        ture_predict = len(ture_label[mask][mask2])
        p = ture_predict / total_predict
        precision_list.append(p)
    return precision_list


def train_and_val(model, train_set, val_set, start_epoch, total_epoch, batch_size, lr, title='',
                  collate_fn=dynamic_sequence_collate_fn, model_name="test.pth"):
    best_acc = 0
    best_loss = 0
    best_auc = 0
    best_ap = 0
    best_ture_label = []
    best_predict_label = []

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8,
                                               pin_memory=True,
                                               worker_init_fn=seed_worker,
                                               collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                             num_workers=8,
                                             worker_init_fn=seed_worker,
                                             pin_memory=True,
                                             collate_fn=collate_fn)
    if start_epoch != 0:
        model.load_state_dict(torch.load(model_name)['model_state_dict'])
    model = model.to(device)
    # if vae is not None:
    #     vae = vae.to(device)
    optimizer = optim.Adam([
        {'params': model.parameters()},
    ], lr=lr[0])
    for epoch in range(start_epoch, total_epoch):
        optimizer.param_groups[0]['lr'] = lr[epoch]

        train_acc, train_loss = train(model, train_loader, epoch, lr[epoch], optimizer)
        val_acc, val_loss, val_auc, val_ap, pred_label, ture_label = validate(model, val_loader)

        vis.line(X=np.array([epoch + 1]), Y=np.array([[train_acc, val_acc]]), win='acc',
                 update='append',
                 opts=dict(title='Train/val accuracy', xlabel='Epoch', ylabel='Accuracy',
                           legend=['Train accuracy', 'Val accuracy']))
        vis.line(X=np.array([epoch + 1]), Y=np.array([[train_loss, val_loss]]), win='loss',
                 update='append',
                 opts=dict(title='Train/val loss', xlabel='Epoch', ylabel='Loss', legend=['Train loss', 'Val loss']))
        vis.line(X=np.array([epoch + 1]), Y=np.array([[val_auc, val_ap]]), win='ap',
                 update='append',
                 opts=dict(title='val auc/ap', xlabel='Epoch', ylabel='ap', legend=['Val auc', 'Val ap']))

        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            best_acc = val_acc
            best_loss = val_loss
            best_ap = val_ap
            best_auc = val_auc
            best_ture_label = ture_label
            best_predict_label = pred_label
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss
            }, 'checkpoint/' + model_name)
            torch.save({
                'model_state_dict': model.state_dict(),
            }, 'checkpoint/final_model.pth')
            print('saved')

        print('[Validation] best  auc: %.5f, ap: %.5f, accuracy: %.5f' % (
            best_auc, best_ap, best_acc))
    print(title + ' ' + str(best_acc))
    # precisions = compute_Precision(best_ture_label, best_predict_label)
    # loger.log('precision {}'.format(precisions))
    plot_confusion_matrix(best_ture_label, best_predict_label, title)

    return best_acc, best_auc, best_ap


def cube_k_fold(augmentation=True, domain = 1, model_type=0, need_cfar=True, ra_conv=True, diff=True, track=True, attention=True, epoch=200,
                batch_size=128, start_epoch=0, collate_fn=dynamic_sequence_collate_fn, data_spliter=None, out_size=6, transformer=None):
    # lr_main = get_propsed_lr(augmentation, epoch)
    # complex_clear_cache()
    set_random_seed(random_seed)
    for t in range(data_spliter.get_domain_num(domain)):
        train_set, test_set = data_spliter.split_data(domain, t)
        print('domain{} len{}', domain,train_set.len + test_set.len)
        #if t < 2:
        #    continue
        if model_type == 0:
            model = RAIRadarGestureClassifier(cfar=need_cfar, track=track, spatial_channels=(4, 8, 16), ra_conv=ra_conv,
                                                      heads=4,
                                                      track_channels=(4, 8, 16), track_out_size=32, conv2d_feat_size=64,
                                                      diff=diff, out_size=out_size,
                                                      ra_feat_size=64, attention=attention, cfar_expand_channels=8, in_channel=1)
        elif model_type == 1:
            model = DiGesture(out_size=out_size)
        elif model_type == 2:
            model = RadarNet(out_size=out_size)
        elif model_type == 3:
            model = DeepSolid(out_size=out_size)
        elif model_type == 4:
            model = Resnet50Classifier(out_size=out_size)
        else:
            model = MobilenetV350Classifier(out_size=out_size)

        print(type(model).__name__)
        lr_main = get_propsed_lr(augmentation, epoch)
        # lr_main = get_di_gesture_lr(augmentation, epoch)
        # train_set, test_set = split_data('cross_person', person_index=3)
        #train_set, test_set = cube_split_data(1, user=(6, 7, 8))

        if not augmentation:
            train_set.transform = test_set.transform

        # model_name = 'cross_person_radar_net.pth'
        # model_name = 'cross_person_di_gesture.pth'
        model_name = 'domain{}_model{}_dataset{}_aug_{}_diff_{}_ra_{}_attrack_{}.pth'.format(domain, type(model).__name__, type(data_spliter).__name__, augmentation, diff, ra_conv, track)
        acc, auc, ap = train_and_val(model, train_set, test_set, start_epoch, epoch, batch_size=batch_size,
                                     lr=lr_main, title='in domain', model_name=model_name, collate_fn=collate_fn)
        loger.log(
            'k_fold_domain{}_complex{} model:{} dataset:{} augmentation:{} need_cfar:{} need_diff:{} need_ra:{} att_track:{} acc:{} auc:{} ap:{}'
            .format(domain, t, type(model).__name__, type(data_spliter).__name__, augmentation, need_cfar, diff, ra_conv, track, acc, auc, ap))


def get_propsed_lr(augmentation=True, epoch=200):
    lr_list = np.zeros(epoch)
    lr_cfar = np.zeros(epoch)
    if augmentation:
        # lr_list = np.linspace(0.001, 0.0001, epoch)
        lr_list[:epoch // 2] = 0.001
        lr_list[epoch // 2:epoch // 2 + epoch // 4] = 0.0003
        lr_list[epoch // 2 + epoch // 4:] = 0.0001
        lr_cfar[:epoch // 2] = 0.0001
        lr_cfar[epoch // 2:epoch // 2 + epoch // 4] = 0.00003
        lr_cfar[epoch // 2 + epoch // 4:] = 0.00001
    else:
        lr_list[:epoch] = 0.001
        lr_cfar[:epoch] = 0.00001
    return lr_list


def get_cube_lr(augmentation=True, epoch=200):
    lr_list = np.zeros(epoch)

    lr_list[:epoch] = 0.0003

    return lr_list


def get_di_gesture_lr(augmentation=True, epoch=200):
    lr_list = np.zeros(epoch)
    lr_list[:] = 0.0001

    return lr_list


def get_radar_net_lr(augmentation=True, epoch=200):
    lr_list = np.zeros(epoch)
    lr_list[:] = 0.001

    return lr_list


if __name__ == '__main__':
    loger = LogHelper()
    visdom_port = 6006
    vis = visdom.Visdom(env='model result', port=visdom_port)
    random_seed = 2023

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()
    criterion_recon = nn.BCELoss()

    # ges_label = ['CT', 'AT', 'PH', 'PL', 'LS', 'RS', 'NG']


    #
    # cross_position((3, 5), augmentation=False, ra_conv=True, diff=True, attention=True, track=True, epoch=100)
    # cross_position((0, 5), augmentation=True, ra_conv=True, epoch=200)

    # five_fold_validation(fold_range=(0,5), augmentation=False, need_cfar=False, epoch=100)
    # cross_person(augmentation=False, ra_conv=True, diff=False, epoch=100)
    # cross_person(augmentation=True, ra_conv=True, diff=True, epoch=200)
    # cross_person(augmentation=False, ra_conv=True, diff=False, attention=False, track=False, epoch=100)
    # cross_person(augmentation=False, ra_conv=False, diff=False, attention=False, track=False, epoch=100)
    # cross_person(augmentation=False, ra_conv=True, diff=False, attention=False, track=False, epoch=100)
    # cross_person(augmentation=False, ra_conv=False, diff=True, attention=False, track=False, epoch=100)
    # cross_person(augmentation=False, ra_conv=True, diff=True, attention=True, track=True, epoch=200)

    # cross_environment(env_range=(0, 6), augmentation=False, ra_conv=False, diff=False, attention=False, track=False,
    #                  epoch=200)
    # cross_position((0, 5), augmentation=False, ra_conv=True, diff=True, attention=True, track=True, epoch=200)
    # cube_k_fold(augmentation=True, model_type=0, domain=1, ra_conv=True, diff=True, attention=True, track=True, epoch=200)
    # cube_k_fold(augmentation=True, domain=2, ra_conv=True, diff=True, attention=True, track=True, epoch=200)
    # cube_k_fold(augmentation=True, domain=3, ra_conv=True, diff=True, attention=True, track=True, epoch=200)

    di_DataSplitter = DIDataSplitter()
    complex_DataSplitter = ComplexDataSplitter()
    # cube_k_fold(augmentation=True, model_type=0, domain=0, ra_conv=True, diff=True, attention=True, track=True, epoch=200, data_spliter=complex_DataSplitter, out_size=6)
    # cube_k_fold(augmentation=True, model_type=1, domain=0, ra_conv=True, diff=True, attention=True, track=True, epoch=200, data_spliter=di_DataSplitter)
    # cube_k_fold(augmentation=True, model_type=4, domain=0, ra_conv=True, diff=True, attention=True, track=True, epoch=200, collate_fn=time_doppler_collate_fn, data_spliter=di_DataSplitter)
    # cube_k_fold(augmentation=True, model_type=5, domain=0, ra_conv=True, diff=True, attention=True, track=True, epoch=200, collate_fn=time_doppler_collate_fn, data_spliter=di_DataSplitter)



    #cube_k_fold(augmentation=True, model_type=0, domain=1, ra_conv=True, diff=True, attention=True, track=True, epoch=200, data_spliter=complex_DataSplitter)
    #cube_k_fold(augmentation=True, model_type=0, domain=2, ra_conv=True, diff=True, attention=True, track=True, epoch=200, data_spliter=complex_DataSplitter)
    #cube_k_fold(augmentation=True, model_type=0, domain=3, ra_conv=True, diff=True, attention=True, track=True, epoch=200, data_spliter=complex_DataSplitter)
    # ges_label = ['CT', 'AT', 'PH', 'PL', 'LS', 'RS', 'NG']

    ges_label = ['0', '1', '2', '3', '4', '5']
    # ges_label = cube_gestures
    num_class = len(ges_label)
    multiclassAUROC = MulticlassAUROC(num_classes=num_class, average='macro').to(device)
    multiclassAveragePrecision = MulticlassAveragePrecision(num_classes=num_class, average='macro').to(device)

    #cube_k_fold(augmentation=True, model_type=0, domain=0, ra_conv=True, diff=True, attention=True, track=True,
     #           epoch=200, data_spliter=complex_DataSplitter, out_size=6)
    #cube_k_fold(augmentation=True, model_type=1, domain=0, ra_conv=True, diff=True, attention=True, track=True,
    #            epoch=200, data_spliter=complex_DataSplitter, out_size=6)
    cube_k_fold(augmentation=False, model_type=4, domain=0, ra_conv=True, diff=True, attention=True, track=True,
                epoch=100, data_spliter=complex_DataSplitter, collate_fn=time_doppler_collate_fn, out_size=6)
    cube_k_fold(augmentation=False, model_type=5, domain=0, ra_conv=True, diff=True, attention=True, track=True,
                epoch=100, data_spliter=complex_DataSplitter, collate_fn=time_doppler_collate_fn, out_size=6)

    ges_label = ['CT', 'AT', 'PH', 'PL', 'LS', 'RS', 'NG']
    # ges_label = cube_gestures
    num_class = len(ges_label)
    multiclassAUROC = MulticlassAUROC(num_classes=num_class, average='macro').to(device)
    multiclassAveragePrecision = MulticlassAveragePrecision(num_classes=num_class, average='macro').to(device)

    cube_k_fold(augmentation=True, model_type=0, domain=0, ra_conv=True, diff=True, attention=True, track=True,
                epoch=200, data_spliter=di_DataSplitter, out_size=7)
    cube_k_fold(augmentation=True, model_type=1, domain=0, ra_conv=True, diff=True, attention=True, track=True,
                epoch=200, data_spliter=di_DataSplitter, out_size=7)
    cube_k_fold(augmentation=False, model_type=4, domain=0, ra_conv=True, diff=True, attention=True, track=True,
                epoch=100, data_spliter=di_DataSplitter, collate_fn=time_doppler_collate_fn, out_size=7)
    cube_k_fold(augmentation=False, model_type=5, domain=0, ra_conv=True, diff=True, attention=True, track=True,
                epoch=100, data_spliter=di_DataSplitter, collate_fn=time_doppler_collate_fn, out_size=7)

    # cube_k_fold(augmentation=False, model_type=0, domain=1, ra_conv=True, diff=True, attention=True, track=True, epoch=100, data_spliter=di_DataSplitter)
    # cube_k_fold(augmentation=False, model_type=0, domain=2, ra_conv=True, diff=True, attention=True, track=True, epoch=100, data_spliter=di_DataSplitter)
    # cube_k_fold(augmentation=False, model_type=0, domain=3, ra_conv=True, diff=True, attention=True, track=True, epoch=100, data_spliter=di_DataSplitter)

    # cube_k_fold(augmentation=True, model_type=1, domain=1, ra_conv=True, diff=True, attention=True, track=True, epoch=200, data_spliter=complex_DataSplitter)
    # cube_k_fold(augmentation=True, model_type=1, domain=2, ra_conv=True, diff=True, attention=True, track=True, epoch=200, data_spliter=complex_DataSplitter)
    # cube_k_fold(augmentation=True, model_type=1, domain=3, ra_conv=True, diff=True, attention=True, track=True, epoch=200, data_spliter=complex_DataSplitter)
    # cube_k_fold(augmentation=False, model_type=0, domain=3, ra_conv=True, diff=True, attention=True, track=True, epoch=100)
    # cube_k_fold(augmentation=False, model_type=1, domain=1, ra_conv=True, diff=True, attention=True, track=True,epoch=100)
    # cube_k_fold(augmentation=False, model_type=1, domain=2, ra_conv=True, diff=True, attention=True, track=True,epoch=100)
    # cube_k_fold(augmentation=False, model_type=1, domain=3, ra_conv=True, diff=True, attention=True, track=True,epoch=100)
    # cross_person(augmentation=False, ra_conv=True, diff=True, attention=True, track=False, epoch=100)
    # cross_person(augmentation=False, ra_conv=False, diff=True, attention=True, track=True, epoch=200)
    # cross_person(augmentation=False, ra_conv=True, diff=False, attention=True, track=True, epoch=200)
    # cross_position((0, 5), augmentation=False, ra_conv=True, diff=True, attention=True, track=True, epoch=100)
    #cross_position((0, 5), augmentation=False, ra_conv=True, diff=True, attention=True, track=True, epoch=100)
    # cross_person(augmentation=False, ra_conv=False, diff=False, attention=False, track=False, epoch=100)
    # cross_person(augmentation=False, ra_conv=True, diff=False, attention=True, track=False, epoch=100)
    # cross_person(augmentation=False, ra_conv=False, diff=False, attention=False, track=False, epoch=100)
    # cross_person(augmentation=False, ra_conv=True, diff=True, attention=True, track=True, epoch=100)
    # cross_person(augmentation=False, ra_conv=True, diff=True, attention=True, track=True, epoch=100)
    # cross_person(augmentation=False, ra_conv=True, diff=True, attention=True, track=True, epoch=100)
    # cube_k_fold(augmentation=False, ra_conv=False, diff=False, attention=False, track=False, epoch=100)
    # cross_person(augmentation=False, ra_conv=True, diff=True, attention=False, track=True, epoch=100)
    # cross_person(augmentation=False, ra_conv=False, diff=True, attention=False, track=True, epoch=100)
    # cross_person(augmentation=False, ra_conv=True, diff=False, attention=True, track=True, epoch=100)
    # cross_person(augmentation=False, ra_conv=True, diff=True, attention=True, track=False, epoch=100)
    # cross_position((3, 5), augmentation=False, ra_conv=True, diff=True, attention=True, track=True, epoch=100)

    # cross_position((0, 5), augmentation=False, ra_conv=True, diff=True, attention=True, track=True, epoch=100)

    # cross_environment(env_range=(0, 6), augmentation=False, ra_conv=True, diff=True, attention=True, track=True,
    #                   epoch=100)
    # cross_environment(env_range=(5, 6), augmentation=False, ra_conv=True, diff=True, attention=True, track=True,
    #                   epoch=100)
    # cross_environment(env_range=(0, 6), augmentation=False, ra_conv=True, diff=False, attention=True, track=True,
    #                   epoch=100)
    # cross_environment(env_range=(0, 6), augmentation=False, ra_conv=True, diff=True, attention=True, track=False,
    #                   epoch=100)

    # cross_position((0, 5), augmentation=False, ra_conv=True, diff=True, attention=True, track=True, epoch=100)
    # cross_position((0, 5), augmentation=False, ra_conv=False, diff=True, attention=False, track=True, epoch=100)
    # cross_position((0, 5), augmentation=False, ra_conv=True, diff=False, attention=True, track=True, epoch=100)
    # cross_position((0, 5), augmentation=False, ra_conv=True, diff=True, attention=True, track=False, epoch=100)
    # cross_environment(env_range=(0, 6), augmentation=False, ra_conv=True, diff=True, attention=True, track=False,
    #                   epoch=100)
    # cross_environment(env_range=(0, 6), augmentation=False, ra_conv=True, diff=False, attention=True, track=True,
    #                   epoch=200)
    # cross_environment(env_range=(0, 6), augmentation=False, ra_conv=True, diff=True, attention=False, track=False,
    #                   epoch=200)
    # cross_position((1, 5), augmentation=False, ra_conv=False, diff=True, attention=True, track=True, epoch=200)
    # cross_position((2, 5), augmentation=False, ra_conv=True, diff=False, attention=True, track=True, epoch=200)

    # cross_person(augmentation=True, ra_conv=True, diff=True, attention=True, track=True, epoch=200)

    # cross_environment(env_range=(0, 6), augmentation=False, ra_conv=False, diff=False, attention=False, track=False, epoch=100)
    # cross_environment(env_range=(0, 6), augmentation=False, ra_conv=True, diff=False, attention=False, track=False, epoch=100)
    # cross_environment(env_range=(0, 6), augmentation=False, ra_conv=False, diff=True, attention=False, track=False, epoch=100)
    # cross_environment(env_range=(0, 6), augmentation=False, ra_conv=False, diff=False, attention=True, track=True, epoch=100)


    # cross_position((0, 5), augmentation=False, ra_conv=True, diff=False, attention=False, track=False, epoch=100)
    # cross_position((0, 5), augmentation=False, ra_conv=False, diff=True, attention=False, track=False, epoch=100)
    # cross_position((0, 5), augmentation=False, ra_conv=False, diff=False, attention=True, track=True, epoch=100)

    # cross_environment(env_range=(5, 6), augmentation=False, ra_conv=True, diff=False, epoch=100)
    # domain_reduction_validation(d=2, augmentation=False, ra_conv=True, epoch=100)
    # domain_reduction_validation(d=1, n_try=6, augmentation=False, track=True, epoch=100)
    # domain_reduction_validation(d=0, augmentation=False, epoch=100)

    # domain_reduction_validation(d=0, n_reduction=5, augmentation=False, ra_conv=True, diff=False, attention=True,
    #                             track=False, epoch=100)
    # domain_reduction_validation(d=0, n_reduction=5, augmentation=False, ra_conv=False, diff=False, attention=False,
    #                             track=False, epoch=100)
    # domain_reduction_validation(d=1, n_try=6, augmentation=False, track=True, epoch=100)
    # domain_reduction_validation(d=3, n_try=(3, 100), augmentation=True, epoch=200)
    # cross_environment(env_range=(0, 6), augmentation=True, ra_conv=True, epoch=200)
    # cross_environment(env_range=(4, 5))
