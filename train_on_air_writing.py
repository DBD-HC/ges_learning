import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.classification import MulticlassAUROC, MulticlassAveragePrecision
from torchmetrics.functional import accuracy
from data import *
import torch.optim as optim
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from log_helper import LogHelper
from model_test import *
from data.air_writing_dataset import *
import seaborn as sns
from  model.network import RAIRadarGestureClassifier
from train import set_random_seed, get_propsed_lr

history = {"acc_train": [], "acc_validation": [], "loss_train": [], "loss_validation": []}
best_ture_label = []
best_predict_label = []

def seed_worker(worker_id):
    seed = worker_id + random_seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def dynamic_sequence_collate_fn(datas_and_labels):
    datas_and_labels = sorted(datas_and_labels, key=lambda x: x[0].size()[0], reverse=True)
    datas = [item[0] for item in datas_and_labels]
    labels = torch.stack([item[1] for item in datas_and_labels])
    data_lengths = [len(x) for x in datas]
    datas = pad_sequence(datas, batch_first=True, padding_value=0)
    return datas, labels, torch.tensor(data_lengths)

def unpack_run_model(model, pack):
    datas = pack[0].to(device)
    labels = pack[1].to(device)
    if len(pack) > 2:
        datalens = pack[2].to(device)
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





def unify_sequence(x, data_len):
    x = x.unsqueeze(0).unsqueeze(0)
    x = F.interpolate(x, size=(data_len, x.size(-2), x.size(-1)), mode='trilinear', align_corners=False)
    return torch.squeeze(x)


def static_sequence_collate_fn(datas_and_labels, sequence_len=40):
    datas_and_labels = sorted(datas_and_labels, key=lambda x: x[0].size()[0], reverse=True)
    datas = torch.stack([unify_sequence(item[0], sequence_len) for item in datas_and_labels])
    labels = torch.stack([item[2] for item in datas_and_labels])
    indexes = [item[3] for item in datas_and_labels]
    data_lengths = [len(x) for x in datas]
    return datas, labels, torch.tensor(data_lengths), indexes


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
    return datas, labels, torch.tensor(data_lengths), indexes


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


def cross_person(augmentation=True, need_cfar=True, ra_conv=True, diff=True, track=True, attention=True, epoch=200,
                 batch_size=16, start_epoch=0):
    # lr_main = get_propsed_lr(augmentation, epoch)
    set_random_seed(random_seed)
    for p_i, p in enumerate(participants):
        model = RAIRadarGestureClassifier(cfar=need_cfar, track=track, spatial_channels=(4, 8, 16), ra_conv=ra_conv,
                                                  heads=4,
                                                  track_channels=(4, 8, 16), track_out_size=32, conv2d_feat_size=64,
                                                  diff=diff,out_size=10,
                                                  ra_feat_size=32, attention=attention, cfar_expand_channels=8, in_channel=1)
        # model = DiGesture()
        # model = RadarNet()
        print(type(model).__name__)
        lr_main = get_propsed_lr(augmentation, epoch)
        # lr_main = get_di_gesture_lr(augmentation, epoch)
        # train_set, test_set = split_data('cross_person', person_index=3)
        train_set, test_set = split_data_air_writing(1, fold=0)
        if not augmentation:
            train_set.transform = test_set.transform
        # model_name = 'cross_person_radar_net.pth'
        # model_name = 'cross_person_di_gesture.pth'
        model_name = 'cross_user_aug_{}_diff_{}_ra_{}_attrack_{}.pth'.format(augmentation, diff, ra_conv, track)
        acc, auc, ap = train_and_val(model, train_set, test_set, start_epoch, epoch, batch_size=batch_size,
                                     lr=lr_main, title='cross person', model_name=model_name)
        loger.log(
            'cross person{} model:{} augmentation:{} need_cfar:{} need_diff:{} need_ra:{} att_track:{} acc:{} auc:{} ap:{}'
            .format(p, type(model).__name__, augmentation, need_cfar, diff, ra_conv, track, acc, auc, ap))



if __name__ == '__main__':
    loger = LogHelper()
    visdom_port = 6006
    vis = visdom.Visdom(env='model result', port=visdom_port)
    random_seed = 2023

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()

    # ges_label = ['CT', 'AT', 'PH', 'PL', 'LS', 'RS', 'NG']
    ges_label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    num_class = len(ges_label)
    multiclassAUROC = MulticlassAUROC(num_classes=num_class, average='macro').to(device)
    multiclassAveragePrecision = MulticlassAveragePrecision(num_classes=num_class, average='macro').to(device)

    cross_person(augmentation=True, ra_conv=True, diff=True, attention=True, track=True, epoch=100)





