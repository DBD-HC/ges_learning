import subprocess

import psutil
import torch.optim as optim
from data.di_gesture_dataset import *
from log_helper import LogHelper
from model.network import *
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC, MulticlassAveragePrecision
import os
from sklearn.metrics import confusion_matrix


def get_model():
    return -1


def get_acc():
    return -1


def check_visdom_running(port):

    for proc in psutil.process_iter(['pid', 'name']):
        if 'visdom' in proc.info['name']:
            return 0

    print("Visdom is not running. Starting Visdom...")
    result = subprocess.run(['visdom', '-port', str(port)], capture_output=True, text=True)
    print(result)


def seed_worker(worker_id):
    random.seed(worker_id + random_seed)


# 定义随机种子固定的函数
def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


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
                                                                                                    running_loss / len(dataloader),
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
    val_ap = multiclassAveragePrecision.compute()
    print('[Validation] all validation: %.5f, correct validation: %.5f' % (total_sample, correct_sample))
    print('[Validation] val loss: %.5f, auc: %.5f, ap: %.5f, accuracy: %.5f' % (
        running_loss / len(dataloader), val_auc, val_ap, val_acc))

    return val_acc, running_loss / len(dataloader), val_auc, val_ap, predict_label, ture_label


def di_gesture_collate_fn(datas_and_labels):
    datas_and_labels = sorted(datas_and_labels, key=lambda x: x[0].size()[0], reverse=True)
    datas = [item[0] for item in datas_and_labels]
    labels = torch.stack([item[2] for item in datas_and_labels])
    indexes = [item[3] for item in datas_and_labels]
    data_lengths = [len(x) for x in datas]
    datas = pad_sequence(datas, batch_first=True, padding_value=0)
    return datas, labels, torch.tensor(data_lengths), indexes


def train_and_val(model, train_set, val_set, collate_fn, start_epoch, total_epoch, batch_size, lr, title=''):
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
            }, model_name)
            torch.save({
                'model_state_dict': model.state_dict(),
            }, 'final_model.pth')
            print('saved')

        print('[Validation] best  auc: %.5f, ap: %.5f, accuracy: %.5f' % (
            best_auc, best_ap, best_acc))
    print(title + ' ' + str(best_acc))
    plot_confusion_matrix(best_ture_label, best_predict_label, title)
    return best_acc, best_auc, best_ap


def five_fold_validation(fold_range=(0, 5), augmentation=True, need_cfar=True, ra_conv=True, diff=True, epoch=200,
                         batch_size=128):
    lr_main, lr_cfar = get_propsed_lr(augmentation, epoch)
    for fold in range(fold_range[0], fold_range[1]):
        train_set, test_set = split_data('in_domain', fold)
        if not augmentation:
            train_set.transform = test_set.transform
        # net = RAIRadarGestureClassifier(cfar=need_cfar, track=True, spatial_channels=(4, 8, 16), ra_conv=True, heads=4,
        #                                 track_channels=(4, 8, 16), track_out_size=64, hidden_size=(128, 128),
        #                                 ra_feat_size=32, attention=True, cfar_expand_channels=8, in_channel=1)
        net = DiGesture()
        acc, auc, ap = train_and_val(net, train_set, test_set, di_gesture_collate_fn, 0, epoch, batch_size=batch_size,
                                     lr=lr_main, title='in_domain fold ' + str(fold))
        loger.log(
            'in_domain model:{} augmentation:{} need_cfar:{} need_diff:{}  acc:{} auc:{} ap:{}'
            .format(type(net).__name__, augmentation, need_cfar, diff, acc, auc, ap))


def cross_person(augmentation=True, need_cfar=True, ra_conv=True, diff=True, epoch=200, batch_size=128):
    lr_main, lr_cfar = get_propsed_lr(augmentation, epoch)
    net = RAIRadarGestureClassifier(cfar=need_cfar, track=True, spatial_channels=(4, 8, 16), ra_conv=ra_conv, heads=4,
                                    track_channels=(4, 8, 16), track_out_size=32, hidden_size=(128, 128), diff=diff,
                                    ra_feat_size=32, attention=True, cfar_expand_channels=8, in_channel=1)
    train_set, test_set = split_data('cross_person')
    if not augmentation:
        train_set.transform = test_set.transform
    acc, auc, ap = train_and_val(net, train_set, test_set, di_gesture_collate_fn, 0, epoch, batch_size=batch_size,
                                 lr=lr_main, title='cross person')
    loger.log('cross person model:{} augmentation:{} need_cfar:{} need_diff:{}  acc:{} auc:{} ap:{}'
        .format(type(net).__name__, augmentation, need_cfar, diff, acc, auc, ap))


def cross_environment(augmentation=True, need_cfar=True, env_range=(0, 6), ra_conv=True, diff=True, epoch=200, batch_size=128):
    print('cross env')
    res_history = []
    lr_main, lr_cfar = get_propsed_lr(augmentation, epoch)
    for e in range(env_range[0], env_range[1]):
        train_set, test_set = split_data('cross_environment', env_index=e)
        if not augmentation:
            train_set.transform = test_set.transform
        net = RAIRadarGestureClassifier(cfar=need_cfar, track=True, spatial_channels=(4, 8, 16), ra_conv=ra_conv,
                                        heads=4,track_channels=(4, 8, 16), track_out_size=32, hidden_size=(128, 128),
                                        diff=diff,ra_feat_size=32, attention=True, cfar_expand_channels=8, in_channel=1)

        acc, auc, ap = train_and_val(net, train_set, test_set, di_gesture_collate_fn, 0, epoch, batch_size=batch_size,
                                     lr=lr_main, title='cross environment ' + str(e))
        loger.log('cross environment model:{} augmentation:{} need_cfar:{} need_diff:{}  acc:{} auc:{} ap:{}'
                  .format(type(net).__name__, augmentation, need_cfar, diff, acc, auc, ap))


def cross_position(loc_range=(4, 5), augmentation=True, need_cfar=True, diff=True, ra_conv=True, epoch=200, batch_size=128):
    res_history = []
    lr_main, lr_cfar = get_propsed_lr(augmentation, epoch)
    for p in range(loc_range[0], loc_range[1]):
        train_set, test_set = split_data('cross_position', position_index=p)
        if not augmentation:
            train_set.transform = test_set.transform
        net = RAIRadarGestureClassifier(cfar=need_cfar, track=False, spatial_channels=(4, 8, 16), ra_conv=ra_conv,
                                        heads=4,
                                        track_channels=(4, 8, 16), track_out_size=32, hidden_size=(128, 128), diff=diff,
                                        ra_feat_size=32, attention=True, cfar_expand_channels=8, in_channel=1)
        # net = DiGesture()
        acc, auc, ap = train_and_val(net, train_set, test_set, di_gesture_collate_fn, 0, epoch, batch_size=batch_size,
                                     lr=lr_main, title='cross position' + str(p))
        loger.log('cross position model:{} augmentation:{} need_cfar:{} need_diff:{}  acc:{} auc:{} ap:{}'
                  .format(type(net).__name__, augmentation, need_cfar, diff, acc, auc, ap))
    return res_history


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
        lr_list[:epoch] = 0.0001
        lr_cfar[:epoch] = 0.00001
    return lr_list, lr_cfar



if __name__ == '__main__':
    loger = LogHelper()
    visdom_port = 6006
    check_visdom_running(visdom_port)
    vis = visdom.Visdom(env='model result', port=visdom_port)
    random_seed = 2023
    set_random_seed(random_seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()

    ges_label = ['CT', 'AT', 'PH', 'PL', 'LS', 'RS', 'NG']
    num_class = len(ges_label)
    multiclassAUROC = MulticlassAUROC(num_classes=num_class, average='macro').to(device)
    multiclassAveragePrecision = MulticlassAveragePrecision(num_classes=num_class, average='macro').to(device)

    model_name = 'test2.pth'
    # five_fold_validation(fold_range=(0,5), augmentation=False, need_cfar=False, epoch=100)
    clear_cache()
    model_name = 'test_cross_person_no_aug.pth'
    # cross_person(augmentation=False, ra_conv=False, diff=True, epoch=100)
    clear_cache()
    model_name = 'test_cross_person.pth'
    # cross_person(augmentation=True,  ra_conv=False, diff=True, epoch=200)
    clear_cache()
    model_name = 'test_cross_environment_no_aug.pth'
    # cross_environment(env_range=(0, 6), augmentation=False, ra_conv=False, epoch=100)
    clear_cache()
    model_name = 'test_cross_environment.pth'
    # cross_environment(env_range=(0, 6), augmentation=True, ra_conv=False, epoch=200)
    # cross_environment(env_range=(4, 5))

    # cross_environment(device)
    # domain_reduction_validation()
    clear_cache()
    model_name = 'test_cross_position_no_aug.pth'
    cross_position((0, 5), augmentation=False, ra_conv=True, epoch=100)
    clear_cache()
    model_name = 'test_cross_position.pth'
    cross_position((0, 5), augmentation=True, ra_conv=True, epoch=200)
