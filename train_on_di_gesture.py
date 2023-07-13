import numpy as np

from data.di_gesture_dataset import *
import torch.optim as optim
from model.network import *
import matplotlib.pyplot as plt
import visdom
from torch.nn.utils.rnn import pad_sequence
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix

history = {"acc_train": [], "acc_validation": [], "loss_train": [], "loss_validation": []}
best_ture_label = []
best_predict_label = []
acc_win = '1_acc'
loss_win = '2_loss'
vis = visdom.Visdom(env='model_result', port=6006)


def plot_result(file_name):
    # 绘制准确率变化图
    plt.plot(history['acc_train'])
    plt.plot(history['acc_validation'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig('accuracy_{}.png'.format(file_name))
    plt.show()
    plt.clf()

    # 绘制损失变化图
    plt.plot(history['loss_train'])
    plt.plot(history['loss_validation'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig('loss_{}.png'.format(file_name))
    plt.show()
    plt.clf()

    cm = confusion_matrix(best_ture_label, best_predict_label)
    col_sum = np.sum(cm, axis=0)
    cm = np.round(100 * cm / col_sum[np.newaxis, :], 1)

    # 绘制热图
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig('confusion_matrix_{}.png'.format(file_name))
    plt.show()
    plt.clf()

    history['acc_train'].clear()
    history['acc_validation'].clear()
    history['loss_train'].clear()
    history['loss_validation'].clear()


def collate_fn(datas_and_labels):
    datas_and_labels = sorted(datas_and_labels, key=lambda x: x[0].size()[0], reverse=True)
    datas = [item[0] for item in datas_and_labels]
    tracks = torch.stack([item[1] for item in datas_and_labels])
    labels = torch.stack([item[2] for item in datas_and_labels])
    indexes = [item[3] for item in datas_and_labels]
    data_lengths = [len(x) for x in datas]
    datas = pad_sequence(datas, batch_first=True, padding_value=0)
    return datas, tracks, labels, torch.tensor(data_lengths), indexes


def train(train_set, test_set, start_epoch=37, net=None):
    if net is None:
        net = RAIRadarGestureClassifier(cfar=True, track=True, spatial_channels=(8, 16, 32), ra_conv=True,
                                              track_channels=(4, 8, 16), track_out_size=64, hidden_size=(128, 128),
                                              ra_feat_size=32, attention=True, cfar_expand_channels=8)
        net.load_state_dict(torch.load('test_cross_environment_3.pth')['model_state_dict'])

    net = net.to(device)
    cfar_params = list(map(id, net.sn.CFAR.parameters()))
    # lstm_nn_params = list(map(id,net.lstm.parameters()))
    # attention_params = list(map(id, net.multi_head_attention.parameters()))
    base_params = filter(lambda p: id(p) not in cfar_params, net.parameters())
    # base_params = filter(lambda p: id(p) not in cfar_params, net.parameters())
    optimizer = optim.Adam([
        # {'params': net.parameters()},
        {'params': base_params},
        # {'params': net.lstm.parameters()},
        # {'params': net.mult i_head_attention.parameters()},
        {'params': net.sn.CFAR.parameters(), 'lr': 0.0001},
    ], lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                             num_workers=8,
                                             pin_memory=True,
                                             collate_fn=collate_fn)
    acc_best = 0
    previous_acc = 0
    previous_loss = 1000000
    ture_label = []
    predict_label = []
    lr_list = np.zeros(total_epoch)
    lr_list[:100] = 0.001
    lr_list[100:150] = 0.0003
    lr_list[150:] = 0.0001
    lr_cfar = np.zeros(total_epoch)
    lr_cfar[:100] = 0.0001
    lr_cfar[100:150] = 0.00001
    lr_cfar[150:] = 0.00001

    pre_lr = 0
    for epoch in range(start_epoch, total_epoch):
        if not pre_lr == lr_list[epoch]:
            new_lr = lr_list[epoch]
            pre_lr = lr_list[epoch]
            print('!!!更新学习率 lr=' + str(new_lr))
            optimizer.param_groups[0]['lr'] = new_lr
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]['lr'] = lr_cfar[epoch]
            # train
        net.train()
        running_loss = 0.0
        all_sample = 0.0
        correct_sample = 0.0
        # for i, (datas, labels, data_lengths) in enumerate(trainloader):
        for i, (datas, tracks, labels, data_lengths, indexes) in enumerate(trainloader):
            datas = datas.to(device)
            tracks = tracks.to(device)
            labels = labels.to(device)
            ture_label.extend([x.item() for x in labels])
            data_lengths = data_lengths.to(device)
            optimizer.zero_grad()
            output = net(datas, tracks, data_lengths)
            # output = net(datas, data_lengths)
            # output = net(datas.float())
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() \
 \
            # calculation of accuracy
            all_sample = all_sample + labels.size(0)
            prediction = torch.argmax(output, 1)
            predict_label.extend([x.item() for x in prediction])
            correct_sample += (prediction == labels).sum().float().item()
            if i % 5 == 4:  # print every 5 mini-batches
                print('[Train] [%d, %5d] loss: %.3f, accuracy: %.5f' % (
                    epoch + 1, i + 1, running_loss / (i + 1), correct_sample / all_sample))
        train_acc = correct_sample / all_sample
        train_loss = running_loss / len(trainloader)
        print(
            '[Train] all_samples: %.5f, correct_samples: %.5f,  loss: %.5f, accuracy: %.5f' % (
                all_sample, correct_sample, train_loss, train_acc))
        history['acc_train'].append(train_acc)
        history['loss_train'].append(train_loss)
        ture_label.clear()
        predict_label.clear()

        # validation and save model
        net.eval()
        validation_loss = 0
        val_all_sample = 0.0
        val_correct_sample = 0.0
        with torch.no_grad():
            for i, (datas, tracks, labels, data_lengths, indexes) in enumerate(testloader):
                # for i, (datas, labels, data_lengths) in enumerate(testloader):
                ture_label.extend([x.item() for x in labels])
                datas = datas.to(device)
                labels = labels.to(device)
                tracks = tracks.to(device)
                data_lengths = data_lengths.to(device)
                output = net(datas, tracks, data_lengths, epoch=epoch, indexes=indexes)
                # output = net(datas, data_lengths)
                # output = net(datas.float())
                loss = criterion(output, labels)
                validation_loss += loss.item()
                val_all_sample = val_all_sample + len(labels)
                prediction = torch.argmax(output, 1)
                predict_label.extend([x.item() for x in prediction])
                val_correct_sample += (prediction == labels).sum().float().item()
        val_acc = val_correct_sample / val_all_sample
        val_loss = validation_loss / len(testloader)
        if val_acc > previous_acc or (val_acc == previous_acc and val_loss < previous_loss):
            acc_best = val_acc
            best_ture_label.clear()
            best_predict_label.clear()
            best_ture_label.extend(ture_label)
            best_predict_label.extend(predict_label)
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss
            }, model_name)
            torch.save({
                'model_state_dict': net.state_dict(),
            }, 'final_model.pth')
            previous_acc = val_acc
            previous_loss = val_loss
            print('saved')
        print('[Test] all validation: %.5f, correct validation: %.5f' % (val_all_sample, val_correct_sample))
        print('[Test] val loss: %.5f, accuracy: %.5f, best accuracy: %.5f ' % (
            val_loss, val_acc, acc_best))

        vis.line(X=np.array([epoch + 1]), Y=np.array([[train_acc, val_acc]]), win=acc_win,
                 update='append',
                 opts=dict(title='Train/val accuracy', xlabel='Epoch', ylabel='Accuracy',
                           legend=['Train accuracy', 'Val accuracy']))
        vis.line(X=np.array([epoch + 1]), Y=np.array([[train_loss, val_loss]]), win=loss_win,
                 update='append',
                 opts=dict(title='Train/val loss', xlabel='Epoch', ylabel='Loss', legend=['Train loss', 'Val loss']))

        history['acc_validation'].append(val_acc)
        history['loss_validation'].append(val_loss)
        ture_label.clear()
        predict_label.clear()
    return acc_best


def five_fold_validation(fold_range=(0, 5)):
    acc_history = []
    for fold in range(fold_range[0], fold_range[1]):
        train_set, test_set = split_data('in_domain', fold)
        acc = train(train_set, test_set)
        acc_history.append(acc)
        plot_result(fold)
        print('=====================fold{} for test acc_history:{}================='.format(fold + 1, acc_history))
    np.save('indomainacc.npy', np.array(acc_history))
    return acc_history


def cross_person():
    train_set, test_set = split_data('cross_person')
    acc = train(train_set, test_set)
    plot_result('cross_person')
    np.save('personacc.npy', np.array([acc]))
    print('cross person accuracy {}'.format(acc))


def cross_environment(env_range=(0, 6)):
    print('cross env')
    acc_history = []
    for e in range(env_range[0], env_range[1]):
        train_set, test_set = split_data('cross_environment', env_index=e)
        acc = train(train_set, test_set)
        acc_history.append(acc)
        plot_result('env_{}'.format(e))
        print('=====================env{} for test acc_history:{}================='.format(e + 1, acc_history))
    np.save('envacc.npy', np.array(acc_history))


def cross_position(loc_range=(4, 5), augmentation=True):
    acc_history = []
    for p in range(loc_range[0], loc_range[1]):
        train_set, test_set = split_data('cross_position', position_index=p)
        if not augmentation:
            train_set.transform = test_set.transform
        acc = train(train_set, test_set)
        acc_history.append(acc)
        plot_result('position_{}'.format(p))
        print('=====================position{} for test acc_history:{}================='.format(p + 1, acc_history))
    np.save('posacc.npy', np.array(acc_history))
    return acc_history


if __name__ == '__main__':
    learning_rate = 0.001
    LPP_lr = 0  # .001

    total_epoch = 200
    batch_size = 128
    # model_name = 'test.pth'

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model_name = 'test.pth'
    # five_fold_validation(fold_range=(0,3))

    # model_name = 'test_cross_person.pth'
    # cross_person()
    # cross_environment(device)
    # clear_cache()
    model_name = 'test_cross_environment_3.pth'
    cross_environment(env_range=(5, 6))
    # cross_environment(env_range=(4, 5))
    clear_cache()
    # model_name = 'test_cross_position.pth'
    # net.load_state_dict(torch.load(model_name)['model_state_dict'])
    # cross_position((2, 5))
    # clear_cache()
    # net.load_state_dict(torch.load('test.pth')['model_state_dict'])

    # net.load_state_dict(torch.load('test.pth')['model_state_dict'])
    clear_cache()
    # acc_full = cross_position(device, augmentation=False)
    # acc_aug_cfar = cross_position(device, loc_range=(4, 5), augmentation=True, need_cfar=True, need_track=False)
    # acc_aug_track = cross_position(device, loc_range=(4, 5), augmentation=True, need_cfar=False, need_track=True)
    # acc_aug = cross_position(device, loc_range=(4, 5), augmentation=True, need_cfar=False, need_track=False)
    # acc_none = cross_position(device, loc_range=(4, 5), augmentation=False, need_cfar=False, need_track=False)

    # vis.bar(
    #         X=np.array([[acc_full[0], acc_aug_cfar[0], acc_aug_track[0], acc_aug[0], acc_none[0]]]),
    #         win='compare2',  # 3个一组
    #         opts=dict(
    #             stacked=False,  # 不堆叠
    #             legend=['full', 'aug_cfar', 'aug_track', 'aug', 'none'],
    #             title='标题',  # 标题
    #             xlabel='position',  # x 轴
    #             ylabel='准确率',  # y轴
    #         )
    #     )
