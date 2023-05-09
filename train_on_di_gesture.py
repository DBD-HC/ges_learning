import numpy as np
import torch
import torch.nn as nn
from di_gesture_dataset import *
import torch.optim as optim
from network import *
import matplotlib.pyplot as plt
import torchvision
import visdom
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
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


def train(net, optimizer, criterion, train_set, test_set, batch_size):
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                             num_workers=8,
                                             pin_memory=True,
                                             collate_fn=collate_fn)
    total_epoch = 200
    model_name = 'test.pth'
    acc_best = 0
    previous_acc = 0
    previous_loss = 1000000
    ture_label = []
    predict_label = []
    for epoch in range(total_epoch):
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
            # data_lengths = data_lengths.to(device)
            optimizer.zero_grad()
            output = net(datas, tracks, data_lengths)
            # output = net(datas, data_lengths)
            # output = net(datas.float())
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

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
                # data_lengths = data_lengths.to(device)
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


def five_fold_validation(device):
    acc_history = []
    for fold in range(3, 5):
        net = DRAI_2DCNNLSTM_DI_GESTURE()
        net = net.to(device)

        optimizer = optim.Adam(net.parameters(), lr=learning_rate)

        criterion = nn.CrossEntropyLoss()

        batch_size = 128
        train_set, test_set = split_data('in_domain', fold)
        acc = train(net, optimizer, criterion, train_set, test_set, batch_size)
        acc_history.append(acc)
        plot_result(fold)
        print('=====================fold{} for test acc_history:{}================='.format(fold + 1, acc_history))
    np.save('indomainacc.npy', np.array(acc_history))
    return acc_history


def cross_person(device):
    net = DRAI_2DCNNLSTM_DI_GESTURE()
    net = net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    batch_size = 128
    train_set, test_set = split_data('cross_person')
    acc = train(net, optimizer, criterion, train_set, test_set, batch_size)
    plot_result('cross_person')
    np.save('personacc.npy', np.array([acc]))
    print('cross person accuracy {}'.format(acc))


def cross_environment(device):
    acc_history = []
    for e in range(5, 6):
        net = DRAI_2DCNNLSTM_DI_GESTURE()
        net = net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)

        criterion = nn.CrossEntropyLoss()

        batch_size = 128
        train_set, test_set = split_data('cross_environment', env_index=e)
        acc = train(net, optimizer, criterion, train_set, test_set, batch_size)
        acc_history.append(acc)
        plot_result('env_{}'.format(e))
        print('=====================env{} for test acc_history:{}================='.format(e + 1, acc_history))
    np.save('envacc.npy', np.array(acc_history))


def cross_position(device, loc_range=(0, 5), augmentation=True, need_cfar=True, need_track=True):
    acc_history = []
    for p in range(loc_range[0], loc_range[1]):
        net = DRAI_2DCNNLSTM_DI_GESTURE_BETA(cfar=need_cfar, track=need_track)
        # net = DRAI_2DCNNLSTM_DI_GESTURE()
        net = net.to(device)
        attention_params = list(map(id, net.multi_head_attention.parameters()))
        if need_cfar:
            cfar_params = list(map(id, net.sn.CFAR.parameters()))
            # lstm_nn_params = list(map(id,net.lstm.parameters()))
            base_params = filter(lambda p: id(p) not in cfar_params + attention_params, net.parameters())
            # base_params = filter(lambda p: id(p) not in cfar_params, net.parameters())
            optimizer = optim.Adam([
                {'params': base_params},
                # {'params': net.lstm.parameters()},
                {'params': net.multi_head_attention.parameters()},
                {'params': net.sn.CFAR.parameters(), 'lr': 0.0001},
            ], lr=learning_rate)
        else:
            base_params = filter(lambda p: id(p) not in attention_params, net.parameters())
            optimizer = optim.Adam([
                {'params': base_params},
                {'params': net.multi_head_attention.parameters()},
            ], lr=learning_rate)

        criterion = nn.CrossEntropyLoss()

        batch_size = 64
        train_set, test_set = split_data('cross_position', position_index=p)
        if not augmentation:
            train_set.transform = test_set.transform
        acc = train(net, optimizer, criterion, train_set, test_set, batch_size)
        acc_history.append(acc)
        plot_result('position_{}'.format(p))
        print('=====================position{} for test acc_history:{}================='.format(p + 1, acc_history))
    np.save('posacc.npy', np.array(acc_history))
    return acc_history


if __name__ == '__main__':
    learning_rate = 0.0003
    LPP_lr = 0  # .001

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # five_fold_validation(device)
    # cross_person(device)
    # cross_environment(device)
    # cross_position(device)
    acc_full = cross_position(device, augmentation=False)
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
