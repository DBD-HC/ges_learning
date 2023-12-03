import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from data import *
import torch.optim as optim
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from model_test import *
from data.air_writing_dataset import *
import seaborn as sns
from  model.network import RAIRadarGestureClassifier

history = {"acc_train": [], "acc_validation": [], "loss_train": [], "loss_validation": []}
best_ture_label = []
best_predict_label = []


def collate_fn(datas_and_labels):
    datas_and_labels = sorted(datas_and_labels, key=lambda x: x[0].size()[0], reverse=True)
    datas = [item[0] for item in datas_and_labels]
    labels = torch.stack([item[1] for item in datas_and_labels])
    # data_lengths = [len(x) for x in datas]
    datas = pad_sequence(datas, batch_first=True, padding_value=0)
    return datas, labels, None


def train(net, optimizer, criterion, train_set, test_set, val_set, batch_size):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8,
                                              pin_memory=True,                                              collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                             num_workers=8,
                                             pin_memory=True,
                                             collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
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
    lr_list = np.zeros(total_epoch)
    lr_list[:100] = 0.0001
    lr_list[100:150] = 0.0003
    lr_list[150:] = 0.00001
    pre_lr = 0.01
    for epoch in range(total_epoch):
        if not pre_lr == lr_list[epoch]:
            new_lr = lr_list[epoch]
            pre_lr = lr_list[epoch]
            print('!!!更新学习率 lr=' + str(new_lr))
            optimizer.param_groups[0]['lr'] = new_lr
            # train
        # train
        net.train()
        running_loss = 0.0
        all_sample = 0.0
        correct_sample = 0.0
        # for i, (datas, labels, data_lengths) in enumerate(train_loader):
        for i, (datas, labels, data_lengths) in enumerate(train_loader):
            datas = datas.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = net(datas, None, None)
            # output = net(datas.float())
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)

            # calculation of accuracy
            all_sample = all_sample + len(labels)
            prediction = torch.argmax(output, 1)
            correct_sample += (prediction == labels).sum().float().item()
            if i % 5 == 4:  # print every 5 mini-batches
                print('[%d, %5d] loss: %.3f, accuracy: %.5f' % (
                    epoch + 1, i + 1, running_loss / all_sample, correct_sample / all_sample))
        print('[%d, %5d] all_samples: %.5f, correct_samples: %.5f,  loss: %.5f, accuracy: %.5f' % (
            epoch + 1, i + 1, all_sample, correct_sample, running_loss / all_sample, correct_sample / all_sample))
        history['acc_train'].append(correct_sample / all_sample)
        history['loss_train'].append(running_loss / all_sample)

        # validation and save model
        net.eval()
        validation_loss = 0
        val_all_sample = 0.0
        val_correct_sample = 0.0
        with torch.no_grad():
            for i, (datas, labels, data_lengths) in enumerate(val_loader):
                # for i, (datas, labels, data_lengths) in enumerate(val_loader):
                ture_label.extend([x.item() for x in labels])
                datas = datas.to(device)
                labels = labels.to(device)
                output = net(datas, None, None)
                # output = net(datas.float())
                loss = criterion(output, labels)
                validation_loss += loss.item() * len(labels)
                val_all_sample = val_all_sample + len(labels)
                prediction = torch.argmax(output, 1)
                predict_label.extend([x.item() for x in prediction])
                val_correct_sample += (prediction == labels).sum().float().item()
        val_acc = val_correct_sample / val_all_sample
        val_loss = validation_loss / val_all_sample
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
        print('all validation: %.5f, correct validation: %.5f' % (val_all_sample, val_correct_sample))
        print('[%d, %5d] val loss: %.5f, accuracy: %.5f' % (epoch + 1, i + 1, val_loss, val_acc))
        history['acc_validation'].append(val_acc)
        history['loss_validation'].append(val_loss)
        test_all_sample = 0
        test_correct_sample = 0
        best_ture_label.clear()
        best_predict_label.clear()
        with torch.no_grad():
            for i, (datas, labels, data_lengths) in enumerate(test_loader):
                # for i, (datas, labels, data_lengths) in enumerate(val_loader):
                best_ture_label.extend([x.item() for x in labels])
                datas = datas.to(device)
                labels = labels.to(device)
                output = net(datas, None, None)
                test_all_sample = test_all_sample + len(labels)
                prediction = torch.argmax(output, 1)
                best_predict_label.extend([x.item() for x in prediction])
                test_correct_sample += (prediction == labels).sum().float().item()
    return test_correct_sample/test_all_sample

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

    cm = confusion_matrix(best_ture_label, best_predict_label)
    col_sum = np.sum(cm, axis=0)
    cm = np.round(100 * cm / col_sum[np.newaxis, :], 1)

    # 绘制热图
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig('confusion_matrix_{}.png'.format(file_name))
    plt.show()

    history['acc_train'].clear()
    history['acc_validation'].clear()
    history['loss_train'].clear()
    history['loss_validation'].clear()

def cus_train(augmentation=True):
    acc_history = []
    for u in range(0, 8):
        net = RAIRadarGestureClassifier(cfar=True, track=True, spatial_channels=(8, 16, 32), ra_conv=True, heads=4,
                                        track_channels=(8, 16, 32), track_out_size=64, hidden_size=(128, 128),
                                        ra_feat_size=32, in_channel=2, attention=True, cfar_expand_channels=16, out_size=10)
        net = net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        batch_size = 8
        train_set, val_set, test_set = new_split_data()
        acc = train(net, optimizer, criterion, train_set, test_set, val_set, batch_size)
        acc_history.append(acc)
        plot_result('user_' + str(u))
        print(acc_history)

    print('avg_acc = {}'.format(np.mean(acc_history)))
    return acc_history

if __name__ == '__main__':
    learning_rate = 0.0001
    LPP_lr = 0  # .001

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    acc = cus_train()
    np.save('avg_acc1.npy', np.array(acc))
    # acc=cus_train(augmentation=False)
    # np.save('avg_acc2.npy', np.array(acc))





