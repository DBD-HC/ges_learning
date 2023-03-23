import numpy as np
import torch
import torch.nn as nn
from di_gesture_dataset import *
import torch.optim as optim
from network import *
import matplotlib.pyplot as plt
import torchvision
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import os

history = {"acc_train": [], "acc_validation": [], "loss_train": [], "loss_validation": []}


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

    history['acc_train'] = []
    history['acc_validation'] = []
    history['loss_train'] = []
    history['loss_validation'] = []


def collate_fn(datas_and_labels):
    datas_and_labels = sorted(datas_and_labels, key=lambda x: x[0].size()[0], reverse=True)
    datas = [item[0] for item in datas_and_labels]
    labels = torch.stack([item[1] for item in datas_and_labels])
    data_lengths = [len(x) for x in datas]
    datas = pad_sequence(datas, batch_first=True, padding_value=0)
    return datas, labels, data_lengths


def train(net, optimizer, criterion, train_set, test_set, batch_size):
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                             num_workers=8,
                                             pin_memory=True,
                                             collate_fn=collate_fn)
    total_epoch = 50
    model_name = 'test.pth'
    acc_best = 0
    previous_acc = 0
    previous_loss = 1000000
    for epoch in range(total_epoch):
        # train
        net.train()
        running_loss = 0.0
        all_sample = 0.0
        correct_sample = 0.0
        for i, (datas, labels, data_lengths) in enumerate(trainloader):
            datas = datas.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = net(datas.float(), data_lengths)
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
            for i, (datas, labels, data_lengths) in enumerate(testloader):
                datas = datas.to(device)
                labels = labels.to(device)
                output = net(datas.float(), data_lengths)
                loss = criterion(output, labels)
                validation_loss += loss.item() * len(labels)
                val_all_sample = val_all_sample + len(labels)
                prediction = torch.argmax(output, 1)
                val_correct_sample += (prediction == labels).sum().float().item()
        val_acc = val_correct_sample / val_all_sample
        val_loss = validation_loss / val_all_sample
        if val_acc > previous_acc or (val_acc == previous_acc and val_loss < previous_loss):
            acc_best = val_acc
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
    return acc_best


def five_fold_validation(device):
    acc_history = []
    for fold in range(5):
        net = DRAI_2DCNNLSTM_DI_GESTURE()
        net = net.to(device)

        optimizer = optim.Adam(net.parameters(), lr=learning_rate)

        criterion = nn.CrossEntropyLoss()

        batch_size = 128
        train_set, test_set = split_data('in_domain', fold + 1)
        acc = train(net, optimizer, criterion, train_set, test_set, batch_size)
        acc_history.append(acc)
        plot_result(fold)
        print('=====================fold{} for test acc_history:{}================='.format(fold + 1, acc_history))


if __name__ == '__main__':
    learning_rate = 0.0001
    LPP_lr = 0  # .001

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    five_fold_validation(device)
