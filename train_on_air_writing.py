import numpy as np

from data import *
import torch.optim as optim
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from model_test import *
import seaborn as sns

history = {"acc_train": [], "acc_validation": [], "loss_train": [], "loss_validation": []}
best_ture_label = []
best_predict_label = []


def collate_fn(datas_and_labels):
    datas_and_labels = sorted(datas_and_labels, key=lambda x: x[0].size()[0], reverse=True)
    datas = [item[0] for item in datas_and_labels]
    labels = torch.stack([item[1] for item in datas_and_labels])
    data_lengths = [len(x) for x in datas]
    datas = pad_sequence(datas, batch_first=True, padding_value=0)
    return datas, labels, torch.tensor(data_lengths)


def train(net, optimizer, criterion, train_set, test_set, batch_size):
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                             num_workers=8,
                                             pin_memory=True,
                                             collate_fn=collate_fn)
    total_epoch = 100
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
        for i, (datas, labels, data_lengths) in enumerate(trainloader):
            datas = datas.to(device)
            labels = labels.to(device)
            # data_lengths = data_lengths.to(device)
            optimizer.zero_grad()
            output = net(datas.float(), data_lengths)
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
            for i, (datas, labels, data_lengths) in enumerate(testloader):
                # for i, (datas, labels, data_lengths) in enumerate(testloader):
                ture_label.extend([x.item() for x in labels])
                datas = datas.to(device)
                labels = labels.to(device)
                # data_lengths = data_lengths.to(device)
                output = net(datas.float(), data_lengths)
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
        ture_label.clear()
        predict_label.clear()
    return acc_best

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

def leave_one(augmentation=True):
    acc_history = []
    for u in range(0, 8):
        net = DRAI_2DCNNLSTM_air_writing_2()
        net = net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        batch_size = 32
        train_set, test_set = split_RDAI(u)
        if not augmentation:
            train_set.transform = test_set.transform
        acc = train(net, optimizer, criterion, train_set, test_set, batch_size)
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
    acc = leave_one()
    np.save('avg_acc1.npy', np.array(acc))
    acc=leave_one(augmentation=False)
    np.save('avg_acc2.npy', np.array(acc))





