from data.air_writing_dataset import *
import torch.optim as optim
from model.network import *
import os

history = {"acc_train": [], "acc_validation": [], "loss_train": [], "loss_validation": []}


# 定义训练函数
def train(net, train_loader, val_loader, num_epochs, criterion, optimizer, device):
    val_acc = 0
    for epoch in range(0, num_epochs):  # loop over the data multiple times
        net.train()
        running_loss = 0.0
        all_sample = 0.0
        correct_sample = 0.0
        for i, (data, label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(label)

            # calculation of accuracy
            all_sample = all_sample + len(label)
            prediction = torch.argmax(output, 1)
            correct_sample += (prediction == label).sum().float().item()
            if i % 5 == 4:  # print every 5 mini-batches
                print('[%d, %5d] loss: %.3f, accuracy: %.5f' % (
                    epoch + 1, i + 1, running_loss / all_sample, correct_sample / all_sample))
        print('[%d, %5d] all_samples: %.5f, correct_samples: %.5f,  loss: %.5f, accuracy: %.5f' % (
            epoch + 1, i + 1, all_sample, correct_sample, running_loss / all_sample, correct_sample / all_sample))

        # validation and save model
        net.eval()
        validation_loss = 0
        val_all_sample = 0.0
        val_correct_sample = 0.0
        with torch.no_grad():
            for i, (data, label) in enumerate(val_loader):
                data = data.to(device)
                label = label.to(device)
                output = net(data)
                loss = criterion(output, label)
                validation_loss += loss.item() * len(label)
                val_all_sample = val_all_sample + len(label)
                prediction = torch.argmax(output, 1)
                val_correct_sample += (prediction == label).sum().float().item()
        val_acc = val_correct_sample / val_all_sample
        val_loss = validation_loss / val_all_sample
        print('all validation: %.5f, correct validation: %.5f' % (val_all_sample, val_correct_sample))
        print('[%d, %5d] val loss: %.5f, accuracy: %.5f' % (epoch + 1, i + 1, val_loss, val_acc))

    return val_acc


def random_search():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 32
    train_set, validation_set, test_set = split_dataset()
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8,
                                              pin_memory=True)
    validationloader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=8,
                                                   pin_memory=True)
    # 定义超参数空间
    learning_rates = [0.03, 0.003, 0.0003]
    beta1 = [0.8, 0.85, 0.9, 0.99]
    beta2 = [0.9, 0.99, 0.999]
    epsilon = [1e-6, 1e-8, 1e-10]
    criterion = nn.CrossEntropyLoss()
    LPP_lr = 0  # .001
    best_acc = 0
    best_hp = [0,0,0,0]
    for t in range(10):
        lr = np.random.choice(learning_rates)
        b1 = np.random.choice(beta1)
        b2 = np.random.choice(beta2)
        ep = np.random.choice(epsilon)
        net = RDT_2DCNNLSTM_Air_Writing()
        net = net.to(device)
        range_nn_params = list(map(id, net.range_net.range_nn.parameters()))
        doppler_nn_params = list(map(id, net.doppler_net.doppler_nn.parameters()))
        base_params = filter(lambda p: id(p) not in range_nn_params + doppler_nn_params, net.parameters())
        optimizer = optim.Adam([
            {'params': base_params},
            {'params': net.range_net.parameters(), 'lr': LPP_lr},
            {'params': net.doppler_net.parameters(), 'lr': LPP_lr},
        ], lr=lr, betas=(b1, b2), eps=ep)
        cur_acc = train(net, trainloader, validationloader, 10, criterion, optimizer,device)
        if best_acc < cur_acc:
            best_acc = cur_acc
            best_hp = [lr,b1,b2,ep]

    print("best acc:{}, beta1:{}, beta2:{}, epsilon:{}".format(best_acc, best_hp[0], best_hp[1], best_hp[2], best_hp[3]))


if __name__ == '__main__':
    random_search()
# base_params = filter(lambda p:id(p) not in range_nn_params, net.parameters())
