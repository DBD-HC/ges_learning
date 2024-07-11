import os
import random

import numpy as np
import torch
import visdom

from data.dan_dataset import DANDataset
from data.mcd_dataset import MCDDataSplitter, data_augmentation as mcd_aug
from data.rai_ges_dataset import RAIGesDataSplitter, data_augmentation as rai_aug
from model.target_free_dann import *
from result_collector import cross_domain_results
from train import get_dataloader, ModelTrainingManager, get_acc_per_class


def set_random_seed(seed=1998):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def deep_rai_collate_fn(datas_and_labels):
    datas_and_labels = sorted(datas_and_labels, key=lambda x: x[0].size()[0], reverse=True)
    datas = [item[0] for item in datas_and_labels]
    labels = torch.stack([item[2] for item in datas_and_labels])
    tracks = torch.stack([item[1] for item in datas_and_labels])
    data_lengths = [len(x) for x in datas]
    datas = pad_sequence(datas, batch_first=True, padding_value=0)
    return datas, torch.tensor(data_lengths), tracks, labels


def dg_collate_fn(datas_and_labels):
    datas_and_labels = sorted(datas_and_labels, key=lambda x: x[0].size()[0], reverse=True)
    inputs1 = [item[0] for item in datas_and_labels]
    data_lengths1 = [len(x) for x in inputs1]
    inputs1 = pad_sequence(inputs1, batch_first=True, padding_value=0)
    tracks1 = torch.stack([item[1] for item in datas_and_labels])
    label1 = torch.stack([item[2] for item in datas_and_labels])

    return inputs1, tracks1, torch.tensor(data_lengths1), label1


def dg_collate_fn2(datas_and_labels):
    datas_and_labels = sorted(datas_and_labels, key=lambda x: x[0].size()[0], reverse=True)
    inputs1 = [item[0] for item in datas_and_labels]
    data_lengths1 = [len(x) for x in inputs1]
    inputs1 = pad_sequence(inputs1, batch_first=True, padding_value=0)
    # tracks1 = torch.stack([item[1] for item in datas_and_labels])
    inputs2 = [item[1] for item in datas_and_labels]
    data_lengths2 = [len(x) for x in inputs2]
    inputs2 = pad_sequence(inputs2, batch_first=True, padding_value=0)
    # tracks2 = torch.stack([item[3] for item in datas_and_labels])
    label1 = torch.stack([item[2] for item in datas_and_labels])

    return inputs1, torch.tensor(data_lengths1), inputs2, torch.tensor(data_lengths2),label1


def get_lr(total_epoch=200):
    lr_list = np.zeros(total_epoch)
    lr_list[:total_epoch // 2] = 0.001
    lr_list[total_epoch // 2:total_epoch // 2 + total_epoch // 4] = 0.0003
    lr_list[total_epoch // 2 + total_epoch // 4:] = 0.0001
    return lr_list

def get_gr_lr(total_epoch=200):
    lr_list = np.zeros(total_epoch)
    lr_list[:total_epoch // 2] = 0.001
    lr_list[total_epoch // 2:total_epoch // 2 + total_epoch // 4] = 0.0003
    lr_list[total_epoch // 2 + total_epoch // 4:] = 0.
    return lr_list


def frame_norm(data):
    data_mean = torch.mean(data, dim=-1, keepdim=True)
    std = torch.std(data, dim=-1, keepdim=True)
    data = (data - data_mean)/(std + 1e-9)
    return data


def negative_cosine_similarity(real_d, fake_d, phi=0.8):
    """ D(p, z) = -(p*z).sum(dim=1).mean() """
    real_r = torch.mean(real_d, dim=-1)
    fake_r = torch.mean(fake_d, dim=-1)
    real_a = torch.mean(real_d, dim=-2)
    fake_a = torch.mean(fake_d, dim=-2)
    angle_recon_loss = F.relu(phi - F.cosine_similarity(frame_norm(real_a), frame_norm(fake_a), dim=-1)).mean()
    range_recon_loss = F.relu(phi - F.cosine_similarity(frame_norm(real_r), frame_norm(fake_r), dim=-1)).mean()
    return range_recon_loss + angle_recon_loss

def recon_loss(real_d, fake_d, data_len=None):
    mask = torch.arange((data_len.max()), dtype=torch.float32, device=data_len.device)
    mask = mask[None, :] < data_len[:, None]
    mask = mask.view(-1)
    real_d = real_d[mask]
    fake_d = fake_d[mask]
    return F.mse_loss(fake_d, real_d)

def get_norm(rais):
    rai_mean = torch.mean(rais, dim=(-2, -1), keepdim=True)
    rai_var = torch.var(rais, dim=(-2, -1), keepdim=True)
    rais = (rais- rai_mean)/torch.sqrt(rai_var + 1e-9)
    return rais

def negative_cosine_similarity2(real_d, fake_d, phi=0.8, data_len=None):
    """ D(p, z) = -(p*z).sum(dim=1).mean() """
    #real_v = torch.mean(real_d, dim=(-2, -1))
    #fake_v = torch.mean(fake_d, dim=(-2, -1))
    mask = torch.arange((data_len.max()), dtype=torch.float32, device=data_len.device)
    mask = mask[None, :] < data_len[:, None]
    mask = mask.view(-1)
    real_d = real_d[mask]
    fake_d = fake_d[mask]
    #return F.mse_loss(real_d, fake_d)
    return F.relu(phi - F.cosine_similarity(real_d, fake_d, dim=-1)).mean() # + F.mse_loss(real_d.mean(-1), fake_d.mean(-1))

def get_track(rai, data_len):
    packed_rai = rai[:data_len]
    # packed_rai = frame_norm(packed_rai)
    # 计算每个维度的最大值、均值和标准差
    rai_max = torch.max(packed_rai, dim=0).values
    rai_mean = torch.mean(packed_rai, dim=0)
    rai_std = torch.std(packed_rai, dim=0)

    # 将结果连接起来
    global_track = torch.cat((rai_max.unsqueeze(0), rai_mean.unsqueeze(0), rai_std.unsqueeze(0)), dim=0)
    return global_track


def get_acc(outputs, labels):
    prediction = torch.argmax(outputs.detach(), 1)
    cor_num = (prediction == labels).sum().float().item()
    temp_num = len(prediction)

    return cor_num / temp_num, cor_num, temp_num


def alpha_decay(x, L=1, x0=50, k=0.1):

    return L / (1 + np.exp(-k * (x - x0)))


# 训练模型
def train_dg(train_index, test_index, domain, dataset_splitter=None, need_test=True, num_epochs=200, batch_size=128,
             phi=0.7, val_aug=False):
    set_random_seed()
    if dataset_splitter == None:
        # dataset_splitter = MCDDataSplitter()
        dataset_splitter = RAIGesDataSplitter()
    vis = visdom.Visdom(env='model result', port=6006)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 定义损失函数和优化器
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    lr = get_lr(num_epochs)
    lr2 = get_gr_lr(num_epochs)

    # train_index = [1]
    # test_index = [0, 2, 3, 4]
    # train_index = [1]
    # test_index = [0, 2, 3, 4]
    # rai_ges_splitter = RAIGesDataSplitter()
    # domain = 3
    # need_test = True
    domain_num = len(test_index) if test_index is not None else 0
    if need_test:
        domain_num = domain_num + 1
    acc_auc_ap = np.zeros((5, domain_num, 3))
    train_manager = ModelTrainingManager(class_num=dataset_splitter.get_class_num())

    for v_i in range(5):
        best_loss = 1e9
        model = DANN(out_size=dataset_splitter.get_class_num())
        g_model = Generator()
        model.to(device)
        g_model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        g_optimizer = torch.optim.Adam(g_model.parameters(), lr=0.001)
        train_set, test_set, val_set = dataset_splitter.split_data(domain, train_index=train_index, need_val=True,
                                                                   need_test=need_test,
                                                                   need_augmentation=True)
        train_set1 = DANDataset(train_set.file_names, train_set.labels, data_root=train_set.data_root, transform=train_set.transform)
        train_loader = get_dataloader(train_set1, True, batch_size, dg_collate_fn2)

        if val_aug:
            val_set.transform = train_set.transform
        val_loader = get_dataloader(val_set, False, batch_size, deep_rai_collate_fn)

        total_step = len(train_loader)
        for epoch in range(num_epochs):
            model.train()
            g_model.train()
            t = 0
            len_dataloader = len(train_loader)
            for i, (datas1, data_len1, datas2, data_len2,  l1) in enumerate(train_loader):
                p = float(i + epoch * len_dataloader) / num_epochs / len_dataloader
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                # alpha = 1
                t += 1
                optimizer.param_groups[0]['lr'] = lr[epoch]
                g_optimizer.param_groups[0]['lr'] = lr[epoch]
                datas1 = datas1.to(device)
                data_len1 = data_len1.to(device)
                datas2 = datas2.to(device)
                data_len2 = data_len2.to(device)
                l1 = l1.to(device)
                #beta = 1/alpha if alpha > 0 else 0

                fake_data, fake_data_rev = g_model(datas2, 1, data_len2)

                # 重构损失
                loss_recon = negative_cosine_similarity2(datas2.view(-1, 1024), fake_data.view(-1, 1024), phi=phi, data_len=data_len2)#\
                # loss_recon = recon_loss(datas2.view(-1, 1024), fake_data.view(-1, 1024), data_len=data_len2)

                fake_track = torch.stack([get_track(x, data_len2[x_i]) for x_i, x in enumerate(fake_data)])
                fake_track_rev = torch.stack([get_track(x, data_len2[x_i]) for x_i, x in enumerate(fake_data_rev)])
                tracks = torch.stack([get_track(x, data_len1[x_i]) for x_i, x in enumerate(datas1)])
                #with torch.no_grad():
                #    fake_data2 = g_model(datas1)

                #fake_track2 = torch.stack([get_track(x, data_len1[x_i]) for x_i, x in enumerate(fake_data2)])
                # weight = torch.randn((datas1.size(0), datas1.size(1), 1, 1), device=device)
                # fake_mix = weight * fake_data + (1 - weight) * datas1

                z1, m1 = model(datas1, tracks, data_len1, alpha)
                z2, m2 = model(fake_data_rev, fake_track_rev, data_len2, alpha, True)
                z3 = model(fake_data, fake_track, data_len2, alpha, False)
                # z_mix, _ = model(fake_mix, fake_track_mix, data_len1, alpha, False)

                temp_acc1, _, _ = get_acc(z1, l1)
                temp_acc2, _, _ = get_acc(z2, l1)

                domain1 = torch.zeros(len(m1), device=device).long()
                domain3 = torch.ones(len(m2), device=device).long()
                # 手势预测损失
                loss_class = criterion(z1, l1) +  criterion(z3, l1) #+ criterion(z3, l1)
                # 域分类损失
                loss_domain = criterion(m1, domain1) + criterion(m2, domain3)

                d_acc1, _, _ = get_acc(m1, domain1)
                d_acc2, _, _ = get_acc(m2, domain3)

                loss = loss_class + loss_domain +  loss_recon  # + F.relu(1 - loss_con)

                with torch.no_grad():
                    vis.heatmap(datas2[0, 5], win='real', opts={
                        'title': 'real'
                    })
                    vis.heatmap(fake_data[0, 5], win='fake', opts={
                        'title': 'fake'
                    })
                    vis.heatmap(tracks[0, 0], win='real_t', opts={
                        'title': 'real_t'
                    })
                    vis.heatmap(fake_track[0, 0], win='fake_t', opts={
                        'title': 'fake_t'
                    })

                g_optimizer.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                g_optimizer.step()
                optimizer.step()

                if (i + 1) % 2 == 0:
                    print(
                        'Epoch [{}/{}], Step [{}/{}], Loss_total: {:.4f}, Loss:_class {:.4f}, Loss_domain: {:.4f}, Loss_rec: {:.4f} temp_acc1: {:.4f} temp_acc2: {:.4f} d_acc1: {:.4f} d_acc2: {:.4f}'
                        .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), loss_class.item(),
                                loss_domain.item(), loss_recon.item(), temp_acc1, temp_acc2, d_acc1, d_acc2))

            running_loss = 0.0
            running_loss1 = 0.0
            running_loss2 = 0.0
            model.eval()
            g_model.eval()
            for i, (datas1, data_len1, tracks1, l1) in enumerate(val_loader):
                datas1 = datas1.to(device)
                l1 = l1.to(device)
                z1, m1 = model(datas1, tracks1.to(device), data_len1.to(device), 1)
                #fake_data, _ = g_model(datas1)
                #fake_track = torch.stack([get_track(x, data_len1[x_i]) for x_i, x in enumerate(fake_data)])
                #z2 = model(fake_data.to(device), fake_track.to(device), data_len1.to(device), 1, False)
                loss = criterion(z1, l1)
                running_loss += loss.item()

            if epoch % 5 == 0:
                val_model = model.predication
                val_model.classifier.need_hidden = False

                if need_test:
                    mid_test_loader = get_dataloader(test_set, False, batch_size, deep_rai_collate_fn)
                    x = train_manager.test_or_val(val_model, mid_test_loader)
                    acc_auc_ap[v_i, 0, 0], _, acc_auc_ap[v_i, 0, 1], acc_auc_ap[v_i, 0, 2], _, _ = x
                if test_index is not None:
                    for i, t_i in enumerate(test_index, start=1 if need_test else 0):
                        mid_test_set = dataset_splitter.get_dataset([t_i])
                        mid_test_loader = get_dataloader(mid_test_set, False, batch_size, deep_rai_collate_fn)
                        acc_auc_ap[v_i, i, 0], _, acc_auc_ap[v_i, i, 1], acc_auc_ap[
                                v_i, i, 2], _, _ = train_manager.test_or_val(val_model, mid_test_loader)
                val_model.classifier.need_hidden = True



            running_loss = running_loss / t
            print(
                'loss1 {:.4f}, loss2 {:.4f}, total loss {:.4f}'.format(running_loss1 / t, running_loss2 / t,
                                                                       running_loss))
            if running_loss < best_loss and epoch > num_epochs * 3 // 4:
                best_loss = running_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': running_loss
                }, os.path.join('checkpoint', 'test_dk.pth'))
                print('==saved==')
        params = torch.load(os.path.join('checkpoint', 'test_dk.pth'))['model_state_dict']
        model.load_state_dict(params)

        val_model = model.predication
        val_model.classifier.need_hidden = False
        val_model.load_state_dict(model.predication.state_dict(), strict=False)
        val_model.to(train_manager.device)
        print("[Results] transferring domain{}".format(domain))

        if need_test:
            test_loader = get_dataloader(test_set, False, batch_size, deep_rai_collate_fn)
            x = train_manager.test_or_val(val_model, test_loader)
            acc_auc_ap[v_i, 0, 0], _, acc_auc_ap[v_i, 0, 1], acc_auc_ap[v_i, 0, 2], pred_label, true_label = x
            class_acc = get_acc_per_class(pred_label, true_label)
            for class_idx, class_acc in enumerate(class_acc):
                print(f'Class {class_idx}: Accuracy {class_acc:.4f}')
        if test_index is not None:
            for i, t_i in enumerate(test_index, start=1 if need_test else 0):
                test_set = dataset_splitter.get_dataset([t_i])
                test_loader = get_dataloader(test_set, False, batch_size, deep_rai_collate_fn)
                print(test_set.len)
                acc_auc_ap[v_i, i, 0], _, acc_auc_ap[v_i, i, 1], acc_auc_ap[
                    v_i, i, 2], _, _ = train_manager.test_or_val(val_model, test_loader)

    test_index = [] if test_index is None else test_index
    test_index = [-1] + test_index if need_test else test_index
    cross_domain_results(model_name='dan',
                         domain=domain,
                         train_indexes=train_index,
                         val_indexes=None,
                         test_indexes=test_index,
                         res=acc_auc_ap,
                         file_name='dan_result.xlsx')


if __name__ == '__main__':
    # dataset_spliter = RAIGesDataSplitter()
    # dataset_spliter = MCDDataSplitter()
    # train_dg(dataset_splitter=dataset_spliter, domain=2, need_test=True, train_index=[1], test_index=[0, 2, 3, 4], phi=0.8)

    # train_dg(dataset_splitter=dataset_spliter, domain=1, need_test=True, train_index=[0], test_index=[1, 2, 3],
    #          phi=0.8)
    '''
    dataset_spliter = RAIGesDataSplitter()
    # train_dg(dataset_splitter=dataset_spliter, domain=3, need_test=True, train_index=[1], test_index=[0, 2, 3, 4],
    #         phi=0.9)
    #
    #train_dg(dataset_splitter=dataset_spliter, domain=2, need_test=True, train_index=[1], test_index=[0, 2, 3, 4],
    #         phi=0.8)
    train_dg(dataset_splitter=dataset_spliter, domain=1, need_test=True, train_index=[0], test_index=[1, 2, 3],
             phi=0.8)
    train_dg(dataset_splitter=dataset_spliter, domain=3, need_test=False, train_index=[0, 6, 7],
             test_index=[1, 2, 3, 4, 5, 8, 9],
             phi=0.8)
    '''
    dataset_spliter = MCDDataSplitter()
    '''
    
    #train_dg(dataset_splitter=dataset_spliter, domain=2, need_test=False, train_index=[5], test_index=[0, 1, 2, 3, 4],
    #         phi=0.8, val_aug=False)
    train_dg(dataset_splitter=dataset_spliter, domain=2, need_test=False, train_index=[0], test_index=[1, 2, 3, 4, 5],
             phi=0.8)
    train_dg(dataset_splitter=dataset_spliter, domain=2, need_test=False, train_index=[5], test_index=[0, 1, 2, 3, 4],
            phi=0.8, val_aug=False)
    train_dg(dataset_splitter=dataset_spliter, domain=1, need_test=False, train_index=[0, 1, 2, 3, 4],
            test_index=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
            phi=0.8, val_aug=False)
    
    train_dg(dataset_splitter=dataset_spliter, domain=3, need_test=True, train_index=[1],
             test_index=[0, 2, 3, 4],
             phi=0.8, val_aug=False)
    '''
    dataset_spliter = RAIGesDataSplitter()
    train_dg(dataset_splitter=dataset_spliter, domain=2, need_test=True, train_index=[1], test_index=[0, 2, 3, 4],
             phi=0.9)
    train_dg(dataset_splitter=dataset_spliter, domain=1, need_test=True, train_index=[0], test_index=[1, 2, 3],
             phi=0.9)
    train_dg(dataset_splitter=dataset_spliter, domain=3, need_test=False, train_index=[0, 6, 7],
             test_index=[1, 2, 3, 4, 5, 8, 9],
             phi=0.9)

    #train_dg(dataset_splitter=dataset_spliter, domain=1, need_test=False,
    #         train_index=[0, 1, 2, 3, 4],
    #         test_index=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    #         phi=0.8)

    train_dg(dataset_type=1, domain=4, need_test=True, train_index=None, test_index=None,
             phi=0.8)
