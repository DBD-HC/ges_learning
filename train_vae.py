from model.vae import *
import torch.optim as optim
import torchvision.transforms as transforms
import visdom
from torch.autograd import Variable
import os
import random

import numpy as np
import torch
# 假设有一个32x32的频谱图作为输入
from torch.nn.utils.rnn import pad_sequence

from data.di_gesture_dataset import split_data, rai_data_root
from train import set_random_seed
from utils import simple_shift_list

def get_kl_loss(mu, logvar):
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_divergence


def train_vae(data_loader, in_channel, latent_dim=64, lr=1e-3, num_epochs=100):
    # 实例化 VAE 模型
    vae = VAE(in_channel=in_channel, latent_dim=latent_dim)
    vae = vae.to(device=device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)
    pre_loss = -1
    noise_factor = 0.1
    for epoch in range(num_epochs):
        running_loss = 0
        total_sample = 0
        # 假设有数据 data 作为输入
        for i, data in enumerate(data_loader):
            train_data = data[0]
            # label_data = train_data.view(train_data.shape[0], train_data.shape[1], -1)
            #             label_max = torch.max(label_data, dim=-1).values[:, :, None]
            #             label_min = torch.min(label_data, dim=-1).values[:, :, None]
            #             label_data = (label_data - label_min) / (label_max - label_min + 1e-9)
            #             label_data = label_data.view(train_data.shape[0], train_data.shape[1], train_data.shape[2],
            #                                          train_data.shape[3])
            # 正向传播
            train_data = train_data.to(device)

            # target_data = cfar_batch(train_data).to(device)
            # noise_data = train_data + noise_factor * torch.randn_like(train_data)
            # noise_data = noise_data.to(device)
            reconstructed, mu, logvar = vae(train_data)
            loss = criterion(reconstructed, train_data)  # 重构损失
            # 加入 KL 散度正则项，帮助学习更好的潜在表示
            kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss += kl_divergence
            running_loss += loss.item()

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_sample += train_data.size(0)
            if i % 5 == 4:  # print every 5 mini-batches
                print('Training [%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / total_sample))
        running_loss = running_loss / total_sample
        print('[Train] lr:%.6f, all_samples: %.5f, loss: %.5f' % (lr, total_sample, running_loss))

        if pre_loss == -1 or pre_loss > running_loss:
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss
            }, 'checkpoint/vae_model_full.pth')
            torch.save({
                'model_state_dict': vae.state_dict(),
            }, 'checkpoint/vae_model_final.pth')
            print('saved')
        pre_loss = running_loss

        vis.line(X=np.array([epoch + 1]), Y=np.array([[running_loss]]), win='loss',
                 update='append',
                 opts=dict(title='train loss', xlabel='Epoch', ylabel='loss', legend=['train']))
        test_data = np.load(os.path.join(rai_data_root, 'y_Pull_e1_u6_p5_s5.npy'))
        with torch.no_grad():
            test_data = torch.from_numpy(data_transform(test_data, None)[0]).type(torch.float32).to(device)[None, :]
            reconstructed_test_data, _, _ = vae(test_data)
            vis.heatmap(X=test_data[0, 10], win='test data', opts=dict(title='test rai'))
            vis.heatmap(X=reconstructed_test_data[0, 10], win='recon data', opts=dict(title='reconstructed rai'))


def get_dataloader(data, collate_fn):
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=8,
                                         pin_memory=True,
                                         worker_init_fn=seed_worker,
                                         collate_fn=collate_fn)
    return loader


def seed_worker(worker_id):
    seed = worker_id + random_seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def dynamic_sequence_collate_fn(datas_and_labels):
    datas_and_labels = sorted(datas_and_labels, key=lambda x: x[0].size()[0], reverse=True)
    datas = [item[0] for item in datas_and_labels]
    labels = torch.stack([item[2] for item in datas_and_labels])
    indexes = [item[3] for item in datas_and_labels]
    data_lengths = [len(x) for x in datas]
    datas = pad_sequence(datas, batch_first=True, padding_value=0)
    # datas = trans2xy(datas)
    return datas, labels, torch.tensor(data_lengths), indexes


def train_cube_out_set():
    train_set, test_set = split_data(domain='cross_person', person_index=3)
    # train_set.transform = data_transform
    train_dataloader = get_dataloader(train_set, dynamic_sequence_collate_fn)
    train_vae(train_dataloader, in_channel=1)

def data_transform(d, label, pos=None):
    # data = 10 * np.log10(data)
    max_d = np.max(d)
    min_d = np.min(d)
    d = (d - min_d)/(max_d - min_d)
    return d, label

def calculate_alpha(N, PFA):
    alpha = N * (np.power(PFA, -1/N) - 1)
    return alpha

if __name__ == '__main__':
    vis = visdom.Visdom(env='vae train', port=6006)
    random_seed = 2023
    set_random_seed(random_seed)
    batch_size = 128
    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 二元交叉熵损失函数用于像素级别的重构
    # criterion = nn.BCELoss()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_cube_out_set()