import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 定义变分自编码器（VAE）类
class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE, self).__init__()

        # 编码器（Encoder）
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, latent_size)
        self.fc_logvar = nn.Linear(hidden_size, latent_size)

        # 解码器（Decoder）
        self.fc2 = nn.Linear(latent_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, input_size)

    def encode(self, x):
        # 编码器的前向传播过程
        x = F.relu(self.fc1(x))
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        # 重参数化技巧
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mean

    def decode(self, z):
        # 解码器的前向传播过程
        z = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(z))

    def forward(self, x):
        # 前向传播过程
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon_x = self.decode(z)
        return recon_x, mean, logvar

# 定义损失函数：重建误差 + KL 散度
def loss_function(recon_x, x, mean, logvar):
    # 计算重建误差（交叉熵损失）
    reconstruction_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # 计算 KL 散度
    kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    # 将重建误差和 KL 散度相加作为最终损失
    return reconstruction_loss + kl_divergence
