import torch
import torchvision.models as models

# 加载预训练的 MobileNetV2 模型
model = models.mobilenet_v3_small(pretrained=True)

# 切换到评估模式
model.eval()
