import torch

from dataset import *
from network import *
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_set, validation_set, test_set = split_dataset()
    batch_size = 32
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8,
                                              pin_memory=True)

    net = RDT_3DCNN_air_writing()

    net.load_state_dict(torch.load('test.pth')['model_state_dict'])
    net.to(device)

    net.eval()
    criterion = nn.CrossEntropyLoss()

    print("===========Test model===========")
    test_loss = 0
    val_all_sample = 0.0
    val_correct_sample = 0.0
    ture_label = []
    predict_label = []
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            ture_label.extend([x.item() for x in label])
            data = data.to(device)
            label = label.to(device)
            output = net(data)
            loss = criterion(output, label)
            test_loss += loss.item() * len(label)
            val_all_sample = val_all_sample + len(label)
            prediction = torch.argmax(output, 1)
            predict_label.extend([x.item() for x in prediction])
            val_correct_sample += (prediction == label).sum().float().item()
    val_acc = val_correct_sample / val_all_sample
    val_loss = test_loss / val_all_sample
    print('all test: %.5f, correct test samples: %.5f, val loss: %.5f, accuracy: %.5f' % (
        val_all_sample, val_correct_sample, val_loss, val_acc))
    cm = confusion_matrix(ture_label, predict_label)
    # 绘制热图
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig('confusion_matrix.png')
    plt.show()
