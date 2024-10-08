import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from utils import get_after_conv_size


class CausalConv1D(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1D, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1D, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result


class CustTCNLayer(CausalConv1D):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(CustTCNLayer, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CustTCNLayer, self).forward(input)
        result = torch.nn.functional.relu(result)
        return result + input


class TinyRadarNN(torch.nn.Module):

    def __init__(self, numberOfSensors, numberOfRangePointsPerSensor, lengthOfWindow, numberOfTimeSteps,
                 numberOfGestures):
        # Parameters that need to be consistent with the dataset
        super(TinyRadarNN, self).__init__()
        self.lWindow = lengthOfWindow
        self.nRangePoints = numberOfRangePointsPerSensor
        self.nSensors = numberOfSensors
        self.nTimeSteps = numberOfTimeSteps
        self.nGestures = numberOfGestures

        self.CNN = torch.nn.Sequential(*self.CreateCNN())
        self.TCN = torch.nn.Sequential(*self.CreateTCN())
        self.Classifier = torch.nn.Sequential(*self.CreateClassifier())

    def CreateCNN(self):
        cnnlayers = []
        cnnlayers += [torch.nn.Conv2d(in_channels=self.nSensors, out_channels=16, kernel_size=(3, 5), padding=(1, 2))]
        cnnlayers += [torch.nn.ReLU()]
        cnnlayers += [torch.nn.MaxPool2d(kernel_size=(3, 5), stride=(3, 5), padding=(0, 0))]
        cnnlayers += [torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 5), padding=(1, 2))]
        cnnlayers += [torch.nn.ReLU()]
        cnnlayers += [torch.nn.MaxPool2d(kernel_size=(3, 5), stride=(3, 5), padding=(0, 0))]
        cnnlayers += [torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 7), padding=(0, 3))]
        cnnlayers += [torch.nn.ReLU()]
        cnnlayers += [torch.nn.MaxPool2d(kernel_size=(1, 7), stride=(1, 7), padding=(0, 0))]
        cnnlayers += [torch.nn.Flatten(start_dim=1, end_dim=-1)]
        return cnnlayers

    def CreateTCN(self):
        tcnlayers = []
        tcnlayers += [CausalConv1D(in_channels=384, out_channels=32, kernel_size=1)]
        tcnlayers += [CustTCNLayer(in_channels=32, out_channels=32, kernel_size=2, dilation=1)]
        tcnlayers += [CustTCNLayer(in_channels=32, out_channels=32, kernel_size=2, dilation=2)]
        tcnlayers += [CustTCNLayer(in_channels=32, out_channels=32, kernel_size=2, dilation=4)]
        return tcnlayers

    def CreateClassifier(self):
        classifier = []
        classifier += [torch.nn.Flatten(start_dim=1, end_dim=-1)]
        classifier += [torch.nn.Linear(32, 64)]
        classifier += [torch.nn.ReLU()]
        classifier += [torch.nn.Linear(64, 32)]
        classifier += [torch.nn.ReLU()]
        classifier += [torch.nn.Linear(32, self.nGestures)]
        return classifier

    def forward(self, x):
        cnnoutputs = []
        for i in range(self.nTimeSteps):
            cnnoutputs += [self.CNN(x[i])]
        tcninput = torch.stack(cnnoutputs, dim=2)
        tcnoutput = self.TCN(tcninput)
        classifierinput = tcnoutput.permute(0, 2, 1)
        outputs = []
        for i in range(self.nTimeSteps):
            outputs += [self.Classifier(classifierinput[:, i])]
        outputs = torch.stack(outputs, dim=1)
        return outputs.permute(1, 0, 2)


class DiGesture(nn.Module):
    def __init__(self, input_size=(32, 32), channel_num=(8, 16, 32), spatial_feat_size=128, hidden_size=128, out_size =7):
        super(DiGesture, self).__init__()
        self.conv1 = nn.Conv2d(1, channel_num[0], 3, bias=False, stride=2)
        self.bn1 = nn.BatchNorm2d(channel_num[0])
        self.input_size = input_size
        self.spatial_feat_size = spatial_feat_size
        self.conv2 = nn.Conv2d(channel_num[0], channel_num[1], 3, bias=False, stride=2)
        self.bn2 = nn.BatchNorm2d(channel_num[1])
        self.conv3 = nn.Conv2d(channel_num[1], channel_num[2], 3, bias=False, stride=2)
        self.bn3 = nn.BatchNorm2d(channel_num[2])
        self.maxpool = nn.MaxPool2d(2, ceil_mode=True)
        self.lstm = nn.LSTM(input_size=spatial_feat_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        linear_input = channel_num[-1] * get_after_conv_size(size=input_size[0], kernel_size=3, layer=3,
                                                             reduction=2) \
                       * get_after_conv_size(size=input_size[1], kernel_size=3, layer=3, reduction=2)
        self.fc_2 = nn.Linear(linear_input, spatial_feat_size)
        self.dropout = nn.Dropout(p=0.5)
        self.fc_3 = nn.Linear(hidden_size, out_size)
        # self.flatten = nn.Flatten
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, data_length, **kwargs):
        batch_size = x.size(0)
        padding_len = x.size(1)
        h = x.size(2)
        w = x.size(3)
        x = x.view(-1, 1, h, w)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = x.view(batch_size, padding_len, -1)
        x = self.fc_2(x)
        x = self.dropout(x)
        # x = x.view(batch_size, padding_len, 128)
        x = pack_padded_sequence(x, data_length.cpu(), batch_first=True)
        output, (h_n, c_n) = self.lstm(x)
        # output, out_len = pad_packed_sequence(output, batch_first=True)
        x = h_n[-1]
        x = self.fc_3(x)
        # x = self.softmax(x)

        return x


class FrameBlock1(nn.Module):
    def __init__(self, feat_size=(32, 32), in_channels=6):
        super(FrameBlock1, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=(2, 1), ceil_mode=True)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.active1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.active2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.active3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(in_channels)
        self.active4 = nn.ReLU()

    def forward(self, x):
        x = self.avg_pool(x)
        w = x.size(-1)
        h = x.size(-2)
        c = x.size(-3)
        batch_size = x.size(0)
        x = torch.transpose(x, -3, -2)
        x = x.contiguous()
        res = x.view(-1, c, w)
        x = self.conv1(res)
        x = self.bn1(x)
        x = self.active1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = res + x
        x = self.active2(x)
        x = x.view(batch_size, h, c, w)
        x = torch.transpose(x, -3, -2)
        x = x.contiguous()

        res = x
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.active3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = res + x
        x = self.active4(x)
        return x


class FrameBlock2(nn.Module):
    def __init__(self, feat_size=(32, 32), in_channels=6, need_shortcut=True):
        super(FrameBlock2, self).__init__()
        self.point1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.active1 = nn.ReLU()
        self.dw2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, groups=in_channels,
                             bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.active2 = nn.ReLU()
        self.point3 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.active3 = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2)
        self.need_shortcut = need_shortcut
        if need_shortcut:
            self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, stride=2, kernel_size=1,
                                      bias=False)
            self.active4 = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.point1(x)
        x = self.bn1(x)
        x = self.active1(x)

        x = self.dw2(x)
        x = self.bn2(x)
        x = self.active2(x)

        x = self.point3(x)
        x = self.bn3(x)
        x = self.active3(x)

        x = self.max_pool(x)

        if self.need_shortcut:
            res = self.shortcut(res)
            res = self.active4(res)
            x = x + res

        return x


class FrameModule(nn.Module):
    def __init__(self, feat_size=(32, 32), in_channels=6):
        super(FrameModule, self).__init__()
        self.block1 = FrameBlock1(in_channels=in_channels)
        self.block2 = FrameBlock2(in_channels=in_channels, need_shortcut=True)
        self.block3 = FrameBlock2(in_channels=in_channels, need_shortcut=False)
        self.dense4 = nn.Linear(96, 64)
        self.active4 = nn.ReLU()
        self.dense5 = nn.Linear(64, 32)
        self.active5 = nn.ReLU()

    def forward(self, x, data_len=None):
        padding_len = x.size(1)
        batch_size = x.size(0)
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(batch_size, padding_len, -1)
        x = self.dense4(x)
        x = self.active4(x)
        x = self.dense5(x)
        x = self.active5(x)

        return x


class TemporalModule(nn.Module):
    def __init__(self, hidden_size=32, out_size=7):
        super(TemporalModule, self).__init__()
        self.lstm1 = nn.LSTM(hidden_size=hidden_size, input_size=hidden_size, batch_first=True, num_layers=1)
        self.dense2 = nn.Linear(in_features=hidden_size, out_features=out_size)

    def forward(self, x, data_len=None):
        x = pack_padded_sequence(x, data_len.cpu(), batch_first=True)
        x, (h, c) = self.lstm1(x)
        x = h[-1]
        x = self.dense2(x)
        return x


class RadarNet(nn.Module):
    def __init__(self, feat_size=(32, 32), in_channel=8, hidden_size=32, out_size=7):
        super(RadarNet, self).__init__()
        self.frame_model = FrameModule(feat_size=feat_size, in_channels=in_channel)
        self.temporal_model = TemporalModule(hidden_size=hidden_size, out_size=out_size)

    def forward(self, x, data_len=None):
        x = self.frame_model(x)
        x = self.temporal_model(x, data_len)

        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channel, kernel_size=5, reduction=(2, 2)):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channel, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.mp = nn.MaxPool2d(kernel_size=reduction, ceil_mode=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.mp(x)
        return x


class MSBlock(nn.Module):
    def __init__(self, in_channel=1, n_channels=(8, 16, 32, 64, 128, 256, 512)):
        super(MSBlock, self).__init__()
        self.conv = nn.Sequential()
        pre_c = in_channel
        for i, c in enumerate(n_channels):
            self.conv.add_module('conv' + str(i), ConvBlock(pre_c, c))
            pre_c = c
        self.fc = nn.Linear(n_channels, n_channels)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MSCNN(nn.Module):
    def __init__(self, in_channel=1, n_channels=(8, 16, 32, 64, 128, 256, 512), out_size=7):
        super(MSCNN, self).__init__()
        self.range_conv = MSBlock(in_channel=in_channel, n_channels=n_channels)
        self.doppler_conv = MSBlock(in_channel=in_channel, n_channels=n_channels)
        self.angel_conv = MSBlock(in_channel=in_channel, n_channels=n_channels)
        self.fc = nn.Linear(n_channels * 3, out_size)

    def forward(self, x):
        r = self.range_conv(x)
        d = self.doppler_conv(x)
        a = self.angel_conv(x)
        x = torch.cat((r, d, a), dim=-1)
        x = self.fc(x)
        return x

class RFDual(nn.Module):
    def __init__(self, feat_size=(32, 32), out_size=6):
        super(RFDual, self).__init__()

        linear_in = get_after_conv_size(size=feat_size[0], kernel_size=3, padding=1, layer=3, stride=2)
        linear_in = 16 * linear_in * linear_in

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.fc = nn.Linear(linear_in, 32)
        self.lstm = nn.LSTM(hidden_size=128, input_size=32, batch_first=True, num_layers=2)
        self.classifier = nn.Linear(128, out_size)

    def forward(self, x, data_len):
        batch_size = x.size(0)
        padding_len = x.size(1)
        h = x.size(2)
        w = x.size(3)
        x = x.view(-1, 1, h, w)
        x = self.conv(x)
        x = x.view(batch_size*padding_len, -1)
        x = self.fc(x)
        x = x.view(batch_size, padding_len, -1)
        x = pack_padded_sequence(x, data_len.cpu(), batch_first=True)
        output, (h_n, c_n) = self.lstm(x)
        x = self.classifier(h_n[-1])
        return x



class DeepSolid(nn.Module):
    def __init__(self, feat_size=(32, 32), out_size=6):
        super(DeepSolid, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.lrn1 = nn.LocalResponseNorm(size=3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.lrn2 = nn.LocalResponseNorm(size=3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.lrn3 = nn.LocalResponseNorm(size=3)

        linear_in = get_after_conv_size(size=feat_size[0], kernel_size=3, padding=0, layer=3)
        linear_in = 128 * linear_in * linear_in
        self.fc_1 = nn.Linear(linear_in, 512)
        self.fc_2 = nn.Linear(512, 512)
        self.lstm1 = nn.LSTM(hidden_size=512, input_size=512, batch_first=True, num_layers=1)
        self.dp_0 = nn.Dropout(0.4)
        self.dp_1 = nn.Dropout(0.5)
        self.fc_3 = nn.Linear(512, out_size)

    def forward(self, x):
        batch_size = x.size(0)
        w = x.size(-1)
        h = x.size(-2)
        x = x.view(-1, 1, h, w)
        x = self.conv1(x)
        x = self.lrn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.lrn2(x)
        x = F.relu(x)
        x = self.dp_0(x)
        x = self.conv3(x)
        x = self.lrn3(x)
        x = F.relu(x)
        x = self.dp_0(x)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.dp_1(x)
        x = self.fc_2(x)
        x = F.relu(x)
        x = self.dp_1(x)
        x = x.view(batch_size, -1, h, w)
        x, (_, _) = self.lstm1(x)
        x = self.dp_1(x)
        x = self.fc_3(x)
        x = torch.mean(x, dim=1)
        return x


class Resnet50Classifier(nn.Module):
    def __init__(self, in_channels=2, out_size=6):
        super(Resnet50Classifier, self).__init__()
        resnet50 = models.resnet50(pretrained=False)
        # 获取 ResNet-50 的特征提取部分
        resnet50_features = nn.Sequential(*list(resnet50.children())[:-1])

        # 获取 ResNet-50 的特征维度
        num_features_resnet50 = resnet50.fc.in_features
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            resnet50_features,
            nn.Flatten(),
            nn.Linear(num_features_resnet50, out_size)
        )


    def forward(self, x):
        x = self.classifier(x)
        return x

class MobilenetV350Classifier(nn.Module):
    def __init__(self, in_channels=2, out_size=6):
        super(MobilenetV350Classifier, self).__init__()
        mobilenet = models.mobilenet_v3_small(pretrained=False)

        # 获取 MobileNetV2 的特征提取部分
        mobilenet_features = mobilenet.features

        # 获取 MobileNetV2 的特征维度
        num_features = mobilenet_features[-1].out_channels

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            mobilenet_features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features, out_size)
        )


    def forward(self, x):
        x = self.classifier(x)
        return x


class CNN3DClassifier(nn.Module):
    def __init__(self):
        super(CNN3DClassifier, self).__init__()
        self.conv1 = nn.Conv3d(1, 4, 3)
        self.bn1 = nn.BatchNorm3d(4)
        self.conv2 = nn.Conv3d(4, 8, 3)
        self.bn2 = nn.BatchNorm3d(8)
        self.conv3 = nn.Conv3d(8, 16, 3)
        self.bn3 = nn.BatchNorm3d(16)
        self.maxpool = nn.MaxPool3d((1, 2, 2), ceil_mode=True)
        self.fc_1 = nn.Linear(13888, 512)
        self.fc_2 = nn.Linear(512, 128)
        self.fc_3 = nn.Linear(128, 12)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = x.view(-1, 13888)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = F.relu(x)
        x = self.fc_3(x)
        return x


class CNN2DClassifier(nn.Module):
    def __init__(self):
        super(CNN2DClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 16, 3)
        self.bn3 = nn.BatchNorm2d(16)
        self.maxpool = nn.MaxPool2d((1, 2), ceil_mode=True)

        self.fc_1 = nn.Linear(960, 512)
        self.fc_2 = nn.Linear(512, 128)
        self.fc_3 = nn.Linear(128, 12)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = x.view(-1, 960)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = F.relu(x)
        x = self.fc_3(x)
        return x




class MTConv(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super(MTConv, self).__init__()
        self.ds_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, groups=in_channel, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, padding=padding, stride=2),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.ds_conv(x)
        x2 = self.conv(x)
        x = x1 + x2
        x = x.view(x.size(0), x.size(1), -1)
        return x2, x

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super(LearnablePositionalEncoding, self).__init__()
        self.positional_encoding = nn.Embedding(max_len, d_model)
        self.max_len = max_len

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        return x + self.positional_encoding(positions)

class MLFF(nn.Module):
    def __init__(self):
        super(MLFF, self).__init__()
        self.conv_input = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv_layer1 = MTConv(64, 64)
        self.conv_layer2 = MTConv(64, 128)
        self.conv_layer3 = MTConv(128, 256, padding=0)
        self.down_sample1 = nn.Linear(1024, 81)
        self.down_sample2 = nn.Linear(289, 81)

    def forward(self, x):
        x_c = self.conv_input(x)
        x_c, x_d1 = self.conv_layer1(x_c)
        x_c, x_d2 = self.conv_layer2(x_c)
        _, x_d3 = self.conv_layer3(x_c)
        x_d1 = self.down_sample1(x_d1)
        x_d2 = self.down_sample2(x_d2)
        x = torch.cat((x_d1, x_d2, x_d3), dim=1)
        return x

# MLFF + Transformer
class MTNet(nn.Module):
    def __init__(self, out_size=6):
        super(MTNet, self).__init__()
        self.range_conv = MLFF()
        self.angle_conv = MLFF()
        self.learnable_pos = LearnablePositionalEncoding(81)

        # 定义一个 Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=448, nhead=9, dropout=0.2, batch_first=True)
        # 定义一个 Transformer 编码器
        self.transformer_encoder  = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.adaptive_avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(448, 128),
            nn.Linear(128, out_size)
        )


    def forward(self, x):
        f_range = self.range_conv(x[:, 0] [:, None, :])
        f_angle = self.angle_conv(x[:, 1] [:, None, :])
        x = f_angle + f_range
        x = self.learnable_pos(x)
        x = self.transformer_encoder(x)
        x = self.adaptive_avg_pooling(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x