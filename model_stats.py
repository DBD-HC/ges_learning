from thop import profile
from model.network import *
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = MobileNet1DV1(4, (8, 16, 32))

input1 = torch.randn(2, 4, 32).type(torch.float32)
input2 = torch.randn(1, 1, 32, 32).type(torch.float32)
input3 = torch.tensor([True, True])
input4 = torch.tensor(128).to(device)
input5 = torch.randn(128, 16, 32, 32).type(torch.float32)

input1 = input1.to(device)
input3 = input3.to(device)
net = net.to(device)

flops, params = profile(net, inputs=(input1,input3))
#
print('total FLOPs:' + str(flops))
print('total Params:' + str(params))
#
# input_sub = torch.randn(1, 128, 32, 32).type(torch.float32).to(device)
# flops, params = profile(net.sn, inputs=(input1, input3))
# print('sn FLOPs:' + str(flops))
# print('sn Params:' + str(params))



# input1 = torch.randn(1, 1, 32, 32).type(torch.float32).to(device)
# flops, params = profile(net.sn.CFAR, inputs=(input1, input4, input3))
# print('CFAR FLOPs:' + str(flops))
# print('CFAR Params:' + str(params))
