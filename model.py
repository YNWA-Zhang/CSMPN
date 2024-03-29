import torch
import os
import numpy as np
import pandas as pd
import logging
import time
import warnings
from torchvision import models
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import scipy.io as scio
from itertools import islice
from torch.autograd import Function
from torch.nn.utils.weight_norm import WeightNorm
from torch.utils.data import TensorDataset,DataLoader
from tqdm import tqdm

class GradReverse(Function):

    # 重写父类方法的时候，最好添加默认参数，不然会有warning（为了好看。。）
    @ staticmethod
    def forward(ctx, x, lambd, **kwargs: None):
        #　其实就是传入dict{'lambd' = lambd}
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, *grad_output):
        # 传入的是tuple，我们只需要第一个
        return grad_output[0] * -ctx.lambd, None

    # 这样写是没有warning，看起来很舒服，但是显然是多此一举咯，所以也可以改写成

    def backward(ctx, grad_output):
        # 直接传入一格数
        return grad_output * -ctx.lambd, None

class CNN(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, out_channel=10):
        super(CNN, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=15),  # 16, 26 ,26
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True))


        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3),  # 32, 24, 24
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            )  # 32, 12,12     (24-2) /2 +1

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3),  # 64,10,10
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3),  # 128,8,8
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4))  # 128, 4,4

        self.layer5 = nn.Sequential(
            nn.Linear(128 * 4, 128),
            nn.ReLU(inplace=True))


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.layer5(x)

        return x

class FCN(nn.Module):
    def __init__(self, in_channel=128, out_channel=7):
        super(FCN, self).__init__()
        self.in_channel = in_channel
        self.fc1 = nn.Sequential(
            nn.Linear(in_channel, 64),
            nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(
            nn.Linear(64, out_channel))


    def forward(self, x):

        x = self.fc1(x)
        x = self.fc2(x)

        return x

class Dis(nn.Module):
    def __init__(self, in_channel=1200, out_channel=2):
        super(Dis, self).__init__()
        self.in_channel = in_channel
        self.fc1 = nn.Sequential(
            nn.Linear(in_channel, 400))
        self.fc2 = nn.Sequential(
            nn.Linear(400, 100),
            nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(
            nn.Linear(100, out_channel))

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class FCN_drop(nn.Module):
    def __init__(self, in_channel=128, out_channel=7):
        super(FCN_drop, self).__init__()
        self.in_channel = in_channel
        self.fc1 = nn.Sequential(
            nn.Linear(in_channel, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(64, out_channel))


    def forward(self, x):

        x = self.fc1(x)
        x = self.fc2(x)

        return x
