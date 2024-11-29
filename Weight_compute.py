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
from model import CNN, Dis
from Loss import search_pseudo, class_count

FE=CNN()
Dis=Dis(in_channel=128, out_channel=2)

FE_weightpath = "./Feat_extractor_dis.pth"
Dis_weightpath = "./Dis_dis.pth"

FE.load_state_dict(torch.load(FE_weightpath))
Dis.load_state_dict(torch.load(Dis_weightpath))

source1 = scio.loadmat('1kN_fft7.mat')#SUDAbearing_3kN_fft7
source1 = source1['1kN_fft7']#1000*1200
source1 = source1[:,0:1200]
source1=torch.FloatTensor(source1)

source2 = scio.loadmat('3kN_fft7.mat')
source2 = source2['3kN_fft7']
source2 = source2[:,0:1200]
source2=torch.FloatTensor(source2)

target = scio.loadmat('2kN_fft7.mat')
target = target['2kN_fft7']
target = target[:,0:1200]
target=torch.FloatTensor(target)

sm=nn.Softmax(dim=1)

cnnout_source1=FE(source1.unsqueeze(1))
out_source1=sm(Dis(cnnout_source1))
cnnout_source2=FE(source2.unsqueeze(1))
out_source2=sm(Dis(cnnout_source2))
cnnout_target=FE(target.unsqueeze(1))
out_target=sm(Dis(cnnout_target))

weight=torch.mean(out_target,dim=0)
print(weight)





