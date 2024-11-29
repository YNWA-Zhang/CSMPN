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
from model import CNN, FCN

sm=nn.Softmax(dim=1)

FE_sepa1=CNN()
Clfier_sepa1=FCN(in_channel=128, out_channel=7)

FE_sepa1_weightpath = "./Feat_extractor_sepa1.pth"
Clfier_sepa1_weightpath = "./Classifier_sepa1.pth"

FE_sepa1.load_state_dict(torch.load(FE_sepa1_weightpath))
Clfier_sepa1.load_state_dict(torch.load(Clfier_sepa1_weightpath))

sourcefew1 = scio.loadmat('source_1.mat')
sourcefew1 = sourcefew1['source_1']
sourcefew1=torch.FloatTensor(sourcefew1)

source1 = scio.loadmat('1kN_fft7.mat')
source1 = source1['1kN_fft7']
source1 = source1[:,0:1200]
source1=torch.FloatTensor(source1)

target = scio.loadmat('2kN_fft7.mat')
target = target['2kN_fft7']
target = target[:,0:1200]
target=torch.FloatTensor(target)

s_label = scio.loadmat('label7.mat')
s_label = s_label['label7']
s_label=torch.FloatTensor(s_label)
s_label = torch.topk(s_label, 1)[1].squeeze(1)

t_label = scio.loadmat('label7.mat')
t_label = t_label['label7']
t_label=torch.FloatTensor(t_label)
t_label = torch.topk(t_label, 1)[1].squeeze(1)

#source No.1
cnnout_sepa1_source_few1=FE_sepa1(sourcefew1.unsqueeze(1))
out_sepa1_source_few1=Clfier_sepa1(cnnout_sepa1_source_few1)
out_sepa1_source_few1=sm(out_sepa1_source_few1)
cnnout_sepa1_source1=FE_sepa1(source1.unsqueeze(1))
out_sepa1_source1=Clfier_sepa1(cnnout_sepa1_source1)
out_sepa1_source1=sm(out_sepa1_source1)
cnnout_sepa1_target=FE_sepa1(target.unsqueeze(1))
out_sepa1_target=Clfier_sepa1(cnnout_sepa1_target)
out_sepa1_target=sm(out_sepa1_target)

cnnout_sepa1_source_few1=cnnout_sepa1_source_few1.detach().numpy()
out_sepa1_source_few1=out_sepa1_source_few1.detach().numpy()
cnnout_sepa1_source1=cnnout_sepa1_source1.detach().numpy()
out_sepa1_source1=out_sepa1_source1.detach().numpy()
cnnout_sepa1_target=cnnout_sepa1_target.detach().numpy()
out_sepa1_target=out_sepa1_target.detach().numpy()

scio.savemat('cnnout_sepa1_src_few1.mat', {'cnnout_sepa1_src_few1':cnnout_sepa1_source_few1})
scio.savemat('cnnout_sepa1_src1.mat', {'cnnout_sepa1_src1':cnnout_sepa1_source1})
scio.savemat('cnnout_sepa1_tar.mat', {'cnnout_sepa1_tar':cnnout_sepa1_target})
scio.savemat('out_sepa1_src_few1.mat', {'out_sepa1_src_few1':out_sepa1_source_few1})
scio.savemat('out_sepa1_src1.mat', {'out_sepa1_src1':out_sepa1_source1})
scio.savemat('out_sepa1_tar.mat', {'out_sepa1_tar':out_sepa1_target})
