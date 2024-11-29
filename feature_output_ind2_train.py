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

FE_sepa2=CNN()
Clfier_sepa2=FCN(in_channel=128, out_channel=7)

FE_sepa2_weightpath = "./Feat_extractor_sepa2.pth"
Clfier_sepa2_weightpath = "./Classifier_sepa2.pth"

FE_sepa2.load_state_dict(torch.load(FE_sepa2_weightpath))
Clfier_sepa2.load_state_dict(torch.load(Clfier_sepa2_weightpath))

sourcefew2 = scio.loadmat('source_2.mat')
sourcefew2 = sourcefew2['source_2']
sourcefew2=torch.FloatTensor(sourcefew2)

source2 = scio.loadmat('3kN_fft7.mat')
source2 = source2['3kN_fft7']
source2 = source2[:,0:1200]
source2=torch.FloatTensor(source2)

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

#source No.2
cnnout_sepa2_source_few2=FE_sepa2(sourcefew2.unsqueeze(1))
out_sepa2_source_few2=Clfier_sepa2(cnnout_sepa2_source_few2)
out_sepa2_source_few2=sm(out_sepa2_source_few2)
cnnout_sepa2_source2=FE_sepa2(source2.unsqueeze(1))
out_sepa2_source2=Clfier_sepa2(cnnout_sepa2_source2)
out_sepa2_source2=sm(out_sepa2_source2)
cnnout_sepa2_target=FE_sepa2(target.unsqueeze(1))
out_sepa2_target=Clfier_sepa2(cnnout_sepa2_target)
out_sepa2_target=sm(out_sepa2_target)

cnnout_sepa2_source_few2=cnnout_sepa2_source_few2.detach().numpy()
out_sepa2_source_few2=out_sepa2_source_few2.detach().numpy()
cnnout_sepa2_source2=cnnout_sepa2_source2.detach().numpy()
out_sepa2_source2=out_sepa2_source2.detach().numpy()
cnnout_sepa2_target=cnnout_sepa2_target.detach().numpy()
out_sepa2_target=out_sepa2_target.detach().numpy()

scio.savemat('cnnout_sepa2_src_few2.mat', {'cnnout_sepa2_src_few2':cnnout_sepa2_source_few2})
scio.savemat('cnnout_sepa2_src2.mat', {'cnnout_sepa2_src2':cnnout_sepa2_source2})
scio.savemat('cnnout_sepa2_tar.mat', {'cnnout_sepa2_tar':cnnout_sepa2_target})
scio.savemat('out_sepa2_src_few2.mat', {'out_sepa2_src_few2':out_sepa2_source_few2})
scio.savemat('out_sepa2_src2.mat', {'out_sepa2_src2':out_sepa2_source2})
scio.savemat('out_sepa2_tar.mat', {'out_sepa2_tar':out_sepa2_target})