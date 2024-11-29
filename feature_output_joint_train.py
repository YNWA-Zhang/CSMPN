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
from Loss import search_pseudo, class_count

FE_joint=CNN()
Clfier_joint=FCN(in_channel=128, out_channel=7)

FE_joint_weightpath = "./Feat_extractor_joint.pth"
Clfier_joint_weightpath = "./Classifier_joint.pth"

FE_joint.load_state_dict(torch.load(FE_joint_weightpath))
Clfier_joint.load_state_dict(torch.load(Clfier_joint_weightpath))

sm=nn.Softmax(dim=1)

sourcefew1 = scio.loadmat('source_1.mat')
sourcefew1 = sourcefew1['source_1']
sourcefew1=torch.FloatTensor(sourcefew1)

sourcefew2 = scio.loadmat('source_2.mat')
sourcefew2 = sourcefew2['source_2']
sourcefew2=torch.FloatTensor(sourcefew2)

source1 = scio.loadmat('1kN_fft7.mat')
source1 = source1['1kN_fft7']
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

s_label = scio.loadmat('label7.mat')
s_label = s_label['label7']
s_label=torch.FloatTensor(s_label)
s_label = torch.topk(s_label, 1)[1].squeeze(1)

t_label = scio.loadmat('label7.mat')
t_label = t_label['label7']
t_label=torch.FloatTensor(t_label)
t_label = torch.topk(t_label, 1)[1].squeeze(1)

#joint
cnnout_joint_source_few1=FE_joint(sourcefew1.unsqueeze(1))
out_joint_source_few1=Clfier_joint(cnnout_joint_source_few1)
out_joint_source_few1=sm(out_joint_source_few1)
cnnout_joint_source1=FE_joint(source1.unsqueeze(1))
out_joint_source1=Clfier_joint(cnnout_joint_source1)
out_joint_source1=sm(out_joint_source1)

cnnout_joint_source_few2=FE_joint(sourcefew2.unsqueeze(1))
out_joint_source_few2=Clfier_joint(cnnout_joint_source_few2)
out_joint_source_few2=sm(out_joint_source_few2)
cnnout_joint_source2=FE_joint(source2.unsqueeze(1))
out_joint_source2=Clfier_joint(cnnout_joint_source2)
out_joint_source2=sm(out_joint_source2)

cnnout_joint_target=FE_joint(target.unsqueeze(1))
out_joint_target=Clfier_joint(cnnout_joint_target)
out_joint_target=sm(out_joint_target)

#joint
cnnout_joint_source_few1=cnnout_joint_source_few1.detach().numpy()
out_joint_source_few1=out_joint_source_few1.detach().numpy()
cnnout_joint_source1=cnnout_joint_source1.detach().numpy()
out_joint_source1=out_joint_source1.detach().numpy()

cnnout_joint_source_few2=cnnout_joint_source_few2.detach().numpy()
out_joint_source_few2=out_joint_source_few2.detach().numpy()
cnnout_joint_source2=cnnout_joint_source2.detach().numpy()
out_joint_source2=out_joint_source2.detach().numpy()

cnnout_joint_target=cnnout_joint_target.detach().numpy()
out_joint_target=out_joint_target.detach().numpy()

scio.savemat('cnnout_joint_src_few1.mat', {'cnnout_joint_src_few1':cnnout_joint_source_few1})
scio.savemat('cnnout_joint_src1.mat', {'cnnout_joint_src1':cnnout_joint_source1})
scio.savemat('cnnout_joint_src_few2.mat', {'cnnout_joint_src_few2':cnnout_joint_source_few2})
scio.savemat('cnnout_joint_src2.mat', {'cnnout_joint_src2':cnnout_joint_source2})
scio.savemat('cnnout_joint_tar.mat', {'cnnout_joint_tar':cnnout_joint_target})

scio.savemat('out_joint_src_few1.mat', {'out_joint_src_few1':out_joint_source_few1})
scio.savemat('out_joint_src1.mat', {'out_joint_src1':out_joint_source1})
scio.savemat('out_joint_src_few2.mat', {'out_joint_src_few2':out_joint_source_few2})
scio.savemat('out_joint_src2.mat', {'out_joint_src2':out_joint_source2})
scio.savemat('out_joint_tar.mat', {'out_joint_tar':out_joint_target})
