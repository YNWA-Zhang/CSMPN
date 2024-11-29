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
from model import CNN,FCN


sm=nn.Softmax(dim=1)

FE_joint=CNN()
Clfier_joint=FCN(in_channel=128, out_channel=7)
FE_joint_weightpath = "./Feat_extractor_joint_test.pth"
Clfier_joint_weightpath = "./Classifier_joint_test.pth"
FE_joint.load_state_dict(torch.load(FE_joint_weightpath))
Clfier_joint.load_state_dict(torch.load(Clfier_joint_weightpath))

FE_sepa1=CNN()
Clfier_sepa1=FCN(in_channel=128, out_channel=7)
FE_sepa1_weightpath = "./Feat_extractor_sepa1_test.pth"
Clfier_sepa1_weightpath = "./Classifier_sepa1_test.pth"
FE_sepa1.load_state_dict(torch.load(FE_sepa1_weightpath))
Clfier_sepa1.load_state_dict(torch.load(Clfier_sepa1_weightpath))

FE_sepa2=CNN()
Clfier_sepa2=FCN(in_channel=128, out_channel=7)
FE_sepa2_weightpath = "./Feat_extractor_sepa2_test.pth"
Clfier_sepa2_weightpath = "./Classifier_sepa2_test.pth"
FE_sepa2.load_state_dict(torch.load(FE_sepa2_weightpath))
Clfier_sepa2.load_state_dict(torch.load(Clfier_sepa2_weightpath))

weight_s1=0.5737
weight_s2=0.4263

target = scio.loadmat('SUDAbearing_2kN_fft7.mat')
target = target['SUDAbearing_2kN_fft7']
target = target[:,0:1200]
target=torch.FloatTensor(target)

t_label = scio.loadmat('label7.mat')
t_label = t_label['label7']
t_label=torch.FloatTensor(t_label)
t_label = torch.topk(t_label, 1)[1].squeeze(1)

cnnout_joint_target=FE_joint(target.unsqueeze(1))
out_joint_target=sm(Clfier_joint(cnnout_joint_target))

cnnout_sepa1_target=FE_sepa1(target.unsqueeze(1))
out_sepa1_target=sm(Clfier_sepa1(cnnout_sepa1_target))

cnnout_sepa2_target=FE_sepa2(target.unsqueeze(1))
out_sepa2_target=sm(Clfier_sepa2(cnnout_sepa2_target))

out_total_target = (out_joint_target + (weight_s1*out_sepa1_target + weight_s2*out_sepa2_target)) / 2
pred_tar = out_total_target.argmax(dim=1)
correct_tar = torch.eq(pred_tar, t_label).float().sum().item()
acc_tar = correct_tar / target.size(0)
print(acc_tar)

scio.savemat('out_total_tar.mat', {'out_total_tar':out_total_target.detach().numpy()})
