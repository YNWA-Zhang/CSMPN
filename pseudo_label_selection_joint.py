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
from model import CNN, FCN,FCN_drop
from Loss import search_pseudo1, class_count,search_pseudo2

FE_joint=CNN()
Clfier_joint=FCN_drop(in_channel=128, out_channel=7)

FE_joint_weightpath = "./Feat_extractor_joint_pre.pth"
Clfier_joint_weightpath = "./Classifier_joint_pre.pth"

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
target = target['2kN_fft7_SNR2']
target = target[:,0:1200]
target=torch.FloatTensor(target)

s_label = scio.loadmat('label7.mat')
s_label = s_label['label7']
s_label=torch.FloatTensor(s_label)
s_label = torch.topk(s_label, 1)[1].squeeze(1)

output_src1=[]
for _ in range(10):
    with torch.no_grad():
        cnnout = FE_joint(source1.unsqueeze(1))
        output=Clfier_joint(cnnout)
        output = F.softmax(output, dim=1)
    output_src1.append(output)
output_src1 = torch.stack(output_src1, dim=0)

output_src2=[]
for _ in range(10):
    with torch.no_grad():
        cnnout = FE_joint(source2.unsqueeze(1))
        output=Clfier_joint(cnnout)
        output = F.softmax(output, dim=1)
    output_src2.append(output)
output_src2 = torch.stack(output_src2, dim=0)

output_srcfew1=[]
for _ in range(10):
    with torch.no_grad():
        cnnout = FE_joint(sourcefew1.unsqueeze(1))
        output=Clfier_joint(cnnout)
        output = F.softmax(output, dim=1)
    output_srcfew1.append(output)
output_srcfew1=torch.stack(output_srcfew1)
out_std1=torch.std(output_srcfew1, dim=0)
average_output_srcfew1 = torch.mean(output_srcfew1, dim=0)

max_value1, max_idx1 = torch.max(average_output_srcfew1, dim=1)
max_std1=out_std1.gather(1, max_idx1.view(-1,1))
std_1=torch.mean(max_std1,dim=0)
print(std_1)

output_srcfew2=[]
for _ in range(10):
    with torch.no_grad():
        cnnout = FE_joint(sourcefew2.unsqueeze(1))
        output=Clfier_joint(cnnout)
        output = F.softmax(output, dim=1)
    output_srcfew2.append(output)
output_srcfew2=torch.stack(output_srcfew2)
out_std2=torch.std(output_srcfew2, dim=0)
average_output_srcfew2 = torch.mean(output_srcfew2, dim=0)

max_value2, max_idx2 = torch.max(average_output_srcfew2, dim=1)
max_std2=out_std2.gather(1, max_idx2.view(-1,1))
std_2=torch.mean(max_std2,dim=0)
print(std_2)

num_src1,list_src1=search_pseudo2(output_src1,0.85,1.25*std_1,7)
count_src1=class_count(list_src1,7)
print(num_src1,list_src1,count_src1)

num_src2,list_src2=search_pseudo2(output_src2,0.85,1.25*std_2,7)
count_src2=class_count(list_src2,7)
print(num_src2,list_src2,count_src2)

scio.savemat('list_j_src1.mat', {'list_j_src1':list_src1})
scio.savemat('list_j_src2.mat', {'list_j_src2':list_src2})
