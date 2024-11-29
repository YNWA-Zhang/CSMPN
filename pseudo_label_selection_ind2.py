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

FE_joint_weightpath = "./Feat_extractor_sepa2_pre.pth"
Clfier_joint_weightpath = "./Classifier_sepa2_pre.pth"

FE_joint.load_state_dict(torch.load(FE_joint_weightpath))
Clfier_joint.load_state_dict(torch.load(Clfier_joint_weightpath))

sm=nn.Softmax(dim=1)

sourcefew2 = scio.loadmat('source_2.mat')
sourcefew2 = sourcefew2['source_2']
sourcefew2=torch.FloatTensor(sourcefew2)

source2 = scio.loadmat('3kN_fft7.mat')
source2 = source2['3kN_fft7']
source2 = source2[:,0:1200]
source2=torch.FloatTensor(source2)

s_label = scio.loadmat('label7.mat')
s_label = s_label['label7']
s_label=torch.FloatTensor(s_label)
s_label = torch.topk(s_label, 1)[1].squeeze(1)

output_src2=[]
for _ in range(10):
    with torch.no_grad():
        cnnout = FE_joint(source2.unsqueeze(1))
        output=Clfier_joint(cnnout)
        output = F.softmax(output, dim=1)

    output_src2.append(output)
output_src2 = torch.stack(output_src2, dim=0)

output_srcfew2=[]
for _ in range(10):
    # 将数据传递给网络进行前向传播
    with torch.no_grad():  # 确保不计算梯度
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

num_src2,list_src2=search_pseudo2(output_src2,0.85,1.25*std_2,7)
count_src2=class_count(list_src2,7)
print(num_src2,list_src2,count_src2)

scio.savemat('list_s2_src2.mat', {'list_s2_src2':list_src2})
