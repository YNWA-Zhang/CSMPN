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
from Loss import same_search_high, same_search_low, different_search

out_j_tar = scio.loadmat('out_joint_tar.mat')
out_j_tar = out_j_tar['out_joint_tar']
out_j_tar=torch.FloatTensor(out_j_tar)

out_s1_tar = scio.loadmat('out_sepa1_tar.mat')
out_s1_tar = out_s1_tar['out_sepa1_tar']
out_s1_tar=torch.FloatTensor(out_s1_tar)

out_s2_tar = scio.loadmat('out_sepa2_tar.mat')
out_s2_tar = out_s2_tar['out_sepa2_tar']
out_s2_tar=torch.FloatTensor(out_s2_tar)

s_label = scio.loadmat('label7.mat')
s_label = s_label['label7']
s_label=torch.FloatTensor(s_label)

weight_s1=0.9282
weight_s2=0.0718
out_s_tar=weight_s1*out_s1_tar+weight_s2*out_s2_tar

num_high,list_total_high=same_search_high(out_j_tar,out_s_tar,0.6,7)
num_low,list_total_low=same_search_low(out_j_tar,out_s_tar,0.6,0.5,7)
num_dif,list_total_dif=different_search(out_j_tar,out_s_tar,0.5,7)
list_total_high=np.array((list_total_high))
list_total_low=np.array((list_total_low))
list_total_dif=np.array((list_total_dif))
print(num_high,list_total_high)
print(num_low,list_total_low)
print(num_dif,list_total_dif)
list_total_low1=np.union1d(list_total_low,list_total_dif)
print(len(list_total_low1))

sorted_list = np.array(list(range(700)))
list_total_low11=np.setdiff1d(sorted_list,list_total_high)

scio.savemat('list_high.mat', {'list_high':list_total_high})
scio.savemat('list_low.mat', {'list_low':list_total_low11})
