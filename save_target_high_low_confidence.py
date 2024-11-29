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

target = scio.loadmat('2kN_fft7.mat')
target = target['2kN_fft7']
target = target[:,0:1200]


t_label = scio.loadmat('label7.mat')
t_label = t_label['label7']

listhigh=scio.loadmat('list_high.mat')
listhigh=listhigh['list_high']

listlow=scio.loadmat('list_low.mat')
listlow=listlow['list_low']

target_high=target[listhigh,:]
target_low=target[listlow,:]

label_high=t_label[listhigh,:]
label_low=t_label[listlow,:]

target_high=torch.FloatTensor(target_high)
target_high=target_high.squeeze(0)
target_low=torch.FloatTensor(target_low)
target_low=target_low.squeeze(0)

label_high=torch.FloatTensor(label_high)
label_high=label_high.squeeze(0)
label_low=torch.FloatTensor(label_low)
label_low=label_low.squeeze(0)

scio.savemat('target_h.mat', {'target_h':target_high.numpy()})
scio.savemat('label_h.mat', {'label_h':label_high.numpy()})
scio.savemat('target_l.mat', {'target_l':target_low.numpy()})
scio.savemat('label_l.mat', {'label_l':label_low.numpy()})
