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
from model import CNN, Dis
from Loss import weights_init

logging.getLogger().setLevel(logging.INFO)
learning_rate=0.001
a=10
b1 = np.random.randint(100,size=a)
b2 = np.random.randint(100,size=a)
b = np.concatenate([b1,b2+100, b2 +200,b2+300,b2+400,b2+500,b2+600])

source1 = scio.loadmat('1kN_fft7.mat')
source1 = source1['1kN_fft7']
source1 = source1[:,0:1200]

source2 = scio.loadmat('3kN_fft7.mat')
source2 = source2['3kN_fft7']
source2 = source2[:,0:1200]

target = scio.loadmat('2kN_fft7.mat')
target = target['2kN_fft7']
target = target[:,0:1200]

label = scio.loadmat('label_dis.mat')
label = label['label_dis']

source1=torch.FloatTensor(source1)
source2=torch.FloatTensor(source2)
target=torch.FloatTensor(target)
label=torch.FloatTensor(label)

source=torch.cat((source1,source2),dim=0)

sm=nn.Softmax(dim=1)

Feat_extractor=CNN(in_channel=1, out_channel=7)
Dis=Dis(in_channel=128, out_channel=2)
weights_init(Feat_extractor)
weights_init(Dis)
Feat_extractor.cuda()
Dis.cuda()

source=source.cuda()
label = label.cuda()
label = torch.topk(label, 1)[1].squeeze(1)

for epoch in range(0, 1000):
    logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, 1000) + '-' * 5)
    Feat_extractor.train()
    Dis.train()
    optimizer_fe = optim.SGD(Feat_extractor.parameters(), lr=learning_rate, momentum=0.9,
                               weight_decay=0.0005, nesterov=True)
    optimizer_d = optim.SGD(list(Dis.parameters()), lr=learning_rate, momentum=0.9,
                              weight_decay=0.0005, nesterov=True)
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer_fe.zero_grad()
    optimizer_d.zero_grad()

    fe_out_src = Feat_extractor(source.unsqueeze(1))

    d_out_src = Dis(fe_out_src)
    loss_dis = criterion(d_out_src, label)
    loss_dis.backward()
    optimizer_fe.step()
    optimizer_d.step()

    logging.info('Epoch: {}  Loss_s: {:.4f} '.format(epoch, loss_dis))

torch.save(Feat_extractor.state_dict(), 'Feat_extractor_dis.pth')
torch.save(Dis.state_dict(), 'Dis_dis.pth')


