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
from model import CNN,FCN,FCN_drop
from Loss import weights_init

logging.getLogger().setLevel(logging.INFO)
learning_rate=0.001
a=10
b1 = np.random.randint(100,size=a)
b2 = np.random.randint(100,size=a)
b = np.concatenate([b1,b2+100, b2 +200,b2+300,b2+400,b2+500,b2+600])

source_lbd1 = scio.loadmat('1kN_fft7.mat')
source_lbd1 = source_lbd1['1kN_fft7']
source_lbd1=source_lbd1[b,0:1200]

source_lbd2 = scio.loadmat('3kN_fft7.mat')
source_lbd2 = source_lbd2['3kN_fft7']
source_lbd2=source_lbd2[b,0:1200]

s_label = scio.loadmat('label7.mat')
s_label = s_label['label7']
s_label = s_label[b,:]

source_lbd1=torch.FloatTensor(source_lbd1)
source_lbd2=torch.FloatTensor(source_lbd2)
s_label=torch.FloatTensor(s_label)

sm=nn.Softmax(dim=1)

#joint feature extractor and classifier
Feat_extractor_joint=CNN(in_channel=1, out_channel=7)
Classifier_joint=FCN_drop(in_channel=128, out_channel=7)
weights_init(Feat_extractor_joint)
weights_init(Classifier_joint)
Feat_extractor_joint.cuda()
Classifier_joint.cuda()

#individual feature extractor and classifier
Feat_extractor_sepa1=CNN(in_channel=1, out_channel=7)
Classifier_sepa1=FCN_drop(in_channel=128, out_channel=7)
weights_init(Feat_extractor_sepa1)
weights_init(Classifier_sepa1)
Feat_extractor_sepa1.cuda()
Classifier_sepa1.cuda()

Feat_extractor_sepa2=CNN(in_channel=1, out_channel=7)
Classifier_sepa2=FCN_drop(in_channel=128, out_channel=7)
weights_init(Feat_extractor_sepa2)
weights_init(Classifier_sepa2)
Feat_extractor_sepa2.cuda()
Classifier_sepa2.cuda()


source_lbd1=source_lbd1.cuda()
source_lbd2=source_lbd2.cuda()
s_label=s_label.cuda()
s_label = torch.topk(s_label, 1)[1].squeeze(1)

for epoch in range(0, 500):
    logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, 500) + '-' * 5)
    Feat_extractor_joint.train()
    Classifier_joint.train()
    Feat_extractor_sepa1.train()
    Classifier_sepa1.train()
    Feat_extractor_sepa2.train()
    Classifier_sepa2.train()
    optimizer_fe_j = optim.SGD(Feat_extractor_joint.parameters(), lr=learning_rate, momentum=0.9,
                             weight_decay=0.0005, nesterov=True)
    optimizer_c_j = optim.SGD(list(Classifier_joint.parameters()), lr=learning_rate, momentum=0.9,
                             weight_decay=0.0005, nesterov=True)
    optimizer_fe_s1 = optim.SGD(Feat_extractor_sepa1.parameters(), lr=learning_rate, momentum=0.9,
                               weight_decay=0.0005, nesterov=True)
    optimizer_c_s1 = optim.SGD(list(Classifier_sepa1.parameters()), lr=learning_rate, momentum=0.9,
                              weight_decay=0.0005, nesterov=True)
    optimizer_fe_s2 = optim.SGD(Feat_extractor_sepa2.parameters(), lr=learning_rate, momentum=0.9,
                               weight_decay=0.0005, nesterov=True)
    optimizer_c_s2 = optim.SGD(list(Classifier_sepa2.parameters()), lr=learning_rate, momentum=0.9,
                              weight_decay=0.0005, nesterov=True)
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer_fe_j.zero_grad()
    optimizer_c_j.zero_grad()
    optimizer_fe_s1.zero_grad()
    optimizer_c_s1.zero_grad()
    optimizer_fe_s2.zero_grad()
    optimizer_c_s2.zero_grad()
    # joint
    fe_out_j_lbd1=Feat_extractor_joint(source_lbd1.unsqueeze(1))
    fe_out_j_lbd2=Feat_extractor_joint(source_lbd2.unsqueeze(1))

    c_out_j_lbd1 = Classifier_joint(fe_out_j_lbd1)
    c_out_j_lbd2 = Classifier_joint(fe_out_j_lbd2)
    loss_cls = criterion(c_out_j_lbd1, s_label)+criterion(c_out_j_lbd2, s_label)
    loss_j=loss_cls
    loss_j.backward()
    optimizer_fe_j.step()
    optimizer_c_j.step()

    #individual 1
    fe_out_s1_lbd1 = Feat_extractor_sepa1(source_lbd1.unsqueeze(1))
    c_out_s1_lbd1 = Classifier_sepa1(fe_out_s1_lbd1)
    loss_cls_s1 = criterion(c_out_s1_lbd1, s_label)
    loss_s1 = loss_cls_s1
    loss_s1.backward()
    optimizer_fe_s1.step()
    optimizer_c_s1.step()

    #individual 2
    fe_out_s2_lbd2 = Feat_extractor_sepa2(source_lbd2.unsqueeze(1))
    c_out_s2_lbd2 = Classifier_sepa2(fe_out_s2_lbd2)
    loss_cls_s2 = criterion(c_out_s2_lbd2, s_label)
    loss_s2 = loss_cls_s2
    loss_s2.backward()
    optimizer_fe_s2.step()
    optimizer_c_s2.step()

    logging.info('Epoch: {} Loss_j: {:.4f} Loss_s1: {:.4f} Loss_s2: {:.4f} '.format(epoch,  loss_j, loss_s1,loss_s2))

torch.save(Feat_extractor_joint.state_dict(), 'Feat_extractor_joint_pre.pth')
torch.save(Classifier_joint.state_dict(), 'Classifier_joint_pre.pth')
torch.save(Feat_extractor_sepa1.state_dict(), 'Feat_extractor_sepa1_pre.pth')
torch.save(Classifier_sepa1.state_dict(), 'Classifier_sepa1_pre.pth')
torch.save(Feat_extractor_sepa2.state_dict(), 'Feat_extractor_sepa2_pre.pth')
torch.save(Classifier_sepa2.state_dict(), 'Classifier_sepa2_pre.pth')
scio.savemat('source_1.mat', {'source_1':source_lbd1.detach().cpu().numpy()})
scio.savemat('source_2.mat', {'source_2':source_lbd2.detach().cpu().numpy()})







