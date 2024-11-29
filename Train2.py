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
from model import CNN, FCN
from Loss import weights_init, entropy2, kl_loss_compute, logit_loss4

logging.getLogger().setLevel(logging.INFO)
learning_rate=0.0001
a=10
weight_s1=0.9282
weight_s2=0.0718
b1 = np.random.randint(100,size=a)
b2 = np.random.randint(100,size=a)
b = np.concatenate([b1,b2+100, b2 +200,b2+300,b2+400,b2+500,b2+600])

target = scio.loadmat('2kN_fft7.mat')
target = target['2kN_fft7']

target_high = scio.loadmat('target_h.mat')
target_high=target_high['target_h']

target_low = scio.loadmat('target_l.mat')
target_low=target_low['target_l']

t_label = scio.loadmat('label7.mat')
t_label = t_label['label7']

label_high = scio.loadmat('label_h.mat')
label_high = label_high['label_h']

label_low = scio.loadmat('label_l.mat')
label_low = label_low['label_l']

label = scio.loadmat('label7.mat')
label = label['label7']
label=label[:,0:7]

target=torch.FloatTensor(target)
target_high=torch.FloatTensor(target_high)
target_low=torch.FloatTensor(target_low)
t_label=torch.FloatTensor(t_label)
label_high=torch.FloatTensor(label_high)
label_low=torch.FloatTensor(label_low)
label=torch.FloatTensor(label)

sm=nn.Softmax(dim=1).cuda()
criterion = nn.CrossEntropyLoss().cuda()

#joint
FE_joint=CNN()
Clfier_joint=FCN(in_channel=128, out_channel=7)
FE_joint_weightpath = "./Feat_extractor_joint.pth"
Clfier_joint_weightpath = "./Classifier_joint.pth"
FE_joint.load_state_dict(torch.load(FE_joint_weightpath))
Clfier_joint.load_state_dict(torch.load(Clfier_joint_weightpath))
FE_joint.cuda()
Clfier_joint.cuda()

#individual
FE_sepa1=CNN()
Clfier_sepa1=FCN(in_channel=128, out_channel=7)
FE_sepa1_weightpath = "./Feat_extractor_sepa1.pth"
Clfier_sepa1_weightpath = "./Classifier_sepa1.pth"
FE_sepa1.load_state_dict(torch.load(FE_sepa1_weightpath))
Clfier_sepa1.load_state_dict(torch.load(Clfier_sepa1_weightpath))
FE_sepa1.cuda()
Clfier_sepa1.cuda()

FE_sepa2=CNN()
Clfier_sepa2=FCN(in_channel=128, out_channel=7)
FE_sepa2_weightpath = "./Feat_extractor_sepa2.pth"
Clfier_sepa2_weightpath = "./Classifier_sepa2.pth"
FE_sepa2.load_state_dict(torch.load(FE_sepa2_weightpath))
Clfier_sepa2.load_state_dict(torch.load(Clfier_sepa2_weightpath))
FE_sepa2.cuda()
Clfier_sepa2.cuda()

target_high=target_high.cuda()
target_low=target_low.cuda()
target=target.cuda()
t_label=t_label.cuda()
label_high=label_high.cuda()
label_low=label_low.cuda()
label=label.cuda()
t_label = torch.topk(t_label, 1)[1].squeeze(1)
label_high = torch.topk(label_high, 1)[1].squeeze(1)
label_low = torch.topk(label_low, 1)[1].squeeze(1)

for epoch in range(0, 500):
    logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, 500) + '-' * 5)
    FE_joint.train()
    Clfier_joint.train()
    FE_sepa1.train()
    Clfier_sepa1.train()
    FE_sepa2.train()
    Clfier_sepa2.train()
    optimizer_fe_j = optim.SGD(FE_joint.parameters(), lr=learning_rate, momentum=0.9,
                             weight_decay=0.0005, nesterov=True)
    optimizer_c_j = optim.SGD(list(Clfier_joint.parameters()), lr=learning_rate, momentum=0.9,
                             weight_decay=0.0005, nesterov=True)
    optimizer_fe_s1 = optim.SGD(FE_sepa1.parameters(), lr=learning_rate, momentum=0.9,
                               weight_decay=0.0005, nesterov=True)
    optimizer_c_s1 = optim.SGD(list(Clfier_sepa1.parameters()), lr=learning_rate, momentum=0.9,
                              weight_decay=0.0005, nesterov=True)
    optimizer_fe_s2 = optim.SGD(FE_sepa2.parameters(), lr=learning_rate, momentum=0.9,
                               weight_decay=0.0005, nesterov=True)
    optimizer_c_s2 = optim.SGD(list(Clfier_sepa2.parameters()), lr=learning_rate, momentum=0.9,
                              weight_decay=0.0005, nesterov=True)

    optimizer_fe_j.zero_grad()
    optimizer_c_j.zero_grad()
    optimizer_fe_s1.zero_grad()
    optimizer_c_s1.zero_grad()
    optimizer_fe_s2.zero_grad()
    optimizer_c_s2.zero_grad()
    # joint
    fe_out_j_tarh= FE_joint(target_high.unsqueeze(1))
    fe_out_j_tarl = FE_joint(target_low.unsqueeze(1))
    fe_out_j_tar = FE_joint(target.unsqueeze(1))

    c_out_j_tarh = Clfier_joint(fe_out_j_tarh)
    c_out_j_tarl = sm(Clfier_joint(fe_out_j_tarl))
    c_out_j_tar = sm(Clfier_joint(fe_out_j_tar))
    loss_j_cls = criterion(c_out_j_tarh, label_high)
    loss_j_cls.backward()

    #source No.1
    fe_out_s1_tarh = FE_sepa1(target_high.unsqueeze(1))
    fe_out_s1_tarl = FE_sepa1(target_low.unsqueeze(1))
    fe_out_s1_tar = FE_sepa1(target.unsqueeze(1))

    c_out_s1_tarh = Clfier_sepa1(fe_out_s1_tarh)
    c_out_s1_tarl = sm(Clfier_sepa1(fe_out_s1_tarl))
    c_out_s1_tar = sm(Clfier_sepa1(fe_out_s1_tar))
    loss_s1_cls = criterion(c_out_s1_tarh, label_high)
    loss_s1_cls.backward()

    #source No.2
    fe_out_s2_tarh = FE_sepa2(target_high.unsqueeze(1))
    fe_out_s2_tarl = FE_sepa2(target_low.unsqueeze(1))
    fe_out_s2_tar = FE_sepa2(target.unsqueeze(1))

    c_out_s2_tarh = Clfier_sepa2(fe_out_s2_tarh)
    c_out_s2_tarl = sm(Clfier_sepa2(fe_out_s2_tarl))
    c_out_s2_tar = sm(Clfier_sepa2(fe_out_s2_tar))
    loss_s2_cls = criterion(c_out_s2_tarh, label_high)
    loss_s2_cls.backward()

    c_out_s_tarl = weight_s1*c_out_s1_tarl + weight_s2*c_out_s2_tarl
    c_out_s_tar = weight_s1*c_out_s1_tar + weight_s2*c_out_s2_tar

    ent1 = entropy2(c_out_s_tarl)
    ent2 = entropy2(c_out_j_tarl)
    loss_logit =0.1*logit_loss4(c_out_s_tarl,c_out_j_tarl, ent1, ent2,c_out_j_tarl.size(0))

    loss_logit.backward()

    c_out_total_tar = (c_out_j_tar + (weight_s1*c_out_s1_tar + weight_s2*c_out_s2_tar)) / 2
    pred_tar = c_out_total_tar.argmax(dim=1)
    correct_tar = torch.eq(pred_tar, t_label).float().sum().item()
    acc_tar = correct_tar / target.size(0)

    optimizer_fe_j.step()
    optimizer_c_j.step()
    optimizer_fe_s1.step()
    optimizer_c_s1.step()
    optimizer_fe_s2.step()
    optimizer_c_s2.step()

    logging.info('Epoch: {} Acc: {:.4f} Loss: {:.4f} '.format(epoch,  acc_tar,loss_logit))#Acc: {:.4f}acc_tar

torch.save(FE_joint.state_dict(), 'Feat_extractor_joint_test.pth')
torch.save(Clfier_joint.state_dict(), 'Classifier_joint_test.pth')
torch.save(FE_sepa1.state_dict(), 'Feat_extractor_sepa1_test.pth')
torch.save(Clfier_sepa1.state_dict(), 'Classifier_sepa1_test.pth')
torch.save(FE_sepa2.state_dict(), 'Feat_extractor_sepa2_test.pth')
torch.save(Clfier_sepa2.state_dict(), 'Classifier_sepa2_test.pth')








