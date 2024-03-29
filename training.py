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
from Loss import weights_init, entropy, data_pseudo, data_pseudo1, pro_cal3, cross_domain_loss

logging.getLogger().setLevel(logging.INFO)
learning_rate=0.01
a=5
weight_s1=0.2395
weight_s2=0.7605
b1 = np.random.randint(100,size=a)
b2 = np.random.randint(100,size=a)
b = np.concatenate([b1,b2+100, b2 +200,b2+300,b2+400,b2+500,b2+600])

source_lbd1 = scio.loadmat('source_1.mat')
source_lbd1 = source_lbd1['source_1']

source_lbd2 = scio.loadmat('source_2.mat')
source_lbd2 = source_lbd2['source_2']

source1 = scio.loadmat('SUDAbearing_0kN_fft7.mat')
source1 = source1['SUDAbearing_0kN_fft7']

source2 = scio.loadmat('SUDAbearing_1kN_fft7.mat')
source2 = source2['SUDAbearing_1kN_fft7']

target = scio.loadmat('SUDAbearing_2kN_fft7.mat')
target = target['SUDAbearing_2kN_fft7']#1000*1200

label = scio.loadmat('label7.mat')
label = label['label7']
s_label = label[b,:]
t_label = label[:,:]

#shared pseudo label
labelpse_j_src1=scio.loadmat('list_j_src1.mat')
labelpse_j_src1 = labelpse_j_src1['list_j_src1']
labelpse_j_src1=labelpse_j_src1.squeeze(0)
source1_j_pse=source1[labelpse_j_src1,:]
label_src1_j_pse=label[labelpse_j_src1,:]

labelpse_j_src2=scio.loadmat('list_j_src2.mat')
labelpse_j_src2 = labelpse_j_src2['list_j_src2']
labelpse_j_src2=labelpse_j_src2.squeeze(0)
source2_j_pse=source2[labelpse_j_src2,:]
label_src2_j_pse=label[labelpse_j_src2,:]

#individual 1 pseudo label
labelpse_s1_src1=scio.loadmat('list_s1_src1.mat')
labelpse_s1_src1 = labelpse_s1_src1['list_s1_src1']
labelpse_s1_src1=labelpse_s1_src1.squeeze(0)
source1_s1_pse=source1[labelpse_s1_src1,:]
label_src1_s1_pse=label[labelpse_s1_src1,:]

#individual 2 pseudo label
labelpse_s2_src2=scio.loadmat('list_s2_src2.mat')
labelpse_s2_src2 = labelpse_s2_src2['list_s2_src2']
labelpse_s2_src2=labelpse_s2_src2.squeeze(0)
source2_s2_pse=source2[labelpse_s2_src2,:]
label_src2_s2_pse=label[labelpse_s2_src2,:]

source_lbd1=torch.FloatTensor(source_lbd1)
source_lbd2=torch.FloatTensor(source_lbd2)
source1=torch.FloatTensor(source1)
source2=torch.FloatTensor(source2)
target=torch.FloatTensor(target)
s_label=torch.FloatTensor(s_label)
t_label=torch.FloatTensor(t_label)

source1_j_pse=torch.FloatTensor(source1_j_pse)
source2_j_pse=torch.FloatTensor(source2_j_pse)
label_src1_j_pse=torch.FloatTensor(label_src1_j_pse)
label_src2_j_pse=torch.FloatTensor(label_src2_j_pse)


source1_s1_pse=torch.FloatTensor(source1_s1_pse)
label_src1_s1_pse=torch.FloatTensor(label_src1_s1_pse)


source2_s2_pse=torch.FloatTensor(source2_s2_pse)
label_src2_s2_pse=torch.FloatTensor(label_src2_s2_pse)

sm=nn.Softmax(dim=1)

#shared alignment feature extractor and classifier
Feat_extractor_joint=CNN(in_channel=1, out_channel=7)
Classifier_joint=FCN(in_channel=128, out_channel=7)
weights_init(Feat_extractor_joint)
weights_init(Classifier_joint)
Feat_extractor_joint.cuda()
Classifier_joint.cuda()

#individual alignment feature extractor and classifier
Feat_extractor_sepa1=CNN(in_channel=1, out_channel=7)
Classifier_sepa1=FCN(in_channel=128, out_channel=7)
weights_init(Feat_extractor_sepa1)
weights_init(Classifier_sepa1)
Feat_extractor_sepa1.cuda()
Classifier_sepa1.cuda()

Feat_extractor_sepa2=CNN(in_channel=1, out_channel=7)
Classifier_sepa2=FCN(in_channel=128, out_channel=7)
weights_init(Feat_extractor_sepa2)
weights_init(Classifier_sepa2)
Feat_extractor_sepa2.cuda()
Classifier_sepa2.cuda()

source_lbd1=source_lbd1.cuda()
source_lbd2=source_lbd2.cuda()
source1=source1.cuda()
source2=source2.cuda()
target=target.cuda()
s_label=s_label.cuda()
t_label=t_label.cuda()

source1_j_pse=source1_j_pse.cuda()
source2_j_pse=source2_j_pse.cuda()
label_src1_j_pse=label_src1_j_pse.cuda()
label_src2_j_pse=label_src2_j_pse.cuda()

source1_s1_pse=source1_s1_pse.cuda()
label_src1_s1_pse=label_src1_s1_pse.cuda()

source2_s2_pse=source2_s2_pse.cuda()
label_src2_s2_pse=label_src2_s2_pse.cuda()

s_label = torch.topk(s_label, 1)[1].squeeze(1)#此处需要将onehot标签转换为普通标签
t_label = torch.topk(t_label, 1)[1].squeeze(1)#此处需要将onehot标签转换为普通标签
label_src1_j_pse = torch.topk(label_src1_j_pse, 1)[1].squeeze(1)
label_src2_j_pse = torch.topk(label_src2_j_pse, 1)[1].squeeze(1)
label_src1_s1_pse = torch.topk(label_src1_s1_pse, 1)[1].squeeze(1)
label_src2_s2_pse = torch.topk(label_src2_s2_pse, 1)[1].squeeze(1)

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
    # shared alignment
    feout_j_lbd1=Feat_extractor_joint(source_lbd1.unsqueeze(1))
    feout_j_lbd2=Feat_extractor_joint(source_lbd2.unsqueeze(1))
    feout_j_src1=Feat_extractor_joint(source1.unsqueeze(1))
    feout_j_src2=Feat_extractor_joint(source2.unsqueeze(1))
    feout_j_tar= Feat_extractor_joint(target.unsqueeze(1))
    feout_j_src1_pse = Feat_extractor_joint(source1_j_pse.unsqueeze(1))
    feout_j_src2_pse = Feat_extractor_joint(source2_j_pse.unsqueeze(1))

    src1_j_pseset=data_pseudo(feout_j_src1,feout_j_lbd1,labelpse_j_src1,7,a)
    src2_j_pseset = data_pseudo(feout_j_src2, feout_j_lbd2, labelpse_j_src2, 7, a)

    proset_j_src1=pro_cal3(src1_j_pseset,7,a)
    proset_j_src2 = pro_cal3(src2_j_pseset, 7, a)

    proset_src = weight_s1*proset_j_src1+weight_s2*proset_j_src2
    loss_j_src = cross_domain_loss(feout_j_src1_pse, proset_src, 0.05)+cross_domain_loss(feout_j_src2_pse, proset_src, 0.05)

    cout_j_src1_pse=Classifier_joint(feout_j_src1_pse)
    cout_j_src2_pse=Classifier_joint(feout_j_src2_pse)
    loss_cls = criterion(cout_j_src1_pse, label_src1_j_pse)+criterion(cout_j_src2_pse, label_src2_j_pse)
    loss_j = loss_j_src
    loss_j_total=loss_cls+loss_j
    loss_j_total.backward()
    optimizer_fe_j.step()
    optimizer_c_j.step()

    #individual alignment
    #source No.1
    feout_s1_lbd1 = Feat_extractor_sepa1(source_lbd1.unsqueeze(1))
    feout_s1_src1 = Feat_extractor_sepa1(source1.unsqueeze(1))
    feout_s1_tar = Feat_extractor_sepa1(target.unsqueeze(1))
    feout_s1_src1_pse = Feat_extractor_sepa1(source1_s1_pse.unsqueeze(1))

    src1_s1_pseset = data_pseudo(feout_s1_src1, feout_s1_lbd1, labelpse_s1_src1, 7, a)

    proset_s1_src1 = pro_cal3(src1_s1_pseset, 7, a)

    cout_s1_src1_pse = Classifier_sepa1(feout_s1_src1_pse)
    loss_cls_s1 = criterion(cout_s1_src1_pse, label_src1_s1_pse)
    loss_s1 = loss_cls_s1
    loss_s1.backward()
    optimizer_fe_s1.step()
    optimizer_c_s1.step()

    #source No.2
    feout_s2_lbd2 = Feat_extractor_sepa2(source_lbd2.unsqueeze(1))
    feout_s2_src2 = Feat_extractor_sepa2(source2.unsqueeze(1))
    feout_s2_tar = Feat_extractor_sepa2(target.unsqueeze(1))
    feout_s2_src2_pse = Feat_extractor_sepa2(source2_s2_pse.unsqueeze(1))

    src2_s2_pseset = data_pseudo(feout_s2_src2, feout_s2_lbd2, labelpse_s2_src2, 7, a)

    proset_s2_src2 = pro_cal3(src2_s2_pseset, 7, a)

    loss_s2_tar = cross_domain_loss(feout_s2_tar, proset_s2_src2, 0.05)

    cout_s2_src2_pse = Classifier_sepa2(feout_s2_src2_pse)
    loss_cls_s2 = criterion(cout_s2_src2_pse, label_src2_s2_pse)
    loss_s2 = loss_cls_s2 + loss_s2_tar
    loss_s2.backward()
    optimizer_fe_s2.step()
    optimizer_c_s2.step()


    logging.info('Epoch: {} Loss_j: {:.4f} Loss_s1: {:.4f} Loss_s2: {:.4f}'.format(epoch,  loss_j, loss_s1,loss_s2))

torch.save(Feat_extractor_joint.state_dict(), 'Feat_extractor_shared.pth')
torch.save(Classifier_joint.state_dict(), 'Classifier_shared.pth')
torch.save(Feat_extractor_sepa1.state_dict(), 'Feat_extractor_indi1.pth')
torch.save(Classifier_sepa1.state_dict(), 'Classifier_indi1.pth')
torch.save(Feat_extractor_sepa2.state_dict(), 'Feat_extractor_indi2.pth')
torch.save(Classifier_sepa2.state_dict(), 'Classifier_indi2.pth')







