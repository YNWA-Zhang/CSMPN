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

sm=nn.Softmax(dim=1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.1)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight, 0, 0.01)#nn.init.normal_(m.weight, 0, 0.01),nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)#
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.1)
        m.bias.data.fill_(0)

def cos_dis(feat):
    ft=F.normalize(feat, p=2, dim=-1)
    sim = torch.mm(ft, ft.T)
    sim_self=torch.diag(sim)
    sim1=(sim.sum(0)-sim_self)/(sim.size(0)-1)
    weight=sim1.expand(ft.size(1),ft.size(0)).T
    pro=(weight*feat).sum(0)/feat.size(0)
    return weight,pro

def pro_cal(set,way,shot):
    pro_set=[]
    for i in range(0,way):
        _,pro=cos_dis(set[i*shot:(i+1)*shot,:])
        pro_set.append(pro)
    pro_set=torch.cat(pro_set).view(way, set.size(1))
    return pro_set

def pro_cal3(set,way,shot):
    pro_set=[]
    for i in range(way):
        _,pro=cos_dis(set[i])
        pro_set.append(pro)
    pro_set=torch.cat(pro_set).view(way, set[0].size(1))
    return pro_set


def contrastive_sim(ins, proto, tao):
    ins=F.normalize(ins, p=2, dim=-1)
    proto=F.normalize(proto, p=2, dim=-1)
    sim = torch.mm(ins, proto.T)/tao
    return sim

def contrastive_sim_z(ins, proto, tao):
    sim_matrix = contrastive_sim(ins, proto, tao)
    return torch.sum(sim_matrix, dim=-1)

def in_domain_vector(ins, proto, tao):
    p=contrastive_sim(ins, proto, tao)
    z=contrastive_sim_z(ins, proto, tao)
    prob=p/z.unsqueeze(-1)
    return prob

def cross_domain_loss(instances, proto, tao):
    vector=contrastive_sim(instances, proto, tao)
    loss=entropy(vector)
    return loss

def entropy(logits):
    p = F.softmax(logits, dim=-1)
    return -torch.sum(p * torch.log(p), dim=-1).mean()

def entropy2(logits):
    #p = F.softmax(logits, dim=-1)
    return -torch.sum(logits * torch.log(logits), dim=-1)

def logit_loss(pred1,pred2,ent1,ent2,label1,label2,sample_num):
    loss=[]
    for i in range(0,sample_num):
        if ent1[i]<ent2[i]:
            loss1=kl(torch.log(pred2[i,:]),label1[i,:])
            loss.append(loss1)
        else:
            loss2=kl(torch.log(pred1[i,:]),label2[i,:])
            loss.append(loss2)
    return sum(loss)

def logit_loss1(pred1,pred2,ent1,ent2,sample_num):
    loss=[]
    for i in range(0,sample_num):
        if ent1[i]<ent2[i]:
            loss1=kl(torch.log(pred2[i,:]),pred1[i,:])
            loss.append(loss1)
        else:
            loss2=kl(torch.log(pred1[i,:]),pred2[i,:])
            loss.append(loss2)
    return sum(loss)



def logit_loss4(pred1, pred2, ent1, ent2, sample_num):
    loss = []
    for i in range(0, sample_num):
        ent_list = [ent1[i], ent2[i]]
        min_ent, _ = torch.min(torch.stack(ent_list), dim=0)
        if ent1[i] == min_ent:
            loss1 = kl_loss_compute(pred2[i, :], pred1[i, :])
            loss.append(loss1)
        else:
            loss2 = kl_loss_compute(pred1[i, :], pred2[i, :])
            loss.append(loss2)
    return sum(loss)

def kl_loss_compute(pred1, pred2):
    """
    Calculate KL loss
    """
    # Compute the softmax of logits1 and logits2
    #pred1 = F.softmax(logits1, dim=1)
    #pred2 = F.softmax(logits2, dim=1)

    # Calculate the KL divergence
    #loss = torch.mean(torch.sum(pred2 * (torch.log(1e-8 + pred2) - torch.log(1e-8 + pred1)), dim=-1))
    loss=torch.mean(torch.sum(pred2 * torch.log(1e-8 + pred2 / (pred1 + 1e-8)), -1))

    return loss


def search_pseudo1(list1,confi,z,way):

    list_total=[]
    confidence=[]
    uncertainty = []
    std = torch.std(list1, dim=1)
    mean = torch.mean(list1, dim=1)
    upper_uncer = mean + z * std
    lower_uncer = mean - z * std
    for i in range(0,way):
        list2=sm(list1)
        a=list(torch.where(list2[i*100:(i+1)*100,i]>=confi))
        b=np.array(a[0]+i*100)
        confidence.extend(b)
    for j in range(len(list1)):
        sample_prediction = list1[j]  # 获取第 i 个样本的输出值
        if all(lower_uncer[j] <= sample_prediction) and all(sample_prediction <= upper_uncer[j]):
            uncertainty.append(j)
    list_total= np.intersect1d(confidence,uncertainty)
    return len(list_total), list_total

def data_pseudo(data,datafew,list,way,shot):
    pseudo=[]
    for i in range(way):
        filtered_list = [x for x in list if i*100 <= x <= (i+1)*100]
        data_total=torch.cat((data[filtered_list,:],datafew[i*shot:(i+1)*shot,:]),dim=0)
        pseudo.append(data_total)
    return pseudo

def data_pseudo1(data,list,way,shot):
    pseudo=[]
    for i in range(way):
        filtered_list = [x for x in list if i*100 <= x <= (i+1)*100]
        pseudo.append(data[filtered_list,:])
    return pseudo

def class_count(list,way):
    count=[]
    for i in range(way):
        filtered_list = [x for x in list if i*100 <= x <= (i+1)*100]
        len_filter=len(filtered_list)
        count.append(len_filter)
    return count

def same_search_high(list1,list2,minthreshold,way):
    list_total=[]
    for i in range(0,way):
        a=list(torch.where(list1[i*100:(i+1)*100,i]>=minthreshold))
        b=list(torch.where(list2[i*100:(i+1)*100,i]>=minthreshold))
        c=np.array([x for x in a[0] if x in b[0]])+i*100
        list_total.extend(c)
    return len(list_total),list_total

def same_search_low(list1,list2,maxthreshold,minthreshold,way):
    list_total=[]
    for i in range(0,way):
        a=list(torch.where(list1[i*100:(i+1)*100,i]>=minthreshold))
        b=list(torch.where(list1[i*100:(i+1)*100,i]<=maxthreshold))
        c=np.array([x for x in a[0] if x in b[0]])+i*100

        d=list(torch.where(list2[i*100:(i+1)*100,i]>=minthreshold))
        e=list(torch.where(list2[i*100:(i+1)*100,i]<=maxthreshold))
        f=np.array([x for x in d[0] if x in e[0]])+i*100
        g=np.intersect1d(c,f)
        list_total.extend(g)
    return len(list_total),list_total

def different_search(list1,list2,minthreshold,way):
    list_total = []
    for i in range(0, way):
        a=list(torch.where(list1[i*100:(i+1)*100,i]>=minthreshold))
        b=list(torch.where(list2[i*100:(i+1)*100,i]<=minthreshold))
        c=np.array([x for x in a[0] if x in b[0]])+i*100

        d=list(torch.where(list1[i*100:(i+1)*100,i]<=minthreshold))
        e=list(torch.where(list2[i*100:(i+1)*100,i]>=minthreshold))
        f=np.array([x for x in a[0] if x in b[0]])+i*100
        g=np.union1d(c,f)
        list_total.extend(g)
    return len(list_total),list_total

