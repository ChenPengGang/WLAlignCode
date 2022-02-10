import time
#from utils.utils4Fobj import *
from utils.util import *
from network import *
from utils.utils4Agg import *
from network import *
import torch.optim as optim
from trainer.MyLoss import *
from model.AggregateModel import *
from trainer.MyOptimizer import adam
#from trainer.MyTrain4Struct import *
import torch.nn as nn
#from trainer.normalizeLoss import *
from sklearn.cluster import KMeans
import collections

def get_mark_node(layers_embedding,network):
    layers_embedding_0 = F.normalize(layers_embedding)
    embedding_f_anchor = layers_embedding_0[network.only_mark_F_intlist]
    embedding_t_anchor = layers_embedding_0[network.only_mark_T_intlist]

    embedding_f_anchor_mark = layers_embedding_0[network.F_intlist]
    embedding_t_anchor_mark = layers_embedding_0[network.T_intlist]
    F2T = torch.mm(embedding_f_anchor, embedding_t_anchor_mark.T)
    T2F = torch.mm(embedding_t_anchor, embedding_f_anchor_mark.T)

    # embedding_f_anchor= layers_embedding[0][network.F_intlist]
    # embedding_t_anchor= layers_embedding[0][network.T_intlist]
    #
    # F2T = F.cosine_similarity(embedding_f_anchor.unsqueeze(1),embedding_t_anchor.unsqueeze(0),dim=2)
    # T2F = F.cosine_similarity(embedding_t_anchor.unsqueeze(1),embedding_f_anchor.unsqueeze(0),dim=2)

    idx2t = np.argmax(F2T.cpu().detach().numpy(), axis=1)
    idx2f = np.argmax(T2F.cpu().detach().numpy(), axis=1)
    idx2t_n = np.max(F2T.cpu().detach().numpy(), axis=1)
    idx2f_n = np.max(T2F.cpu().detach().numpy(), axis=1)
    print()
    print(idx2f_n)
    print(idx2t_n)

    allign_list1 = []
    allign_list2 = []
    result_old_list = []

    for t, f in zip(idx2t, network.only_mark_F_intlist):
        # print(network.int2vocab[int(f)],network.int2vocab[int(network.T_intlist[t])])
        allign_list1.append(network.int2vocab[int(f)] + '-' + network.int2vocab[int(network.T_intlist[t])])
        if network.int2vocab[int(network.T_intlist[t])] in network.mark_pair.keys():
            result_old_list.append(network.int2vocab[int(f)] + '-' + network.int2vocab[int(network.T_intlist[t])])

    for f, t in zip(idx2f, network.only_mark_T_intlist):
        # print(network.int2vocab[int(network.F_intlist[f])],network.int2vocab[int(t)])
        allign_list2.append(network.int2vocab[int(network.F_intlist[f])] + '-' + network.int2vocab[int(t)])
        if network.int2vocab[int(network.F_intlist[f])] in network.mark_pair.keys():
            result_old_list.append(network.int2vocab[int(network.F_intlist[f])] + '-' + network.int2vocab[int(t)])
    # print('result old list:',result_old_list)

    result_list = list(set(allign_list1 + allign_list2))


    return result_list

def get_candate_pair(layers_embedding,network):
    layers_embedding_0 = F.normalize(layers_embedding)
    embedding_f_anchor = layers_embedding_0[network.mark_F_intlist]
    embedding_t_anchor = layers_embedding_0[network.mark_T_intlist]


    embedding_f_anchor_mark = layers_embedding_0[network.mark_F_intlist]
    embedding_t_anchor_mark = layers_embedding_0[network.mark_T_intlist]
    F2T = torch.mm(embedding_f_anchor, embedding_t_anchor_mark.T)
    T2F = torch.mm(embedding_t_anchor, embedding_f_anchor_mark.T)

    if np.all(F2T.cpu().detach().numpy()==0):
        return [],[],[]

    # embedding_f_anchor= layers_embedding[0][network.F_intlist]
    # embedding_t_anchor= layers_embedding[0][network.T_intlist]
    #
    # F2T = F.cosine_similarity(embedding_f_anchor.unsqueeze(1),embedding_t_anchor.unsqueeze(0),dim=2)
    # T2F = F.cosine_similarity(embedding_t_anchor.unsqueeze(1),embedding_f_anchor.unsqueeze(0),dim=2)

    idx2t = np.argmax(F2T.cpu().detach().numpy(), axis=1)
    idx2f = np.argmax(T2F.cpu().detach().numpy(), axis=1)
    idx2t_n = np.max(F2T.cpu().detach().numpy(), axis=1)
    idx2f_n = np.max(T2F.cpu().detach().numpy(), axis=1)
    print()
    print(idx2f_n)
    print(idx2t_n)
    for i in range(len(idx2t_n)):
        F2T[i][idx2t[i]]=0
    for i in range(len(idx2f_n)):
        T2F[i][idx2f[i]]=0

    idx2t_sec = np.argmax(F2T.cpu().detach().numpy(), axis=1)
    idx2f_sec = np.argmax(T2F.cpu().detach().numpy(), axis=1)
    idx2t_n_sec = np.max(F2T.cpu().detach().numpy(), axis=1)
    idx2f_n_sec = np.max(T2F.cpu().detach().numpy(), axis=1)
    # print(idx2f_n_sec)
    # print(idx2t_n_sec)


    allign_list1 = []
    allign_list2 = []
    result_old_list = []

    for t, f in zip(idx2t, network.mark_F_intlist):
        # print(network.int2vocab[int(f)],network.int2vocab[int(network.T_intlist[t])])
        allign_list1.append(network.int2vocab[int(f)] + '-' + network.int2vocab[int(network.mark_T_intlist[t])])
        if network.int2vocab[int(network.mark_T_intlist[t])] in network.mark_pair.keys():
            result_old_list.append(network.int2vocab[int(f)] + '-' + network.int2vocab[int(network.mark_T_intlist[t])])

    for f, t in zip(idx2f, network.mark_T_intlist):
        # print(network.int2vocab[int(network.F_intlist[f])],network.int2vocab[int(t)])
        allign_list2.append(network.int2vocab[int(network.mark_F_intlist[f])] + '-' + network.int2vocab[int(t)])
        if network.int2vocab[int(network.mark_F_intlist[f])] in network.mark_pair.keys():
            result_old_list.append(network.int2vocab[int(network.mark_F_intlist[f])] + '-' + network.int2vocab[int(t)])
    #print('result old list:',result_old_list)

    result_list=[]
    for a in allign_list1:
        if a in allign_list2:
            result_list.append(a)

    result_list2=list(set(allign_list1+allign_list2))

    result_list_un=[]
    for re in result_list2:
        if re not in result_list:
            result_list_un.append(re)
    result_list_new,_,_=get_candate_pair_new(layers_embedding,network)

    result_list.extend(result_list_new)
    return result_list,result_list_un,result_old_list


def get_candate_pair_new(layers_embedding,network):
    layers_embedding_0 = F.normalize(layers_embedding)
    embedding_f_anchor = layers_embedding_0[network.F_intlist]
    embedding_t_anchor = layers_embedding_0[network.T_intlist]


    embedding_f_anchor_mark = layers_embedding_0[network.F_intlist]
    embedding_t_anchor_mark = layers_embedding_0[network.T_intlist]
    F2T = torch.mm(embedding_f_anchor, embedding_t_anchor_mark.T)
    T2F = torch.mm(embedding_t_anchor, embedding_f_anchor_mark.T)

    # embedding_f_anchor= layers_embedding[0][network.F_intlist]
    # embedding_t_anchor= layers_embedding[0][network.T_intlist]
    #
    # F2T = F.cosine_similarity(embedding_f_anchor.unsqueeze(1),embedding_t_anchor.unsqueeze(0),dim=2)
    # T2F = F.cosine_similarity(embedding_t_anchor.unsqueeze(1),embedding_f_anchor.unsqueeze(0),dim=2)
    if np.all(F2T.cpu().detach().numpy()==0):
        return [],[],[]
    idx2t = np.argmax(F2T.cpu().detach().numpy(), axis=1)
    idx2f = np.argmax(T2F.cpu().detach().numpy(), axis=1)
    idx2t_n = np.max(F2T.cpu().detach().numpy(), axis=1)
    idx2f_n = np.max(T2F.cpu().detach().numpy(), axis=1)
    print()
    # print(idx2f_n)
    # print(idx2t_n)
    for i in range(len(idx2t_n)):
        F2T[i][idx2t[i]]=0
    for i in range(len(idx2f_n)):
        T2F[i][idx2f[i]]=0



    allign_list1 = []
    allign_list2 = []
    result_old_list = []

    for t, f in zip(idx2t, network.F_intlist):
        # print(network.int2vocab[int(f)],network.int2vocab[int(network.T_intlist[t])])
        allign_list1.append(network.int2vocab[int(f)] + '-' + network.int2vocab[int(network.T_intlist[t])])
        if network.int2vocab[int(network.T_intlist[t])] in network.mark_pair.keys():
            result_old_list.append(network.int2vocab[int(f)] + '-' + network.int2vocab[int(network.T_intlist[t])])

    for f, t in zip(idx2f, network.T_intlist):
        # print(network.int2vocab[int(network.F_intlist[f])],network.int2vocab[int(t)])
        allign_list2.append(network.int2vocab[int(network.F_intlist[f])] + '-' + network.int2vocab[int(t)])
        if network.int2vocab[int(network.F_intlist[f])] in network.mark_pair.keys():
            result_old_list.append(network.int2vocab[int(network.F_intlist[f])] + '-' + network.int2vocab[int(t)])
    #print('result old list:',result_old_list)

    result_list=[]
    for a in allign_list1:
        if a in allign_list2:
            result_list.append(a)

    result_list2=list(set(allign_list1+allign_list2))

    result_list_un=[]
    for re in result_list2:
        if re not in result_list:
            result_list_un.append(re)

    return result_list,result_list_un,result_old_list

def get_candate_pair_self(layers_embedding,network):
    layers_embedding_0 = F.normalize(layers_embedding)
    embedding_f_anchor = layers_embedding_0[network.mark_F_intlist]
    embedding_t_anchor = layers_embedding_0[network.mark_T_intlist]


    embedding_f_anchor_mark = layers_embedding_0[network.mark_F_intlist]
    embedding_t_anchor_mark = layers_embedding_0[network.mark_T_intlist]
    T2T = torch.mm(embedding_t_anchor, embedding_t_anchor_mark.T)
    F2F = torch.mm(embedding_f_anchor, embedding_f_anchor_mark.T)

    if np.all(T2T.cpu().detach().numpy()==0):
        return [],[],[]

    # embedding_f_anchor= layers_embedding[0][network.F_intlist]
    # embedding_t_anchor= layers_embedding[0][network.T_intlist]
    #
    # F2T = F.cosine_similarity(embedding_f_anchor.unsqueeze(1),embedding_t_anchor.unsqueeze(0),dim=2)
    # T2F = F.cosine_similarity(embedding_t_anchor.unsqueeze(1),embedding_f_anchor.unsqueeze(0),dim=2)

    idx2t = np.argmax(T2T.cpu().detach().numpy(), axis=1)
    idx2f = np.argmax(F2F.cpu().detach().numpy(), axis=1)
    idx2t_n = np.max(T2T.cpu().detach().numpy(), axis=1)
    idx2f_n = np.max(F2F.cpu().detach().numpy(), axis=1)
    print()
    print(idx2f_n)
    print(idx2t_n)
    for i in range(len(idx2t_n)):
        T2T[i][idx2t[i]]=0
    for i in range(len(idx2f_n)):
        F2F[i][idx2f[i]]=0

    idx2t = np.argmax(T2T.cpu().detach().numpy(), axis=1)
    idx2f = np.argmax(F2F.cpu().detach().numpy(), axis=1)
    # print(idx2f_n_sec)
    # print(idx2t_n_sec)


    allign_list1 = []
    allign_list2 = []
    result_old_list = []

    for f_t, f in zip(idx2f, network.mark_F_intlist):
        # print(network.int2vocab[int(f)],network.int2vocab[int(network.T_intlist[t])])
        allign_list1.append(network.int2vocab[int(f)] + '-' + network.int2vocab[int(network.mark_F_intlist[f_t])])
        if network.int2vocab[int(network.mark_F_intlist[f_t])] in network.mark_pair.keys():
            result_old_list.append(network.int2vocab[int(f)] + '-' + network.int2vocab[int(network.mark_F_intlist[f_t])])

    for t_f, t in zip(idx2t, network.mark_T_intlist):
        # print(network.int2vocab[int(network.F_intlist[f])],network.int2vocab[int(t)])
        allign_list2.append(network.int2vocab[int(network.mark_T_intlist[t_f])] + '-' + network.int2vocab[int(t)])
        if network.int2vocab[int(network.mark_T_intlist[t_f])] in network.mark_pair.keys():
            result_old_list.append(network.int2vocab[int(network.mark_T_intlist[t_f])] + '-' + network.int2vocab[int(t)])
    #print('result old list:',result_old_list)
    allign_count1=dict(collections.Counter(allign_list1))
    allign_list1=[]
    for key,value in allign_count1.items():
        if value>1:
            allign_list1.append(key)
    allign_count2 = dict(collections.Counter(allign_list2))
    allign_list2 = []
    for key, value in allign_count2.items():
        if value > 1:
            allign_list2.append(key)
    result_list=list(set(allign_list1+allign_list2))

    return result_list,[],result_old_list



def get_candate_dict(candate_pair,all_candate_dict):
    for pair in candate_pair:
        if pair not in all_candate_dict.keys():
            all_candate_dict[pair]=0
        else:
            all_candate_dict[pair]+=1
    return all_candate_dict

def remark(all_candate,network):
    result_candate=[]
    for can in all_candate:
        cans=can.split('-')
        if network.mark_pair[cans[0]]==network.mark_pair[cans[1]]:
            result_candate.append(can)
    result_candate=list(set(result_candate))
    return result_candate

def get_bipartite(layers_embedding,network,graph_B):
    layers_embedding_0 = F.normalize(layers_embedding)
    embedding_f_anchor = layers_embedding_0[network.mark_F_intlist]
    embedding_t_anchor = layers_embedding_0[network.mark_T_intlist]

    embedding_f_anchor_mark = layers_embedding_0[network.mark_F_intlist]
    embedding_t_anchor_mark = layers_embedding_0[network.mark_T_intlist]
    F2T = torch.mm(embedding_f_anchor, embedding_t_anchor_mark.T)
    T2F = torch.mm(embedding_t_anchor, embedding_f_anchor_mark.T)

    if np.all(F2T.cpu().detach().numpy() == 0):
        return [], [], []

    values2t, indices2t = F2T.topk(30, dim=1, largest=True, sorted=True)
    values2f, indices2f = T2F.topk(30, dim=1, largest=True, sorted=True)

    for i,(values,indexs) in enumerate(zip(values2t,indices2t)):
        for value,index in zip(values,indexs):
            if value!=0:
                graph_B.add_edge(network.int2vocab[int(network.mark_F_intlist[i])],
                                 network.int2vocab[int(network.mark_T_intlist[index])],weight=float(value))
    for i,(values,indexs) in enumerate(zip(values2f,indices2f)):
        for value,index in zip(values,indexs):
            if value!=0:
                graph_B.add_edge(network.int2vocab[int(network.mark_T_intlist[i])],
                                 network.int2vocab[int(network.mark_F_intlist[index])],weight=float(value))
