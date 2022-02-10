import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
from tqdm import tqdm as tqdm
import torch.nn.functional as F

class GraphConvolution(Module):
    def __init__(self, in_features, out_features,W=True, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight.data.uniform_(-1,1)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.W = W

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, x, adj):
        if self.W:
            support = torch.mm(x, self.weight)
        else:
            support = x

        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GraphDecoder(Module):
    def __init__(self,in_feature,out_feature):
        super(GraphDecoder, self).__init__()
        self.in_feature=in_feature
        self.out_feature=out_feature
        self.decoder=nn.Linear(in_feature,out_feature)

    def forward(self,embedding):
        x=self.decoder(embedding)
        return x

class ProcessUnit(torch.nn.Module):
    def __init__(self,input_dim,emb_dim,device):
        super(ProcessUnit, self).__init__()
        self.encoder = GraphConvolution(input_dim, emb_dim)
        self.decoder = GraphDecoder(emb_dim,input_dim)
        self.device=device

    def rebuild_onehot(self,remap):
        emb_zero = torch.zeros_like(remap)
        emb_one = torch.ones_like(remap)
        result=torch.where(remap > 0, emb_one, emb_zero)

        return result

    # def rebuildEmb(self,emb_zero, remap):
    #
    #     for ring, index in zip(emb_zero, rebuild_decode):
    #         ring[index] = 1

    def forward(self, x, edge_index,layers):
        label_embeddings=[]
        label_labels=[]
        for i in range(layers):
            embedding = self.encoder(x, edge_index)
            label_embeddings.append(embedding)
            re_map = self.decoder(embedding)
            label_labels.append(re_map)
            x=self.rebuild_onehot(re_map)

        return label_embeddings,label_labels

class AggregateLabel(torch.nn.Module):
    def __init__(self,input_dim,emb_dim,device):
        super(AggregateLabel, self).__init__()
        self.aggregater = GraphConvolution(input_dim, emb_dim,W=False,bias=False)
        self.device=device

    def forward(self, x, edge_index,layer):
        label_list=[]
        for i in range(layer):
            one_layer_label = self.aggregater(x,edge_index)
            one_layer_label_arr=one_layer_label.cpu().numpy()
            one_layer_label=torch.FloatTensor(np.int64(one_layer_label_arr > 0)).to(self.device)
            label_list.append(one_layer_label-x)
            x=one_layer_label

        return label_list

class LabelEncoder(torch.nn.Module):
    def __init__(self,input_dim,emb_dim):
        super(LabelEncoder, self).__init__()
        self.encoder = GraphConvolution(input_dim, emb_dim)
        self.map = nn.Linear(emb_dim,emb_dim)

    def forward(self, x, edge_index):
        embedding = self.encoder(x, edge_index)
        embedding = F.normalize(embedding)
        embedding = self.map(embedding)
        return embedding