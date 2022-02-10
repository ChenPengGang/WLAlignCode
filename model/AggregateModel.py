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
