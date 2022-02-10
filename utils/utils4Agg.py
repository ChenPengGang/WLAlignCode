import torch
import torch.nn as nn

def get_graph_label(network,device):
    init_embed = nn.Embedding(len(network.vocab2int), len(network.vocab2int)).to(device)
    init_emb = torch.eye(len(network.vocab2int)).to(device)
    i = 0
    for node in network.G.nodes():
        if node.endswith('_anchor') or node.endswith('_mark'):
            init_embed.weight.data[network.vocab2int[node]] = init_emb[i]
            i += 1
        else:
            init_embed.weight.data[network.vocab2int[node]] = torch.zeros_like(init_emb[i])
    return init_embed


def get_graph_label(network,init_dim,device):
    init_embed = nn.Embedding(len(network.vocab2int), init_dim).to(device)
    init_emb = torch.eye(init_dim).to(device)
    i = 0
    for node in network.G.nodes():
        if node in network.mark_pair.keys():
            init_embed.weight.data[network.vocab2int[node]] = init_emb[network.mark_pair[node]]
            i += 1
        else:
            init_embed.weight.data[network.vocab2int[node]] = torch.zeros_like(init_emb[-1])
    return init_embed