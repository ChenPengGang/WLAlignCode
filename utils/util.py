import torch
import numpy as np
import networkx as nx
from torch.autograd import Variable
import random
from collections import Counter
import copy
import collections
#import matplotlib.pyplot as plt

def save_candate(candate_pair,file_path):
    fileObject = open(file_path, 'w')
    for ip in candate_pair:
        fileObject.write(str(ip))
        fileObject.write('\n')
    fileObject.close()

def load_candate(file_path):
    f = open(file_path, "r")
    table = f.read()
    f.close()
    return table

def graph_draw(G_communites):
    pos = nx.spring_layout(G_communites)
    # print(edge_labels)
    nx.draw_networkx(G_communites, pos,node_size=100,font_size=0, width=0.5)
    plt.show()


def get_noise_dist(network,all_walks):
    int_all_words = [network.vocab2int[w] for w in all_walks]
    int_word_counts = Counter(int_all_words)
    total_count = len(int_all_words)
    word_freqs = {w: c / total_count for w, c in int_word_counts.items()}

    word_freqs = np.array(list(word_freqs.values()))
    unigram_dist = word_freqs / word_freqs.sum()
    noise_dist = torch.from_numpy(unigram_dist ** (0.75) / np.sum(unigram_dist ** (0.75)))
    return noise_dist

def FTsplit(int_word_counts,network):
    int_word_counts_f=copy.copy(int_word_counts)
    int_word_counts_t=copy.copy(int_word_counts)
    count_f=0
    count_t=0
    for key in int_word_counts.keys():
        if network.int2vocab[key].endswith("_foursquare"):
            int_word_counts_t[key]=0
            count_f+=int_word_counts[key]
        if network.int2vocab[key].endswith("_twitter"):
            int_word_counts_f[key]=0
            count_t+=int_word_counts[key]
    return int_word_counts_f,int_word_counts_t,count_f,count_t

def progress(percent,loss, width=50):
    if percent >= 100:
        percent = 100

    show_str = ('[%%-%ds]' % width) % (int(width * percent / 100) * "#")  # 字符串拼接的嵌套使用
    print('\r%s %d%% loss:%f' % (show_str, percent,loss), end='')

def get_G_rank(G_anchor,nx_G):
    g_result=nx.DiGraph()
    for edge in nx_G.edges():
        if edge[0] in list(G_anchor.nodes()) or edge[1] in list(G_anchor.nodes()):
            g_result.add_edge(edge[0],edge[1],weight=1)
    return g_result

def get_target(words, idx, window_size):
    ''' Get a list of words in a window around an index. '''

    R = np.random.randint(1, window_size + 1)
    start = idx - R if (idx - R) > 0 else 0
    stop = idx + R
    target_words = words[start:idx] + words[idx + 1:stop + 1]

    return list(target_words)


def get_batch(source,targets, BATCH_SIZE):
    
    for idx in range(0, len(source), BATCH_SIZE):
        batch_x, batch_y = [],[]
        x = source[idx:idx+BATCH_SIZE]
        y = targets[idx:idx+BATCH_SIZE]

        for i in range(len(x)):
            batch_x.append(x[i])
            batch_y.append(y[i])

        yield batch_x, batch_y


def get_neighbor(anchor_node,network):
    anchor_node=network.int2vocab[anchor_node]

    seed = random.randint(0, 9)
    if seed % 2 == 0:
        try:
            node_f = random.sample(network.input_link_f[anchor_node], 1)
        except KeyError:
            node_f = random.sample(network.output_link_f[anchor_node],1)
        try:
            node_t = random.sample(network.input_link_t[anchor_node], 1)
        except KeyError:
            node_t = random.sample(network.output_link_t[anchor_node],1)
    else:
        try:
            node_f = random.sample(network.output_link_f[anchor_node], 1)
        except KeyError:
            node_f = random.sample(network.input_link_f[anchor_node], 1)
        try:
            node_t = random.sample(network.output_link_t[anchor_node], 1)
        except KeyError:
            node_t = random.sample(network.input_link_t[anchor_node], 1)


    node_f=node_f[0]
    node_t=node_t[0]
    return network.vocab2int[node_f],network.vocab2int[node_t]


def writeFile(out_emb,network, ouput_filename_network,pix):
    f = open(ouput_filename_network, 'w')
    f.seek(0)
    vectors = out_emb
    for k, v in network.int2vocab.items():
        if v.endswith('_label'):
            continue
        if v.endswith('_anchor'):
            if '-' in v:
                # print(v)
                # v=v.replace('_anchor','')
                # vs=v.split('-')
                # if pix=='_foursquare':
                #     v=vs[0]+'_anchor'
                # else:
                #     v=vs[1]+'_anchor'
                continue
            v=v.replace('_anchor',pix)

        if v.endswith(pix):
            f.write(v)
            f.write(" ")
            value = torch.Tensor.cpu(vectors[k])
            value = value.data.numpy()
            for val in value:
                f.write(str(val))
                f.write("|")
            f.write("\n")
    print(len(vectors))
    f.flush()
    f.close()

def record_embedding(out_emb,network):
    vectors = out_emb
    embedding_dict=dict()
    for k, v in network.int2vocab.items():
        value = torch.Tensor.cpu(vectors[k])
        embedding_dict[v]=value
    return embedding_dict

def readData(file_name, pix, anchor, graph ,graph_another):

    with open(file_name, 'r', encoding='gbk', errors='ignore') as f:
        for line in f:
            array_edge = line.split(" ", 1)

            array_edge[1] = array_edge[1].replace("\n", "")
            array_edge_0 = array_edge[0] + pix
            array_edge_1 = array_edge[0] + pix
            if array_edge[0] in anchor:
                array_edge[0] += '_anchor'
            else:
                array_edge[0] += pix
            if array_edge[1] in anchor:
                array_edge[1] += '_anchor'
            else:
                array_edge[1] += pix

            graph.add_node(array_edge[0])
            graph.add_node(array_edge[1])
            graph.add_edge(array_edge[0], array_edge[1], weight=1)

            graph_another.add_node(array_edge_0)
            graph_another.add_node(array_edge_1)
            graph_another.add_edge(array_edge_0, array_edge_1, weight=1)

    del anchor
    f.close()


def getAnchors(network,anchor_file):
    answer_list = []
    file_name = anchor_file
    # 读取自己的twitter文件
    with open(file_name, 'r', encoding='gbk', errors='ignore') as f:
        for line in f:
            array_edge = line
            array_edge = array_edge.replace("\n", "")
            answer_list.append(array_edge)
            array_edge = array_edge + '_anchor'
            network.add_node(array_edge)
    print(len(answer_list))
    return answer_list

def remove_node(nx_G):
    closure_list = []
    graph_x = nx.Graph()
    for edge in nx_G.edges():
        graph_x.add_edge(edge[0], edge[1])
    for closure in nx.connected_components(graph_x):
        closure_list.append(closure)
    if len(closure_list[0]) > len(closure_list[1]):
        for node in list(closure_list[1]):
            nx_G.remove_node(node)
    else:
        for node in list(closure_list[0]):
            nx_G.remove_node(node)
    return nx_G

def read_graph(filex,filey,anchor_file):
    '''
    Reads the input network in networkx.
    '''
    networkx_file = filex
    networky_file = filey
    graph = nx.DiGraph()
    graph_f=nx.DiGraph()
    graph_t=nx.DiGraph()

    anchor_list = getAnchors(graph,anchor_file)
    readData(networkx_file, "_foursquare", anchor_list, graph ,graph_f)
    readData(networky_file, "_twitter", anchor_list, graph ,graph_t)
    return graph,graph_f,graph_t,anchor_list

def read_graph_one(file,pix):
    '''
        Reads the input network in networkx.
    '''
    graph = nx.DiGraph()

    anchor_list = getAnchors(graph)
    readData(file, pix, anchor_list, graph)
    return graph

def get_graph_anchorBy_mark(G,mark_pair,num_mark):
    graph_anchor = nx.DiGraph()
    for edge in G.edges():
        if edge[0] in mark_pair.keys() or edge[1] in mark_pair.keys():
            graph_anchor.add_edge(edge[0],edge[1],weight=1)

    return graph_anchor

def get_graph_anchor(nx_G):
    graph_anchor = nx.DiGraph()
    for edge in nx_G.edges():
        if edge[0].endswith("_anchor") or edge[1].endswith("_anchor"):
            graph_anchor.add_edge(edge[0],edge[1],weight=1)
    return graph_anchor

def get_graph_noise(nx_G,G_anchor):

    list_n=list(nx_G.nodes())
    list_a=list(G_anchor.nodes())
    G_noise= list(set(list_n).difference(set(list_a)))
    return G_noise

def change2tensor(list):
    list = torch.Tensor(list)
    list = list.squeeze()
    list = Variable(list)
    return list
def test_candate_pair(candate_pair,what,test_anchor_list,mark_pair):
    mark=0
    miss=0
    for l in candate_pair:
        ls=l.split('-')
        lsf=ls[0].split('_')
        lst=ls[1].split('_')
        if lsf[0]==lst[0]:
            mark+=1

    print('\n',what,mark,',all all:',len(candate_pair))
    return mark,miss

def get_test_anchor(file_name):
    answer_list = []
    with open(file_name, 'r', encoding='gbk', errors='ignore') as f:
        for line in f:
            array_edge = line
            array_edge = array_edge.replace("\n", "")
            answer_list.append(array_edge)
    return answer_list