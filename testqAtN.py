import torch
import random
from torch.autograd import Variable
from utils import *
import argparse
from utils.utils4ReadData import *

def getpAtN(network_x,network_y,anchor_list,test_file):
    f_test = open(test_file)
    # f_test = open("twitter_foursquare_groundtruth/groundtruth.9.foldtrain.train.number")
    pAtN_x_map=dict()

    print('-------------------------')
    line = f_test.readline()
    all = 0
    i = 0
    while line:
        target = 0
        array_edge = line
        array_edge = array_edge.replace("\n", "")
        y = array_edge + "_twitter"
        x = array_edge + "_foursquare"
        a=0
        b=0
        if x in network_x.keys() and y in network_y.keys():
            a+=1
            sam = torch.cosine_similarity(network_x[x], network_y[y], dim=0)
            for key,value in network_y.items():
                if key==y:
                    continue
                if (torch.cosine_similarity(network_x[x], value, dim=0).double() >= sam.double()):
                    #if key.replace("_twitter", "") not in anchor_list:
                    target += 1
        else:
            print('c')
            b+=1
            all-=1

        pAtN_x_map[array_edge]=target
        all += 1
        line = f_test.readline()
        i += 1
    f_test.close()
    return pAtN_x_map


def getpAtN_Revers(network_y, network_x,anchor_list,test_file):
    f_test = open(test_file)
    # f_test = open("twitter_foursquare_groundtruth/groundtruth.9.foldtrain.train.number")
    pAtN_y_map = dict()

    print('-------------------------')
    line = f_test.readline()
    all = 0
    i = 0
    while line:
        target = 0
        array_edge = line
        array_edge = array_edge.replace("\n", "")
        y = array_edge + "_twitter"
        x = array_edge + "_foursquare"
        if x in network_x.keys() and y in network_y.keys():
            sam = torch.cosine_similarity(network_x[x], network_y[y],dim=0)
            for key,value in network_x.items():
                if key==x:
                    continue
                if (torch.cosine_similarity(network_y[y], value,dim=0).double() >= sam.double()):
                    #if key.replace("_foursquare","") not in anchor_list:
                    target += 1
        else:
            all-=1
        pAtN_y_map[array_edge] = target
        all += 1
        line = f_test.readline()
        i += 1
    f_test.close()
    return pAtN_y_map,all
def test(ouput_filename_networkx,ouput_filename_networky,networkx_file, networky_file,anchor_file,test_file):
    networkx_file = networkx_file
    networky_file = networky_file

    a=[0]*30
    b=[0]*30
    f_networkx = open(ouput_filename_networkx + ".number_T")
    f_networky = open(ouput_filename_networky + ".number_T")

    nx_G, G_f, G_t, anchor_list = read_graph(networkx_file, networky_file)

    network_x=dict()
    network_y=dict()

    line=f_networkx.readline()
    while line:
        listx = []
        line=line.replace("|\n","")
        sp=line.split(" ",1)
        vector_array=sp[1].split("|",10000)
        for x in vector_array:
            listx.append(x)
        listx=list(map(float,listx))
        vector=change2tensor(listx)
        i=0
        if sp[0].split('_')[0] in anchor_list:
            line = f_networkx.readline()
            continue
        network_x[sp[0]]=vector
        line=f_networkx.readline()
    f_networkx.close()

    line = f_networky.readline()
    while line:
        listy = []
        line = line.replace("|\n", "")
        sp = line.split(" ", 1)
        vector_array = sp[1].split("|", 10000)
        for y in vector_array:
            listy.append(y)
        listy = list(map(float, listy))
        vector = change2tensor(listy)
        if sp[0].split('_')[0] in anchor_list:
            line = f_networky.readline()
            continue
        network_y[sp[0]] = vector
        line = f_networky.readline()
    f_networky.close()

    map_x=getpAtN(network_x,network_y,anchor_list,test_file)
    map_y,all=getpAtN_Revers(network_y,network_x,anchor_list,test_file)

    for i in range(30):
        for value in map_x.values():
            if value==i:
                a[i]+=1
    for i in range(30):
        for value in map_y.values():
            if value==i:
                b[i]+=1

    for i in range(30):
        a[i]/=all
    for i in range(30):
        b[i]/=all

    for i in range(1,30):
        a[i]+=a[i-1]
        b[i]+=b[i-1]
    print(all)
    for i in range(30):
        print(i,':',(a[i]+b[i])/2,end=', ')
def change2tensor(list):
    list = torch.Tensor(list)
    list = list.squeeze()
    list = Variable(list)
    return list

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
    with open(file_name, 'r', encoding='gbk', errors='ignore') as f:
        for line in f:
            array_edge = line
            array_edge = array_edge.replace("\n", "")
            answer_list.append(array_edge)
            array_edge = array_edge + '_anchor'
            network.add_node(array_edge)
    print(len(answer_list))
    return answer_list


def read_graph(filex,filey):
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pe', help='dataset name.')
    parser.add_argument('--ratio', type=int, default=5, help='training ratio')
    args = parser.parse_args()

    data_file = args.dataset
    ratio = args.ratio
    # File parameters
    ouput_filename_networkx, \
    ouput_filename_networky, \
    networkx_file, \
    networky_file, \
    anchor_file, \
    test_file = get_data(ratio, data_file)

    test(ouput_filename_networkx,ouput_filename_networky,networkx_file, networky_file,anchor_file,test_file)