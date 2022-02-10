import random
import torch
import numpy as np
from utils import *
from tqdm import tqdm
import torch.nn.functional as F


class networkC:
    def __init__(self, nx_G,G_all, is_directed, p, q,anchor_list):
        self.G = nx_G
        self.G_all=G_all
        self.is_directed = is_directed
        self.p = p
        self.q = q
        self.anchor_list=anchor_list
        self.max_mark = 0
        self.mark_pair = self.get_mark_pair()
        self.vocab_prepare()
        self.edge_prepare()
        self.node_mark=self.dict_mark()
        self.converge_distribution=[]
        self.converge_distribution_pre=[]

    def get_G_anchor(self,G_anchor):
        self.G_anchor=G_anchor

    def get_mark_pair(self):
        mark_pair=dict()
        for node in self.G.nodes():
            if node.endswith('_anchor'):
                mark_pair[node]=self.max_mark
                self.max_mark+=1
        return mark_pair

    def is_convergence(self):
        converge_dict=dict()
        for key,value in self.mark_pair.items():
            if value not in converge_dict.keys():
                converge_dict[value]=1
            else:
                converge_dict[value]+=1
        converge_distribution=sorted(list(converge_dict.values()))

        if len(self.converge_distribution)!=len(converge_distribution) and len(self.converge_distribution_pre)!=len(converge_distribution):
            self.converge_distribution_pre = self.converge_distribution
            self.converge_distribution=converge_distribution
            return False
        else:
            if len(self.converge_distribution)==len(converge_distribution):
                for a,b in zip(self.converge_distribution,converge_distribution):
                    if a!=b:
                        self.converge_distribution_pre = self.converge_distribution
                        self.converge_distribution = converge_distribution
                        return False
            else:
                for a,b in zip(self.converge_distribution_pre,converge_distribution):
                    if a!=b:
                        self.converge_distribution_pre = self.converge_distribution
                        self.converge_distribution = converge_distribution
                        return False
        return True

    def dict_mark(self):
        node_mark=dict()
        for node in self.G_all.nodes():
            if node.endswith('_anchor'):
                node_mark[node]=True
            else:
                node_mark[node]=False
        return node_mark

    def remark_node(self,result_dict):
        for pair in result_dict:
            pairs=pair.split('-')
            self.node_mark[pairs[0]]=True
            self.node_mark[pairs[1]]=True

    def is_mark_finished(self):
        for key,value in self.node_mark.items():
            if value==False:
                return value
        return True
    def num_mark(self):
        num=0
        for key,value in self.node_mark.items():
            if value==True:
                num+=1

        return num

    def hash_mark(self):
        hash_dict=dict()
        for key,value in self.mark_pair.items():
            if value not in hash_dict.keys():
                hash_dict[value]=[key]
            else:
                hash_dict[value].append(key)
        mark_pair=dict()
        for i,value in enumerate(hash_dict.values()):
            for v in value:
                mark_pair[v]=i
        self.mark_pair=mark_pair

    def reset_anchor(self, result_list, get_graph_anchor, is_change):
        for node in result_list:
            nodes = node.split('-')
            if nodes[0] in self.mark_pair.keys() and nodes[1] not in self.mark_pair.keys():
                self.mark_pair[nodes[1]] = self.mark_pair[nodes[0]]
            elif nodes[1] in self.mark_pair.keys() and nodes[0] not in self.mark_pair.keys():
                self.mark_pair[nodes[0]] = self.mark_pair[nodes[1]]
            elif nodes[0] in self.mark_pair.keys() and nodes[1] in self.mark_pair.keys():
                if self.mark_pair[nodes[0]] == self.mark_pair[nodes[1]]:
                    pass
                else:
                    self.mark_pair[nodes[0]] = self.max_mark
                    self.mark_pair[nodes[1]] = self.max_mark
                    self.max_mark += 1
            else:
                self.mark_pair[nodes[0]] = self.max_mark
                self.mark_pair[nodes[1]] = self.max_mark
                self.max_mark += 1
        # print(self.mark_pair)
        self.hash_mark()
        self.G = get_graph_anchor(self.G, self.mark_pair, self.num_mark())

        self.vocab_prepare()
        self.edge_prepare()

    def reset_edges(self,result_dict,get_graph_anchor,candate_pair_self):
        # for node in result_dict:
        #     nodes = node.split('-')
        #     if nodes[0] in self.mark_pair.keys() and nodes[1] not in self.mark_pair.keys():
        #         self.mark_pair[nodes[1]] = self.mark_pair[nodes[0]]
        #     elif nodes[1] in self.mark_pair.keys() and nodes[0] not in self.mark_pair.keys():
        #         self.mark_pair[nodes[0]] = self.mark_pair[nodes[1]]
        #     elif nodes[0] in self.mark_pair.keys() and nodes[1] in self.mark_pair.keys():
        #         pass
        #     else:
        #         self.mark_pair[nodes[0]] = self.max_mark
        #         self.mark_pair[nodes[1]] = self.max_mark
        #         self.max_mark += 1
        for node in candate_pair_self:
            nodes = node.split('-')
            if nodes[0] in self.mark_pair.keys() and nodes[1] not in self.mark_pair.keys():
                self.mark_pair[nodes[1]] = self.mark_pair[nodes[0]]
            elif nodes[1] in self.mark_pair.keys() and nodes[0] not in self.mark_pair.keys():
                self.mark_pair[nodes[0]] = self.mark_pair[nodes[1]]
            else:
                pass
        self.G = get_graph_anchor(self.G_all, self.mark_pair, self.num_mark())

        self.vocab_prepare()
        self.edge_prepare()

    def reset_anchor2(self,result_dict,get_graph_anchor):
        for node in result_dict.keys():
            nodes=node.split('-')
            self.G.remove_node(nodes[0])
            self.G.remove_node(nodes[1])

        self.G = get_graph_anchor(self.G_all,self.mark_pair)

        self.vocab_prepare()
        self.edge_prepare()

    def reset_graph(self,graph_all,init_emb,device):
        print('reset label')
        init_emb = F.normalize(init_emb)
        self.vocab_Re_prepare(graph_all,init_emb,device)
        self.edge_prepare()
        return self.init_emb
    def get_same_label(self,all_candate,mark_pair):
        left=[]
        right=[]
        for node in all_candate:
            nodes = node.split('-')
            left_node=self.vocab2int[nodes[0]]
            right_node=self.vocab2int[nodes[1]]


            left.append(left_node)
            right.append(right_node)
        return left,right

    def set_closure_mark(self,all_candate):
        all_nodes = list(self.G.nodes())
        graph_x = nx.Graph()
        for edge in all_candate:
            nodes = edge.split('-')
            graph_x.add_edge(nodes[0], nodes[1])

        closure_list = []
        closure_noded_all = []
        nodes_left = []
        for closure in nx.connected_components(graph_x):
            closure_list.append(list(closure))
            closure_noded_all.extend(list(closure))
        for node in all_nodes:
            if node not in closure_noded_all:
                nodes_left.append(node)
        closure_list.append(nodes_left)

        closure_dict=dict()
        for i in range(len(closure_list)):
            for closure in closure_list[i]:
                closure_dict[closure]=i
        self.closure_list=closure_list
        self.closure_dict=closure_dict
        self.num_label=len(closure_list)

    def vocab_Re_prepare(self,graph_all,init_emb,device):
        self.G=graph_all
        bias=len(list(self.G.nodes()))-init_emb.shape[1]
        emb_bias=torch.zeros(init_emb.shape[0],bias).to(device)
        init_emb = torch.cat((init_emb,emb_bias),dim=1)
        eye_emb=torch.eye(len(list(self.G.nodes()))).to(device)
        emb_list=[]
        for tensor in init_emb:
            emb_list.append(tensor)
        num_node = len(list(self.vocab2int.keys()))
        for node in self.G.nodes():
            if node not in self.vocab2int.keys():
                self.vocab2int[node]=num_node
                self.int2vocab[num_node]=node
                emb_list.append(eye_emb[num_node])
                num_node+=1
        self.init_emb=torch.stack(emb_list,dim=0)
        print(len(list(self.vocab2int.keys())))
        print(self.init_emb.shape)
    def vocab_prepare(self):
        self.vocab2int = {w: c for c, w in enumerate(self.G.nodes())}
        self.int2vocab = {c: w for c, w in enumerate(self.G.nodes())}
        F_intlist=[]
        T_intlist=[]
        mark_F_intlist=[]
        mark_T_intlist = []
        for key,value in self.vocab2int.items():
            if key not in self.mark_pair.keys():
                if key.endswith('_foursquare'):
                    F_intlist.append(value)
                if key.endswith('_twitter'):
                    T_intlist.append(value)
            else:
                if key.endswith('_foursquare'):
                    mark_F_intlist.append(value)
                if key.endswith('_twitter'):
                    mark_T_intlist.append(value)
        all_list=list(self.vocab2int.values())
        intlist_p=[]
        for i in all_list:
            if i not in F_intlist+T_intlist:
                intlist_p.append(i)
        self.all_intlist_p=intlist_p
        self.all_intlist_n=F_intlist+T_intlist

        self.F_intlist=torch.LongTensor(F_intlist)
        self.T_intlist=torch.LongTensor(T_intlist)
        self.mark_F_intlist=torch.LongTensor(F_intlist+mark_F_intlist)
        self.mark_T_intlist = torch.LongTensor(T_intlist + mark_T_intlist)
        self.only_mark_F_intlist = torch.LongTensor(mark_F_intlist)
        self.only_mark_T_intlist = torch.LongTensor(mark_T_intlist)
        #self.all_intlist_n=torch.LongTensor(F_intlist+T_intlist)

    def get_intlist_p(self,length):
        p_list=[]
        for i in range(length):
            p_list.append(random.choices(self.all_intlist_p)[0])
        return torch.LongTensor(p_list)

    def get_intlist_n(self,length):
        n_list=[]
        for i in range(length):
            n_list.append(random.choices(self.all_intlist_n)[0])
        return torch.LongTensor(n_list)

    def edge_prepare(self):
        source_list = []
        target_list = []
        adj_real = torch.eye(len(self.vocab2int), len(self.vocab2int))
        for edge in self.G.edges():
            source_list.append(self.vocab2int[edge[0]])
            target_list.append(self.vocab2int[edge[1]])
            adj_real[self.vocab2int[edge[0]]][self.vocab2int[edge[1]]] = float(self.G[edge[0]][edge[1]]['weight'])
            adj_real[self.vocab2int[edge[1]]][self.vocab2int[edge[0]]] = float(self.G[edge[0]][edge[1]]['weight'])
        adjacency_matrix = [source_list, target_list]
        adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.long)
        self.adjacency_matrix = adjacency_matrix
        self.adj_real = adj_real

    def num_nodes(self):
        return len(self.vocab2int)

    def get_embedding_index(self,embedding):
        self.embedding=embedding

        embedding_dict=dict()
        vocab2embedding=dict()
        for key in self.vocab2int.keys():
            embedding_dict[key]=torch.tensor([self.vocab2int[key]])
            vocab2embedding[key]=self.embedding[embedding_dict[key]]

        self.embedding_index=embedding_dict
        self.vocab2embedding=vocab2embedding

    def embedding_rebuild(self):
        embedding_list=[]
        i=0
        self.vocab2int=dict()
        self.int2vocab=dict()
        for key,value in self.vocab2embedding.items():
            embedding_list.append(value)
            self.vocab2int[key]=i
            self.int2vocab[i]=key
            i+=1
        self.embedding=torch.stack(embedding_list, 0)

    def get_emb_dict_by_1file(self,file,anchor_list,pix):
        f_network = open(file)

        network_dict = dict()

        line = f_network.readline()
        i=0
        while line:
            listx = []
            line = line.replace("|\n", "")
            sp = line.split(" ", 1)
            vector_array = sp[1].split("|", 127)
            for x in vector_array:
                listx.append(x)
            listx = list(map(float, listx))
            vector = change2tensor(listx)
            if sp[0].replace(pix,'') in anchor_list:
                sp[0]=sp[0].replace(pix,"_anchor")
                i+=1
            network_dict[sp[0]] = vector
            line = f_network.readline()
        f_network.close()

        self.vocab2embedding=network_dict
        self.embedding_rebuild()
        return i

    def get_emb_dict_by_2file(self,file1,file2,anchor_list):
        f_networkx = open(file1)
        f_networky = open(file2)

        network_dict = dict()

        line = f_networkx.readline()
        while line:
            listx = []
            line = line.replace("|\n", "")
            sp = line.split(" ", 1)
            vector_array = sp[1].split("|", 127)
            for x in vector_array:
                listx.append(x)
            listx = list(map(float, listx))
            vector = change2tensor(listx)
            if sp[0].replace("_foursquare",'') in anchor_list:
                sp[0]=sp[0].replace("_foursquare","_anchor")
            network_dict[sp[0]] = vector
            line = f_networkx.readline()
        f_networkx.close()


        line = f_networky.readline()
        while line:
            listy = []
            line = line.replace("|\n", "")
            sp = line.split(" ", 1)
            vector_array = sp[1].split("|", 127)
            for y in vector_array:
                listy.append(y)
            listy = list(map(float, listy))
            vector = change2tensor(listy)
            if sp[0].replace("_twitter",'') in anchor_list:
                sp[0]=sp[0].replace("_twitter","_anchor")
            network_dict[sp[0]] = vector
            line = f_networky.readline()
        f_networky.close()
        self.vocab2embedding=network_dict
        self.embedding_rebuild()

    def get_anchor_index(self,anchor_list):
        anchor_index_list=[]
        for anchor in anchor_list:
            anchor+='_anchor'
            index=self.vocab2int[anchor]
            anchor_index_list.append(index)
        self.anchor_index=torch.tensor(anchor_index_list)

        return self.anchor_index


    def get_training_set(self,batch_size):
        source = []
        target = []
        edges_list = []
        mark_list = []
        for i in range(10):
            edges = random.sample(list(self.G_anchor.edges()), batch_size)
            edges_list += edges
        # edges=[random.choice(list(self.G.edges())) for _ in range(batch_size)]
        for edge in edges_list:
            mark = 0
            x = edge[0]
            y = edge[1]
            if edge[0].endswith("_anchor"):
                x = random.sample(list(self.G_anchor.successors(edge[0])), 1)[0]
                mark += 1
            if edge[1].endswith("_anchor"):
                y = random.sample(list(self.G_anchor.predecessors(edge[1])), 1)[0]
                mark += 2

            mark_list.append(mark)
            source.append(x)
            target.append(y)
            if mark != 0:
                source.append(edge[0])
                target.append(edge[1])

        return source, target,mark_list

    def get_training_set2(self,batch_size):
        source = []
        target = []
        edges_list = []
        mark_list = []
        for i in range(10):
            edges = random.sample(list(self.G.edges()), batch_size)
            edges_list += edges
        # edges=[random.choice(list(self.G.edges())) for _ in range(batch_size)]
        for edge in edges_list:
            mark = 0
            x = edge[0]
            y = edge[1]

            source.append(x)
            target.append(y)

        return source, target,mark_list

    def get_training_set_by_anchor(self,batch_size):
        source = []
        target = []
        edges_list = []
        mark_list = []
        for i in range(10):
            edges = random.sample(list(self.G_anchor.edges()), batch_size)
            edges_list += edges
            edges = random.sample(list(self.G_B.edges()), batch_size)
            edges_list += edges
        # edges=[random.choice(list(self.G.edges())) for _ in range(batch_size)]
        for edge in edges_list:
            x = edge[0]
            y = edge[1]
            source.append(x)
            target.append(y)

        return source, target,mark_list

    def get_all_freq(self):
        walk = []
        for edge in self.G.edges():
            walk.append(edge[0])
            walk.append(edge[1])
        return walk

        return walk
    def get_all_freqX(self,source,target):
        walk=[]
        for x,y in zip(source,target):
            walk.append(x)
            walk.append(y)
        return walk


