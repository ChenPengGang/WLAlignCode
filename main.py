from network import *
import torch.optim as optim
from trainer.MyLoss import *
from trainer.MyOptimizer import adam
#from trainer.MyTrain4Struct import *
import torch.nn as nn
from utils.utils4Mapping import *
#from trainer.Trainer import *
from trainer.Trainer_T import *
from copy import copy
from utils.utils4ReadData import *
import os
import argparse
# from testEmb import testEmb

def seed_torch(seed=2021):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

#Parameter setting
EMBEDDING_DIM = 128
PRINT_EVERY = 100
EPOCHS = 100
EPOCHS4Struct = 0
BATCH_SIZE = 1000
BATCHS=100
N_SAMPLES = 20
LR=0.005
LR4Struct=0.005
device = 'cuda:1'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='pe', help='dataset name.')
parser.add_argument('--ratio', type=int, default=5, help='training ratio')
args = parser.parse_args()

data_file=args.dataset
ratio=args.ratio
# File parameters
ouput_filename_networkx, \
ouput_filename_networky, \
networkx_file, \
networky_file, \
anchor_file, \
test_file = get_data(ratio,data_file)



#Read network and related structure data
nx_G,G_f,G_t,anchor_list= read_graph(networkx_file,networky_file,anchor_file)
graph_B=nx.DiGraph()

G_anchor=get_graph_anchor(nx_G)

network = networkC(G_anchor,nx_G, True, 1, 1,anchor_list)
network_all = networkC(nx_G,G_anchor, True, 1, 1,anchor_list)

init_dim=len(network.vocab2int)
print(network.num_nodes())
print('length of all nodes:',len(list(network.G.nodes())))
print('length of all nodes:',len(list(network.G_all.nodes())))
length_nodes=len(list(network.G_all.nodes()))
test_anchor_list=get_test_anchor(test_file)

print(device)
all_candate=[]
un_candate_pair_list=[]
neg_result=[]
embedding_dict_list=[]
n=100
is_change=False
all_mark=0
all_mis=0
all_closure=dict()

#Initialize training function
#trainerI = Trainer(EMBEDDING_DIM, 0.0001, EPOCHS4Struct, BATCHS, BATCH_SIZE, N_SAMPLES, network_all, device)
trainerT = Trainer_T(EMBEDDING_DIM, LR4Struct, 8000, BATCHS, BATCH_SIZE, N_SAMPLES, network_all, device)

y=1
num_mark=0
acc_list=[]

#Training process
while(True):
    print(len(network.vocab2int))
    #Initialize aggregate function and node label
    agg_model = AggregateLabel(len(network.vocab2int), len(network.vocab2int), device).to(device)
    onehot_label=get_graph_label(network,len(network.vocab2int),device).to(device)
    # if y==1:
    #     pre_layer_label=torch.zeros(len(network_all.vocab2int),EMBEDDING_DIM).to(device)
    # else:
    #     pre_layer_label=embedding_I.detach()
    # print(pre_layer_label.shape,'============================================')

    # label aggregate=================================================================
    layers_label = agg_model(onehot_label.weight.data, network.adj_real.to(device), 1)

    #Judge the colored node pair by similarity=================================================================
    candate_pair,un_candate_pair,result_old_list=get_candate_pair(layers_label[0],network)
    candate_pair_self, un_candate_pair_self, result_old_list_self = get_candate_pair_self(layers_label[0], network)

    y+=1

    #Judge whether convergence
    if not network.is_convergence():
        network_tmp = copy(network)
        network_tmp.vocab2int = copy(network.vocab2int)
        network_tmp.int2vocab = copy(network.int2vocab)
        candate_pair = list(set(candate_pair))
        all_candate.extend(candate_pair)
        all_candate = list(set(all_candate))

        # embedding_I, network_all = trainerI.train_anchor(network_tmp, all_candate, layers_label[0],pre_layer_label, nx_G,network.mark_pair,0)
        #
        embedding_T, network_all = trainerT.train_anchor(network_all, all_candate, un_candate_pair_list,
                                                         layers_label[0],network.mark_pair,ouput_filename_networkx,ouput_filename_networky,50)
        # If there is no convergence, remap the label
        network.remark_node(candate_pair)
        network.reset_anchor(candate_pair, get_graph_anchorBy_mark, len(candate_pair))

        print(network.is_mark_finished())
        network.reset_edges(candate_pair, get_graph_anchorBy_mark, candate_pair_self)
        num_mark = network.num_mark()
        continue
    else:
        num_mark = network.num_mark()

        candate_pair = list(set(candate_pair))
        all_candate = remark(all_candate, network)
        # all_candate = list(set(all_candate))

        all_candate.extend(candate_pair)

        embedding_T, network_all = trainerT.train_anchor(network_all, all_candate, un_candate_pair_list, layers_label[0],network.mark_pair,ouput_filename_networkx,ouput_filename_networky)
        print(embedding_T.shape)
        writeFile(embedding_T, network_all, ouput_filename_networkx + ".number_T", "_foursquare")
        writeFile(embedding_T, network_all, ouput_filename_networky + ".number_T", "_twitter")

        break
print('finished!')






