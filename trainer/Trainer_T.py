from utils import *
from model import *
import random
import torch.optim as optim
import torch.nn as nn
from trainer.MyLoss import NegativeSamplingLoss,TripCenterLoss_margin
from model.EmbeddingModel import *
import numpy as np
from trainer.MyOptimizer import adam
from sklearn.cluster import KMeans
from tqdm import tqdm
#from testEmb import testEmb

class Trainer_T:
    def __init__(self,EMBEDDING_DIM,LR,EPOCHS,BATCHS,BATCH_SIZE,N_SAMPLES,network,device,isanchor=False):
        self.Embedding_dim=EMBEDDING_DIM
        self.lr=LR
        self.Epochs=EPOCHS
        self.Batchs=BATCHS
        self.Batch_size=BATCH_SIZE
        self.n_samples=N_SAMPLES
        self.isanchor=isanchor
        self.model=EmbeddingModel(len(network.vocab2int), EMBEDDING_DIM).to(device)

    def train_anchor(self,network,all_candate, all_closure,layer_label,mark_pair,ouput_filename_networkx,ouput_filename_networky,epoches=None):
        device = 'cuda:1'
        # network.reset_graph(all_candate)
        acc_list=[]
        # network.set_closure_mark(all_candate, all_closure)
        # prepare for the cosine loss=====================================================================
        int_left, int_right = network.get_same_label(all_candate, mark_pair)
        mark_tensor = torch.LongTensor([1] * len(int_left)).to(device)
        mark_tensor_noise = torch.LongTensor([0] * len(int_left)).to(device)
        int_left = torch.LongTensor(int_left).to(device)
        int_right = torch.LongTensor(int_right).to(device)

        all_nodes = torch.LongTensor([network.vocab2int[i] for i in mark_pair.keys()]).to(device)
        all_labels = torch.LongTensor([mark_pair[node] for node in mark_pair.keys()]).to(device)
        #================================================================

        G_anchor = get_graph_anchor(network.G)
        network.get_G_anchor(G_anchor)
        node_noise = get_graph_noise(network.G, G_anchor)
        EMBEDDING_DIM, LR, EPOCHS, BATCHS, BATCH_SIZE, N_SAMPLES\
            =self.Embedding_dim,self.lr,self.Epochs,self.Batchs,self.Batch_size,self.n_samples

        if epoches:
            EPOCHS=epoches

        print('network length:', len(network.vocab2int))
        model = self.model
        criterion = NegativeSamplingLoss()
        cos = nn.CosineEmbeddingLoss(margin=0)
        center_criterion = TripCenterLoss_margin(num_classes=len(list(set(list(mark_pair.values())))), feat_dim=EMBEDDING_DIM,
                                                 use_gpu=True, device=device)
        optimizer_center = optim.Adam(center_criterion.parameters(), lr=0.0001, betas=(0.5, 0.999))

        # =======================================================================================
        noise_dist=self.noise_get(network.get_all_freq(),network).to(device)
        noise_dist2 = self.noise_get(node_noise,network).to(device)
        if len(noise_dist2)==0:
            noise_dist2=noise_dist
        # =======================================================================================

        steps = 0
        sqrs = []
        vs = []
        for param in model.parameters():
            sqrs.append(torch.zeros_like(param.data))
            vs.append(torch.zeros_like(param.data))
        for e in range(EPOCHS):

            i = 0
            if self.isanchor:
                sourcex, targetx, mark_list = network.get_training_set_by_anchor(BATCHS)
            else:
                sourcex, targetx, mark_list = network.get_training_set(BATCHS)
            for source, target in get_batch(sourcex, targetx, BATCH_SIZE):
                steps += 1
                int_source = [network.vocab2int[w] for w in source]
                int_target = [network.vocab2int[w] for w in target]
                inputs, targets = torch.LongTensor(int_source).to(device), torch.LongTensor(int_target).to(device)

                target_input_vectors = model.forward_input(targets)
                source_output_vectors = model.forward_output(inputs)
                self_in_vectors = model.forward_self(inputs)
                self_out_vectors = model.forward_self(targets)

                self_left=model.forward_self(int_left)
                self_right=model.forward_self(int_right)


                self_all_nodes=model.forward_self(all_nodes)

                size, _ = target_input_vectors.shape
                noise_vectors_self, noise_vectors_input, noise_vectors_output = model.forward_noise(size, N_SAMPLES,
                                                                                                    device, noise_dist)

                noise_vectors_self2, noise_vectors_input2, noise_vectors_output2 = model.forward_noise2(size, N_SAMPLES,
                                                                                                        device,
                                                                                                        noise_dist2)

                loss = criterion(self_in_vectors, self_out_vectors, target_input_vectors, source_output_vectors,
                                 noise_vectors_self, noise_vectors_input, noise_vectors_output,
                                 noise_vectors_self2, noise_vectors_input2, noise_vectors_output2)
                loss += cos(self_left, self_right, mark_tensor)

                model.zero_grad()
                loss.backward()
                adam(model.parameters(), vs, sqrs, LR, steps)

                i += 1

            progress(e / EPOCHS * 100, loss)
        print(acc_list)
        return model.self_embed.weight.data,network


    def noise_get(self,noise_list,network):
        all_walks = noise_list
        int_all_words = [network.vocab2int[w] for w in all_walks]

        int_word_counts = Counter(int_all_words)
        total_count = len(int_all_words)
        word_freqs = {w: c / (total_count) for w, c in int_word_counts.items()}

        word_freqs = np.array(list(word_freqs.values()))
        unigram_dist = word_freqs / word_freqs.sum()
        noise_dist = torch.from_numpy(unigram_dist ** (0.75) / np.sum(unigram_dist ** (0.75)))
        return noise_dist