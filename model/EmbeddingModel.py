import torch
import torch.nn as nn
from model.AggregateModel import *

class EmbeddingModel(nn.Module):
    def __init__(self, n_vocab, n_embed):
        super().__init__()
        self.n_vocab = n_vocab
        self.n_embed = n_embed

        self.self_embed = nn.Embedding(n_vocab,n_embed)
        self.in_embed = nn.Embedding(n_vocab, n_embed)
        self.out_embed = nn.Embedding(n_vocab, n_embed)

        self.self_embed.weight.data.uniform_(-1,1)
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)

    def forward_input(self, input_words):
        input_vectors = self.in_embed(input_words)
        return input_vectors

    def forward_output(self, output_words):
        output_vectors = self.out_embed(output_words)
        return output_vectors
    def forward_self(self,words):
        self_vectors=self.self_embed(words)
        return self_vectors

    def forward_noise(self, size, N_SAMPLES,device,noise_dist):
        noise_dist = noise_dist

        noise_words = torch.multinomial(noise_dist,
                                        size * N_SAMPLES,
                                        replacement=True)
        noise_vectors_self = self.self_embed(noise_words).view(size, N_SAMPLES, self.n_embed)
        noise_vectors_in = self.in_embed(noise_words).view(size, N_SAMPLES, self.n_embed)
        noise_vectors_out = self.out_embed(noise_words).view(size, N_SAMPLES, self.n_embed)
        return noise_vectors_self, noise_vectors_in, noise_vectors_out

    def forward_noise2(self, size, N_SAMPLES,device,noise_dist):
        noise_dist = noise_dist.to(device)

        noise_words = torch.multinomial(noise_dist,
                                        size * N_SAMPLES,
                                        replacement=True)
        noise_vectors_self = self.self_embed(noise_words).view(size, N_SAMPLES, self.n_embed)
        noise_vectors_in = self.in_embed(noise_words).view(size, N_SAMPLES, self.n_embed)
        noise_vectors_out = self.out_embed(noise_words).view(size, N_SAMPLES, self.n_embed)
        return noise_vectors_self, noise_vectors_in, noise_vectors_out