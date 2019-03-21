import numpy as np

import torch
from torch import nn
from torch.nn.init import kaiming_normal_


class CBOW_one_matrix(nn.Module):
    def __init__(self, vocab_size, hidden_size, device):
        super(CBOW_one_matrix, self).__init__()
        self.encode_layer = nn.Embedding(num_embeddings=vocab_size+1, 
                                         embedding_dim=hidden_size).to(device)
        
        self.encode_layer.weight.data = kaiming_normal_(self.encode_layer.weight.data)
        
        self.all = torch.LongTensor(np.arange(vocab_size + 1)).to(device)

    def forward(self, x):
        context = self.encode_layer(x).mean(dim=1)
        embs = self.encode_layer(self.all)
        activations = torch.matmul(context, embs.transpose(0, 1))
        return activations