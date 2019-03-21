import numpy as np
from scipy.spatial import KDTree

import torch
from torch import nn
from torch.nn.init import kaiming_normal_


class CBOW(nn.Module):
    def __init__(self, vocab_size, hidden_size, device):
        super(CBOW, self).__init__()
        
        self.encode_layer = nn.Embedding(num_embeddings=vocab_size+1, 
                                         embedding_dim=hidden_size).to(device)
        self.decode_layer = nn.Linear(hidden_size, vocab_size+1, bias=False).to(device)
        
        self.encode_layer.weight.data = kaiming_normal_(self.encode_layer.weight.data)
        self.decode_layer.weight.data = kaiming_normal_(self.decode_layer.weight.data)
        
    def forward(self, x):
        context = self.encode_layer(x).mean(dim=1)
        activations = self.decode_layer(context)
        return activations
    
    
def get_encode_emb(words, model, batcher, device):
    words_idx = [batcher.word2ind.get(word, 0) for word in words]
    t = torch.LongTensor(words_idx).to(device)
    vectors = model.encode_layer(t).to('cpu').data.numpy()
    return vectors


def get_decode_emb(words, model, batcher, device):
    words_idx = [batcher.word2ind.get(word, 0) for word in words]
    weights = model.decode_layer.weight.data
    vectors = weights[words_idx].to('cpu').data.numpy()
    return vectors


class Index:
    
    def __init__(self, model, vocab, batcher, get_emb, device):
        self.get_emb = get_emb
        self.size = model.encode_layer.weight.size(1)
        self.device = device
        embs = []
        self.idxes = []
        for word in vocab:
            vector = self.get_emb([word], model, batcher, device)
            vector = vector / np.linalg.norm(vector)
            embs.append(vector)
            self.idxes.append(batcher.word2ind[word])
        embs = np.concatenate(embs, axis=0)
        self.t = KDTree(embs)
        
        self.batcher = batcher
        self.model = model
        
    def most_similar(self, pos, neg=None, k=10):
        emb = np.zeros(self.size, dtype=np.float32)
        if pos is not None:
            for w in pos:
                emb += self.get_emb([w], self.model, self.batcher, self.device).flatten()
        if neg is not None:
            for w in neg:
                emb -= self.get_emb([w], self.model, self.batcher, self.device).flatten()
        
        emb = (emb / np.linalg.norm(emb)).flatten()
#         emb = emb.flatten()
        dists, idxes = self.t.query(emb, k)
        return [(self.batcher.ind2word[self.idxes[i]], d) for i, d in zip(idxes, dists)]