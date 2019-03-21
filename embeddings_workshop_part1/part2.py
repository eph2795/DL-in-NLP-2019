from collections import Counter
from itertools import chain 

import numpy as np

from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords 
        

def process_corpus(fname):
    corpus = []
    with open(fname, 'r') as f:
        for line in f:
            for sent in sent_tokenize(line):
                corpus.append([word.strip() for word in word_tokenize(sent.lower())])
    return corpus


def build_vocab(corpus, min_freq=3):
    stop_words = set(stopwords.words('english')) 
    cnt = Counter(chain(*corpus))
    vocab = [word for word in cnt.keys() if (cnt[word] >= min_freq) and (word not in stop_words)]
    return vocab


class CBOWBatcher:
    
    def __init__(self, batch_size, window_size, shuffle):
        self.batch_size = batch_size
        self.window_size = window_size
        self.shuffle = shuffle
        
    def init(self, corpus, vocab):
        self.word2ind = {w: i+1 for i, w in enumerate(vocab)}
        self.word2ind['$UNK'] = 0
        self.ind2word = {i: w for w, i in self.word2ind.items()}
        
        x_all = []
        y_all = []
        
        for sent in corpus:
            encoded = np.array([self.word2ind.get(w, 0) 
                                for w in ['$UNK'] * self.window_size 
                                         + sent
                                         + ['$UNK'] * self.window_size])
            total = len(encoded)
            x_sent = []
            for neg_offset in range(-self.window_size, 0):
                left = self.window_size + neg_offset
                right = -self.window_size + neg_offset 
                x_sent.append(encoded[left:right].reshape(-1, 1))   

            for pos_offset in range(1, self.window_size + 1):
                left = self.window_size + pos_offset
                right = total - self.window_size + pos_offset 
                x_sent.append(encoded[left : right].reshape(-1, 1))

            x_all.append(np.concatenate(x_sent, axis=1))
            y_all.append(encoded[self.window_size : -self.window_size])
        
        self.x = np.concatenate(x_all, axis=0)
        self.y = np.concatenate(y_all, axis=0)
        self.len = self.x.shape[0]
        
    def __iter__(self):
        order = np.arange(self.len)
        if self.shuffle:
            np.random.shuffle(order)
        batches_num = (self.len  + self.batch_size - 1) // self.batch_size
        for i in range(batches_num):
            idxes = order[i*self.batch_size : (i+1)*self.batch_size]
            yield self.x[idxes], self.y[idxes]
    
    def __len__(self):
        return self.len
    
    def transform(self, sents):
        idxes = []
        for sent in sents:
            idxes.append([self.word2ind[w] for w in sent])
        return idxes        
        
    def inverse_transform(self, idxes):
        sents = []
        for row in idxes:
            sents.append([self.ind2word[i] for i in row])
        return sents
