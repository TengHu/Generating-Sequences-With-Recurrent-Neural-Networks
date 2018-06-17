import os
import torch
import numpy as np
from cache import cached
from bisect import bisect_left
from bisect import bisect_right
import pdb

@cached()
def get_corpus(path="", special_tokens="", corpus=None):
    if corpus is not None:
        return Corpus(vocabulary=corpus.vocabulary, path=path, special_tokens=special_tokens)
    else:
        return Corpus(path=path, special_tokens=special_tokens)

class Vocabulary(object):
    def __init__(self):
        self.char2idx = {}
        self.idx2char = []

    def add_char(self, char):
        if char not in self.char2idx:
            self.idx2char.append(char)
            self.char2idx[char] = len(self.idx2char) - 1
        return self.char2idx[char]

    def __len__(self):
        return len(self.idx2char)


class Corpus(object):
    def __init__(self, vocabulary=None, path="", special_tokens=""):
        print("Initializing Corpus from file")
        self.name = path.split('/')[-1]
        self.vocabulary = vocabulary
        self.data = self.tokenize(path, special_tokens=special_tokens)            

    def split(self, corpus, ratio=0.5):
        '''
        random split into 2 tensors
        '''
        sz = corpus.shape[0]
        start = np.random.randint(sz)
        end = int(start + ratio * sz) % sz

        if end > start:
            corpus1 = corpus[start:end]
            corpus2 = torch.cat((corpus[0:start], corpus[end:sz]), dim=0)
            return corpus1, corpus2
        else:
            corpus1 = torch.cat((corpus[0:end], corpus[start:sz]), dim=0)
            corpus2 = corpus[end:start]
            return corpus1, corpus2

    def position_encode(self, idx, token_idxs):
        '''
        idx is index in data 
        i is index in the token idx list
        return 0 < ratio  < 1
        '''
        i = bisect_left(token_idxs, idx)
        if token_idxs[i] == idx:
            return 0
        else:
            l1 = token_idxs[i - 1]
            l2 = token_idxs[i]
            return (idx - l1) / (l2 - l1)

    def shuffle(self, train=0.8):
        '''
        random shuffling into train, valid in tensors
        '''

        train_data, remain = self.split(self.data, ratio=train)
        return train_data, remain

    def tokenize(self, path, special_tokens=""):
        """
        Tokenizes a text file.
        Return (number of chars, ids)
        """
        assert os.path.exists(path)

        file = open(path, encoding='utf-8').read()
        ntokens = len(file)
        print ("File has {} tokens".format(ntokens))
        
        if self.vocabulary == None:
            self.vocabulary = Vocabulary()
            # Construct vocabulary for the corpus
            print("Constructing Vocabulary")
            for char in file:
                self.vocabulary.add_char(char)

        print("Generating Special Tokens")
        special_tokens_idxes = []
        if special_tokens != "":
            # Position Tokens
            for st in special_tokens:
                buf = [i for i, c in enumerate(file) if c == st]
                special_tokens_idxes.append([-1] + buf + [ntokens])

        print("Tokenizing file")                

        # Tokenize file content
        ids = torch.FloatTensor(ntokens, 1 + len(special_tokens_idxes))
        token = 0
        for idx, char in enumerate(file):
            ids[token][0] = self.vocabulary.char2idx[char]
            
            # Positional Encoding
            for i, _ in enumerate(special_tokens):
                ids[token][i+1] = self.position_encode(idx, special_tokens_idxes[i])       
            token += 1
            if token % 1000000 == 0:
                print("In Progress: {} / {}".format(token, ntokens))
        del file
        print("Done!\n")
        return ids
