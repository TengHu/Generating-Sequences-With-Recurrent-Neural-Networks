import os
import torch
import numpy as np
from cache import cached

@cached()
def get_corpus(path=""):
    return Corpus(path)

class Vocabulary(object):
    def __init__(self):
        self.char2idx = {}
        self.idx2char = []
        self.ntokens = 0

    def add_char(self, char):
        if char not in self.char2idx:
            self.idx2char.append(char)
            self.char2idx[char] = len(self.idx2char) - 1
        return self.char2idx[char]

    def __len__(self):
        return len(self.idx2char)


class Corpus(object):
    def __init__(self, path):
        print("Initializing Corpus from file")
        
        self.vocabulary = Vocabulary()
        self.data = self.tokenize(path)

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

    def shuffle(self, train=0.8, valid=0.1, test=0.1):
        '''
        random shuffling into train, valid and test in tensors
        '''
        assert train + valid + test == 1.0

        train_data, remain = self.split(self.data, ratio=train)
        valid_data, test_data = self.split(
            remain, ratio=valid / (valid + test))
        return train_data, valid_data, test_data

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)

        # Construct vocabulary for the corpus
        print("Constructing Vocabulary")

        file = open(path, encoding='utf-8').read()
        ntokens = 0
        print(len(file))
        for char in file:
            self.vocabulary.add_char(char)
            ntokens += 1
        self.vocabulary.ntokens = ntokens

        print("Tokenizing file")
        # Tokenize file content
        ids = torch.LongTensor(ntokens)
        token = 0
        for char in file:
            ids[token] = self.vocabulary.char2idx[char]
            token += 1
            if token % 10000 == 0:
                print("In Progress: {} / {}".format(token, ntokens))
        del file
        print("Done!\n")
        return ids
