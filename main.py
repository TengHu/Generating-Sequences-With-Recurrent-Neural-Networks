from __future__ import unicode_literals, print_function, division
from io import open
import glob
import torch
import pylab
import torch.nn.functional as F
import torchvision.models as models
import random
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import unicodedata
import string
import time
import math
import matplotlib.ticker as ticker
import torch.utils.data as data_utils
import shutil
import pdb
import torch.multiprocessing as mp
import pickle

torch.manual_seed(1)

###############################################################################
# Helper Functions
###############################################################################
def find_files(path):
    return glob.glob(path)


def read_lines(filename):
    lines = open(filename, encoding='utf-8').read()  # remove newline
    return lines


### Data Preprocessing
# return tensor: (1, sequence, n_letter)
def string2Tensor(text):
    tensor = np.zeros((1, len(text), n_letters), dtype=np.float32)
    for li in range(len(text)):
        letter = text[li]
        tensor[0][li][char_to_index[letter]] = 1
        return torch.Tensor(tensor)


def get_batch(text, idx, sequence, batch_size=1):
    inputs = torch.cat(
        [
            string2Tensor(text[i:i + sequence])
            for i in range(idx, min(idx + batch_size,
                                    len(text) - sequence))
        ],
        dim=0)

    targets = torch.LongTensor(
        [[char_to_index[text[j]] for j in range(i + 1, i + sequence + 1)]
         for i in range(idx, min(idx + batch_size,
                                 len(text) - sequence))])
    # list of list of target long number
    return inputs, targets


###############################################################################
# Load Data
###############################################################################
