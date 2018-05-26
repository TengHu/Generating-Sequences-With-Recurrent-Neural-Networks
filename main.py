from __future__ import unicode_literals, print_function, division
from io import open
import os
import glob
import torch
import pylab
import torch.nn.functional as F
import torchvision.models as models
import models
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
import data
import pdb
import torch.multiprocessing as mp
import argparse

import data

torch.manual_seed(1)

###############################################################################
parser = argparse.ArgumentParser(description='Character Level Language Model')
parser.add_argument(
    '--data', type=str, default='./enwiki', help='location of the data corpus')

parser.add_argument(
    '--import_model',
    type=str,
    default='NONE',
    help='import model if specified otherwise train from random initialization'
)

parser.add_argument(
    '--model', type=str, default='DLSTM3', help='models: DLSTM3')

parser.add_argument(
    '--hidden_size', type=int, default=128, help='# of hidden units')

parser.add_argument(
    '--batch_size', type=int, default=50, help='# of hidden units')

parser.add_argument('--epochs', type=int, default=3, help='# of epochs')

parser.add_argument(
    '--lr', type=float, default=0.001, help='initial learning rate')

parser.add_argument('--clip', type=float, default=0.5, help='gradient clipp')

parser.add_argument(
    '--bptt', type=int, default=50, help='backprop sequence length')

parser.add_argument(
    '--print_every', type=int, default=50, help='print every # iterations')

parser.add_argument(
    '--save_every',
    type=int,
    default=2000,
    help='save model every # iterations')

parser.add_argument(
    '--plot_every',
    type=int,
    default=50,
    help='plot the loss every # iterations')
args = parser.parse_args()


###############################################################################
# Helper Functions
###############################################################################
def find_files(path):
    return glob.glob(path)


def read_lines(filename):
    lines = open(filename, encoding='utf-8').read()  # remove newline
    return lines


'''
Data Preprocessing
return tensor: (1, sequence, n_letter)
'''


def string2Tensor(text):
    tensor = np.zeros((1, len(text), n_letters), dtype=np.float32)
    for li in range(len(text)):
        letter = text[li]
        tensor[0][li][char_to_index[letter]] = 1
        return torch.Tensor(tensor)


###############################################################################
# Load Data
###############################################################################

TEXT_PATH = './enwik8_tail_1000'
corpus = data.Corpus(TEXT_PATH)

###############################################################################
# Build Model
###############################################################################

feature_size = corpus.vocabulary.ntokens
hidden_size = args.hidden_size
model_type = args.model
print(args.import_model)
if args.import_model != 'NONE':
    print("=> loading checkpoint ")
    checkpoint = torch.load(args.import_model)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model_type = args.import_model.split('.')[1]
if model_type == 'DLSTM3':
    model = models.DLSTM3(feature_size, hidden_size)

###############################################################################
# Training code
###############################################################################
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.NLLLoss()


def save_checkpoint(state, filename='checkpoint.pth'):
    '''
    One dimensional array
    '''
    torch.save(state, filename)
    print("{} saved ! \n".format(filename))


def batchify(data):
    '''
    Tensorify the data make it ready for getting batches
    (batch_size, nbatch)
    '''
    nbatch = data.shape[0] // args.batch_size
    data = data[:nbatch * args.batch_size]
    return data.view(batch_size, -1)


def get_batch(data, idx):
    inputs = data[:, idx:idx + args.bptt]
    targets = data[:, (idx + 1):(idx + 1) + args.bptt]
    return inputs, targets


def get_loss(outputs, targets):
    loss = 0
    for i in range(0, args.bptt):
        loss += criterion(outputs[i], target_tensor[i])
    return loss


def train(data):
    length = data.shape[1]  # DLSTM3
    losses = []
    hiddens = model.initHidden(
        layer=3, batch_size=args.batch_size, use_gpu=True)

    for batch_idx, idx in enumerate(range(0, length - args.bptt, args.bptt)):
        inputs, targets = get_batch(data, idx)
        optimizer.zero_grad()
        outputs = model(inputs, hiddens)

        loss = get_loss(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), max_clip_norm)
        optimizer.step()

        if iter % args.print_every == 0 and iter > 0:
            print(
                "Epoch : {}, Iteration {} / {}, Loss per {} :  {}, Takes {} Seconds".
                format(epoch, iter, iters, print_every, loss,
                       time.time() - start))

        if iter % args.plot_every == 0 and iter > 0:
            losses.append(total_loss / plot_every)
            total_loss = 0

        if iter % args.save_every == 0 and iter > 0:
            save_checkpoint({
                'epoch': epoch,
                'iter': iter,
                'losses': losses,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, "checkpoint_{}_epoch_{}_iteration_{}.{}.pth".format(
                int(time.time()), epoch, iter, model_type))
    return losses


'''
Training Loop
Can interrupt with Ctrl + C
'''
start = time.time()
all_losses
try:
    for epoch in range(1, args.epochs + 1):
        '''All in tenors '''
        train_data, valid_data, test_data = corpus.shuffle()
        train_data = batchify(train_data)
        valid_data = batchify(valid_data)
        test_data = batchify(test_data)
        loss = train(train_data)
        all_losses.append(loss)

except KeyboardInterrupt:
    print('#' * 90)
    print('Exiting from training early')

print('#' * 90)
print("Training finished ! Takes {} seconds ".format(time.time() - start))
