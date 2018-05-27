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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###############################################################################
parser = argparse.ArgumentParser(description='Character Level Language Model')
parser.add_argument(
    '--data', type=str, default='./text/enwik8', help='location of the data corpus')

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
# Load Data
###############################################################################

corpus = data.get_corpus(path=args.data)
###############################################################################
# Build Model
###############################################################################

feature_size = corpus.vocabulary.ntokens
hidden_size = args.hidden_size
model_type = args.model
if args.import_model != 'NONE':
    print("=> loading checkpoint ")
    checkpoint = torch.load(args.import_model)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model_type = args.import_model.split('.')[1]
if model_type == 'DLSTM3':
    model = models.DLSTM3(feature_size, hidden_size)

model = model.to(device)

###############################################################################
# Helper Functions
###############################################################################


def save_checkpoint(state, filename='checkpoint.pth'):
    '''
    One dimensional array
    '''
    torch.save(state, filename)
    print("{} saved ! \n".format(filename))


def batchify(data):
    '''
    data make it ready for getting batches
    Output: (batch_size, nbatch, features)
    '''
    nbatch = data.shape[0] // args.batch_size
    data = data[:nbatch * args.batch_size]
    return data.view(args.batch_size, -1, corpus.vocabulary.ntokens)


def OneHotEncoding(idxs):
    '''
    Output: (ntokens, features)
    '''
    length = idxs.shape[0]
    pdb.set_trace()
    tensor = np.zeros((length, corpus.vocabulary.ntokens), dtype=np.float32)
    for i in range(length):
        tensor[i][idxs[i]] = 1
    return torch.Tensor(tensor).pin_memory()


def get_batch(data, idx):
    '''
    Output: 
    (batch_size, sequence, features)
    (batch_size, features)
    '''
    inputs = data[:, idx:(idx + args.bptt), :]
    targets = data[:, (idx + args.bptt), :]
    return inputs, targets.long()


def tensor2idx(tensor):
    '''
    Input: (#batch, feature)
    Output: (#batch)
    '''
    batch_size = tensor.shape[0]
    idx = np.zeros((batch_size), dtype=np.int64)

    for i in range(0, batch_size):
        idx[i] = torch.nonzero(tensor[i])[0].data[0]
    return torch.LongTensor(idx)


###############################################################################
# Training code
###############################################################################
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.NLLLoss()


def get_loss(outputs, targets):
    loss = 0
    #for i in range(0, args.bptt):
    loss += criterion(outputs[:, -1, :], tensor2idx(targets).to(device))
    return loss


def detach(layers):
    '''
    Remove variables' parent node after each sequence, 
    basically no where to propagate gradient 
    '''
    if (type(layers) is list) or (type(layers) is tuple):    
        for l in layers:
            detach(l)
    else:
        layers.detach_()

def train(data):
    length = data.shape[1]  # DLSTM3
    losses = []
    total_loss = 0
    hiddens = model.initHidden(
        layer=3, batch_size=args.batch_size, use_gpu=True)

    for batch_idx, idx in enumerate(range(0, length - args.bptt, args.bptt)):
        
        inputs, targets = get_batch(data, idx)
        detach(hiddens)
        
        optimizer.zero_grad()

        outputs, hiddens = model(inputs, hiddens)
        loss = get_loss(outputs, targets)
        loss.backward()

        #torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % args.print_every == 0 and batch_idx > 0:
            print(
                "Epoch : {}, Iteration {} / {}, Loss every {} iteration :  {}, Takes {} Seconds".
                format(epoch, batch_idx, int((length - args.bptt) / args.bptt),
                       args.print_every, loss.item(),
                       time.time() - start))

        if batch_idx % args.plot_every == 0 and batch_idx > 0:
            losses.append(total_loss / args.plot_every)
            total_loss = 0

        if batch_idx % args.save_every == 0 and batch_idx > 0:
            save_checkpoint({
                'epoch': epoch,
                'iter': batch_idx,
                'losses': losses,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, "checkpoint_{}_epoch_{}_iteration_{}.{}.pth".format(
                int(time.time()), epoch, iter, model_type))
        del loss, outputs

    return losses


'''
Training Loop
Can interrupt with Ctrl + C
'''
start = time.time()
all_losses = []
try:
    print("Start Training\n")
    for epoch in range(1, args.epochs + 1):
        '''All in tenors '''
        train_data, valid_data, test_data = corpus.shuffle()

        
        train_data = batchify(OneHotEncoding(train_data)).to(device)
        loss = train(train_data)
        all_losses.append(loss)

except KeyboardInterrupt:
    print('#' * 90)
    print('Exiting from training early')

print('#' * 90)
print("Training finished ! Takes {} seconds ".format(time.time() - start))
