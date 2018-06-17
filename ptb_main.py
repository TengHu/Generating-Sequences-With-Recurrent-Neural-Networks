from __future__ import unicode_literals, print_function, division
from io import open
import os
import glob
import torch
import torch.nn.functional as F
import torchvision.models as models
import models
import random
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import string
import time
import math
import torch.utils.data as data_utils
import shutil
import data
import pdb
import argparse
from utils import one_hot
import data

torch.manual_seed(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

###############################################################################
parser = argparse.ArgumentParser(description='Character Level Language Model')

parser.add_argument(
    '--num_workers',
    type=int,
    default=2,
    help='number of worker to load the data')


parser.add_argument(
    '--train',
    type=str,
    default='data/ptb.char.train.strip.txt',
    help='location of the data corpus')

parser.add_argument(
    '--test',
    type=str,
    default='data/ptb.char.test.strip.txt',
    help='location of the data corpus')

parser.add_argument(
    '--valid',
    type=str,
    default='data/ptb.char.valid.strip.txt',
    help='location of the data corpus')

parser.add_argument(
    '--import_model',
    type=str,
    default='NONE',
    help='import model if specified otherwise train from random initialization'
)

parser.add_argument(
    '--model', type=str, default='DLSTM3', help='models: DLSTM3')

parser.add_argument(
    '--position_codes', type=str, default='', help='number of position features')

parser.add_argument(
    '--position_feature_size', type=int, default=100, help='position feature size')

parser.add_argument(
    '--hidden_size', type=int, default=128, help='# of hidden units')

parser.add_argument(
    '--batch_size', type=int, default=50, help='# of hidden units')

parser.add_argument('--epochs', type=int, default=3, help='# of epochs')

parser.add_argument(
    '--lr', type=float, default=0.001, help='initial learning rate')

parser.add_argument('--clip', type=float, default=1, help='gradient clipp')

parser.add_argument(
    '--bptt', type=int, default=50, help='backprop sequence length')

parser.add_argument(
    '--print_every', type=int, default=50, help='print every # iterations')

parser.add_argument(
    '--save_every',
    type=int,
    default=500,
    help='save model every # iterations')

parser.add_argument(
    '--plot_every',
    type=int,
    default=50,
    help='plot the loss every # iterations')

parser.add_argument(
    '--sample_every',
    type=int,
    default=300,
    help='print the sampled text every # iterations')

parser.add_argument(
    '--output_file',
    type=str,
    default="output",
    help='sample characters and save to output file')

parser.add_argument(
    '--max_sample_length',
    type=int,
    default=500,
    help='max sampled characters')

args = parser.parse_args()

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
    Input: (number of characters, ids)
    Output: (batch * batch_size, -1)
    '''
    nbatch = data.shape[0] // args.batch_size
    data = data[:nbatch * args.batch_size, :]
    return data.to(device)

def get_batch(data, idx):
    '''
    Input: (number_of_chars, ids)
    Output: 
    (batch_size, sequence, ids)
    (batch_size)
    '''
    inputs = data[:, idx:(idx + args.bptt), :]
    targets = data[:, (idx + args.bptt)]

    ## For target, we only need the first char element, discard position codes
    return inputs.to(device), targets[:,0].long().to(device)

def sequentialize(data):
    '''
    Input: (number_of_chars, ids)
    Output:
    (batch, sequence, ids)
    (batch)
    batch size = 1
    '''
    ids_size = data.shape[1]
    nsequence = int(data.shape[0] / args.bptt)
    data = data[0:nsequence*args.bptt, :].view(-1, args.bptt, ids_size)
    targets = data[:, -1, 0] # only want the character id as target
    return data, targets

def tensor2idx(tensor):
    '''
    Input: (#batch, feature)
    Output: (#batch)
    '''
    batch_size = tensor.shape[0]
    idx = np.zeros((batch_size), dtype=np.int64)

    for i in range(0, batch_size):
        value, indice = tensor[i].max(0) # dimension 0
        idx[i] = indice
    return torch.LongTensor(idx)


def preprocess(data):
    inputs, targets = sequentialize(batchify(data))
    inputs = one_hot(inputs, feature_size)
    return torch.utils.data.TensorDataset(inputs, targets)

###############################################################################
# Load Data And PreProcessing
###############################################################################
corpus = data.get_corpus(path=args.train, special_tokens=args.position_codes)

print ("Loading Data, Be aware of old cache may not be compatible with loading data!!")

'''
inputs: (batch, sequence, feature)
targets: (batch)
'''

feature_size = len(corpus.vocabulary) + len(args.position_codes)

train_data = corpus.data
train_dataset = preprocess(train_data)

valid_data = data.get_corpus(corpus=corpus, path=args.valid).data
valid_dataset = preprocess(valid_data)

test_data = data.get_corpus(corpus=corpus, path=args.test).data
test_dataset = preprocess(test_data)



###############################################################################
# Build Model
###############################################################################

#feature_size = len(corpus.vocabulary) + len(args.position_codes) * args.positi#on_feature_size

hidden_size = args.hidden_size
model_type = args.model

if args.import_model != 'NONE':
    print("=> loading checkpoint ")
    if torch.cuda.is_available() is False:
        checkpoint = torch.load(args.import_model, map_location=lambda storage, loc:storage)    
    else:
        checkpoint = torch.load(args.import_model)


    model_type = args.import_model.split('.')[1]
    
    if model_type == 'DLSTM3':
        model = models.DLSTM3(feature_size, hidden_size)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise ValueError("Model type not recognized")
else:
    if model_type == 'DLSTM3':
        model = models.DLSTM3(feature_size, hidden_size)
    else:
        raise ValueError("Model type not recognized")

model = model.to(device)



###############################################################################
# Training code
###############################################################################

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
try:
    optimizer.load_state_dict(checkpoint['optimizer'])
except NameError:
    print("Optimizer initializing")

criterion = nn.NLLLoss().to(device)
warm_up_text = open(args.valid, encoding='utf-8').read()[0:args.bptt]

def get_loss(outputs, targets):
    loss = 0
    loss += criterion(outputs[:, -1, :], targets.long())
    return loss

def sample(text, save_to_file=False, max_sample_length=300, temperature=1.0):
    '''
    Havent figured out how to do sampling
    '''
    try:
        assert len(text) >= args.bptt
    except AssertionError:
        print("Sampling must start with text has more than {} characters \n".
              format(args.bptt))
    output_text = text
    text = text[-args.bptt:] # drop last incomplete sequence

    
    ids = torch.LongTensor(len(text), 1).to(device)
    token = 0
    for c in text:
        ## character id
        ids[token][0] = corpus.vocabulary.char2idx[c]
        token += 1
    
    inputs = ids.unsqueeze(0).to(device)
    hiddens = model.initHidden(layer=3, batch_size=1)

    for i in range(0, max_sample_length):
        outputs, hiddens = model(one_hot(inputs, feature_size), hiddens)    # TODO (niel.hu) temporarily use vocabulary size as feature size 

        # sample character
        char_id = torch.multinomial(
            outputs[0][-1].exp() / temperature, num_samples=1).item()
        output_text += corpus.vocabulary.idx2char[char_id]

        # append text
        text = text[1:] + corpus.vocabulary.idx2char[char_id]

        # rolling inputs
        inputs[0][0:(args.bptt - 1)] = inputs[0][1:]
        inputs[0][args.bptt - 1] = char_id
        del outputs

    if save_to_file:
        with open(args.output_file, 'w') as f:
            f.write(output_text)
            print("Finished sampling and saved it to {}".format(
                args.output_file))
    else:
        print('#' * 90)
        print("\nSampling Text: \n" + output_text + "\n")
        print('#' * 90)


def detach(layers):
    '''
    Remove variables' parent node after each sequence, 
    basically no where to propagate gradient 
    '''
    if (type(layers) is list) or (type(layers) is tuple):
        for l in layers:
            detach(l)
    else:
        layers = layers.detach() # layers.detach_()


def train(dataset):
    losses = []
    total_loss = 0
    hiddens = model.initHidden(layer=3, batch_size=args.batch_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers, drop_last=True)
    for batch_idx, data in enumerate(dataloader, 0):
        inputs, targets = data
        detach(hiddens)
        optimizer.zero_grad()

        outputs, hiddens = model(inputs, hiddens)
        loss = get_loss(outputs, targets)
        loss.backward(retain_graph=True)

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % args.print_every == 0 and batch_idx > 0:
            print(
                "Epoch : {} / {}, Iteration {} / {}, Loss every {} iteration :  {}, Takes {} Seconds".
                format(epoch, args.epochs, batch_idx, int(len(trainset) / args.bptt), args.print_every,
                       loss.item(),
                       time.time() - start))

        if batch_idx % args.plot_every == 0 and batch_idx > 0:
            losses.append(total_loss / args.plot_every)
            total_loss = 0

        if batch_idx % args.sample_every == 0 and batch_idx > 0:
            sample(warm_up_text)

        if batch_idx % args.save_every == 0 and batch_idx > 0:
            save_checkpoint({
                'epoch': epoch,
                'iter': batch_idx,
                'losses': losses,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, "checkpoint_{}_epoch_{}_iteration_{}.{}.pth".format(
                int(time.time()), epoch, batch_idx, model_type))

        del loss, outputs

    return losses


def evaluate(dataset):
    '''
    Undynamic evaluation
    '''
    hiddens = model.initHidden(layer=3, batch_size=args.batch_size)
    total = 0.0
    correct = 0.0
    voc_length = len(corpus.vocabulary)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,shuffle=False, num_workers=args.num_workers, drop_last=True)
    
    bpc = []
    for batch_idx, data in enumerate(dataloader, 0):
        inputs, targets = data
        detach(hiddens)
        outputs, hiddens = model(inputs, hiddens)
        loss = get_loss(outputs, targets)
        total += outputs.shape[0]
        
        ## only need feature vocabulary length 
        correct += (tensor2idx(outputs[:,-1,:voc_length]) == targets.long().cpu()).sum()
        bpc.append(loss)
    
    if total == 0:
        print ("Validation Set too small, reduce batch size")
    else:
        ## Debug Accuracy calculation
        print("Evaluation: Bits-per-character: {}\n, Perplexity: {}\n Accuracy: {} % \n".format(sum(bpc) / len(bpc), "N/A", correct.numpy() / total))
    del loss, outputs
    

'''
Training Loop
Can interrupt with Ctrl + C
'''
start = time.time()
all_losses = []

try:
    print("Start Training\n")
    for epoch in range(1, args.epochs + 1):
        loss = train(train_dataset)
        all_losses += loss
        evaluate(valid_dataset)
except KeyboardInterrupt:
    print('#' * 90)
    print('Exiting from training early')


print ("DONE TRAINING")

print ("Testing")
evaluate(test_dataset)

sample(warm_up_text, save_to_file=True, max_sample_length=args.max_sample_length)

with open("losses", 'w') as f:
    f.write(str(all_losses))


model_name = "{}.{}.pth".format(model_type, time.time())
save_checkpoint({'state_dict': model.state_dict()}, model_name)
print ("Model {} Saved".format(model_name))
    
print('#' * 90)
print("Training finished ! Takes {} seconds ".format(time.time() - start))
