import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SingleLSTM(nn.Module):
    '''
    Input: (batch, seq, ids)
    Hidden: (num_layers * direction, batch, hidden_feature)
    Output: (batch, seq, feature)
    '''
    def __init__(self, feature_size, hidden_size):
        super(SingleLSTM, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        '''Weights'''
        self.h1 = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True)
        self.h2o = nn.Linear(hidden_size, feature_size)

    def forward(self, inputs, hiddens):
        h, h_and_c = self.h1(inputs, hiddens)
        outputs = self.h2o(h)
        return F.log_softmax(outputs, 2), h_and_c

    def initHidden(self, batch_size=50):
        h = torch.randn(1, batch_size, self.hidden_size).to(device)
        c = torch.randn(1, batch_size, self.hidden_size).to(device)
        return [h, c]


class DLSTM3(nn.Module):
    '''
    Three layer RNN with lstm cells
    Input: (batch, seq, ids)
    Hidden: (num_layers * direction, batch, hidden_feature)
    Output: (batch, seq, feature)
    '''
    def __init__(self, feature_size, hidden_size):
        super(DLSTM3, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        '''Weights'''
        self.h1 = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True)
        self.h2 = nn.LSTM(
            input_size=feature_size + hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True)
        self.h3 = nn.LSTM(
            input_size=feature_size + hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True)

        self.h2o = nn.Linear(hidden_size * 3, feature_size)

    def forward(self, inputs, hiddens):
        [h1, h2, h3] = hiddens
        h1_outputs, h1 = self.h1(inputs, h1)
        h2_outputs, h2 = self.h2(torch.cat((inputs, h1_outputs), 2), h2)
        h3_outputs, h3 = self.h3(torch.cat((inputs, h2_outputs), 2), h3)
        outputs = self.h2o(torch.cat((h1_outputs, h2_outputs, h3_outputs), 2))
        return F.log_softmax(outputs, 2), [h1, h2, h3]

    def initHidden(self, layer=3, batch_size=50):
        hidden_layers = []
        for i in range(0, layer):
            h = torch.randn(1, batch_size, self.hidden_size).to(device)
            c = torch.randn(1, batch_size, self.hidden_size).to(device)
            hidden_layers.append((h, c))
        return hidden_layers
