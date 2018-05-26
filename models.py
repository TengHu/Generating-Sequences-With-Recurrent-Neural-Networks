import torch
import torch.nn as nn


class DLSTM3(nn.Module):
    '''
    Three layer RNN with lstm cells
    
    Input: (batch, seq, feature)
    Hidden: (num_layers * direction, batch, feature)
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
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True)
        self.h3 = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True)
        self.h1Toi = nn.Linear(hidden_size + feature_size, feature_size)
        self.h2Toi = nn.Linear(hidden_size + feature_size, feature_size)

        self.h2o = nn.Linear(hidden_size * 3, feature_size)

    def forward(self, input, hiddens):
        [h1, h2, h3] = hiddens

        h1_outputs, _ = self.h1(input, h1)

        h2_input = self.h1Toi(torch.cat((input, h1_outputs), 2))
        h2_outputs, _ = self.h2(h2_input, h2)

        h3_input = self.h2Toi(torch.cat((input, h2_outputs), 2))
        h3_outputs, _ = self.h3(h3_input, h3)

        outputs = self.h2o(torch.cat((h1_outputs, h2_outputs, h3_outputs), 2))
        return F.log_softmax(outputs, 2)

    def initHidden(self, layer=1, batch_size=50, use_gpu=True):
        hidden_layers = []
        for i in range(0, layer):
            h = torch.randn(1, self.batch_size, self.hidden_size).pin_memory()
            c = torch.randn(1, self.batch_size, self.hidden_size).pin_memory()
            if use_gpu:
                h = h.to(device)
                c = c.to(device)
            hidden_layers.append((h, c))
        return hidden_layers
