""" A one line summary of the module or program
Copyright：©2011-2022 北京华如科技股份有限公司
This module provide configure file management service in i18n environment.
Authors: zhanghantang
DateTime:  2022/11/4 14:24
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.hyper_parameters import Model_Parameters as MP

class Core(nn.Module):

    def __init__(self, batch_size=MP.batch_size, sequence_length=MP.sequence_length,
                 hidden_dim=MP.hidden_size, n_layers=MP.lstm_layers,
                 embedding_dim=MP.lstm_input, drop_prob=0.0, device=torch.device("cpu")):
        super(Core, self).__init__()


        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers,
                            dropout=drop_prob, batch_first=True)

        self.to(device)
        self.batch_size = batch_size
        self.sequence_length = sequence_length



    def forward(self, embedded_scalar, embedded_entity,
                sequence_length=None, hidden_state=None):
        batch_size = embedded_entity.shape[0]
        sequence_length = sequence_length if sequence_length is not None else self.sequence_length


        input_tensor = torch.cat([embedded_entity, embedded_scalar], dim=-1)
        embedding_size = input_tensor.shape[-1]
        input_tensor = input_tensor.reshape(batch_size, sequence_length, embedding_size)

        if hidden_state is None:
            hidden_state = self.init_hidden_state(batch_size=batch_size)

        lstm_output, hidden_state = self.forward_lstm(input_tensor, hidden_state)
        lstm_output = lstm_output.reshape(batch_size * sequence_length, self.hidden_dim)

        return lstm_output, hidden_state

    def init_hidden_state(self, batch_size=1):
        device = next(self.parameters()).device
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
                  torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device))

        return hidden

    def forward_lstm(self, x, hidden):
        # note: No projection is used.
        # note: The outputs of the LSTM are the outputs of this module.
        lstm_out, hidden = self.lstm(x, hidden)

        return lstm_out, hidden

if __name__ == '__main__':
    # todo
    embedding_size = 384
    input_tensor = torch.zeros((2, 384))
    sequence_length = 1
    input_tensor = input_tensor.reshape(2, sequence_length, embedding_size)