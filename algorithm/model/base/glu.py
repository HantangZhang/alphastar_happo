""" A one line summary of the module or program
Copyright：©2011-2022 北京华如科技股份有限公司
This module provide configure file management service in i18n environment.
Authors: zhanghantang
DateTime:  2022/11/4 16:33
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.hyper_parameters import Model_Parameters as MP

class GLU(nn.Module):
    '''
    Gating Linear Unit.
    Inputs: input, context, output_size
    '''

    def __init__(self, input_size=384, context_size=MP.scalar_context_dim,
                 output_size=256):
        super().__init__()
        self.fc_1 = nn.Linear(context_size, input_size)
        self.fc_2 = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, context):
        # context shape: [batch_size x context_size]
        gate = self.sigmoid(self.fc_1(context))
        # gate shape: [batch_size x input_size]

        # The line is the same as below: gated_input = torch.mul(gate, x)
        # x shape: [batch_size x input_size]
        gated_input = gate * x

        # gated_input shape: [batch_size x input_size]
        output = self.fc_2(gated_input)

        del context, x, gate, gated_input

        return output