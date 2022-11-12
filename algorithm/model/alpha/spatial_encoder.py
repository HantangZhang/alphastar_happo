""" A one line summary of the module or program
Copyright：©2011-2022 北京华如科技股份有限公司
This module provide configure file management service in i18n environment.
Authors: zhanghantang
DateTime:  2022/11/4 15:05
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock1D(nn.Module):

    def __init__(self, inplanes, planes, seq_len,
                 stride=1, downsample=None, norm_type='prev'):
        super(ResBlock1D, self).__init__()

        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)

        self.ln1 = nn.LayerNorm([planes, seq_len])
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.ln2 = nn.LayerNorm([planes, seq_len])
        self.normtype = norm_type

    def forward(self, x):
        if self.normtype == 'prev':
            residual = x
            x = F.relu(self.ln1(x))
            x = self.conv1(x)
            x = F.relu(self.ln2(x))
            x = self.conv2(x)
            x = x + residual
            del residual
            return x
        elif self.normtype == 'post':
            residual = x
            x = F.relu(self.conv1(x))
            x = self.ln1(x)
            x = F.relu(self.conv2(x))
            x = self.ln2(x)
            x = x + residual
            del residual
            return x


