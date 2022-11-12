""" A one line summary of the module or program
Copyright：©2011-2022 北京华如科技股份有限公司
This module provide configure file management service in i18n environment.
Authors: zhanghantang
DateTime:  2022/11/4 14:22
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.hyper_parameters import State_Space_Parameters as SSP
from configs.hyper_parameters import Model_Parameters as MP
from algorithm.model.alpha.battle_transformer import Transformer
from utils.util import check

class EntityEncoder(nn.Module):

    def __init__(self, dropout=0.0, max_entity_size=SSP.eneity_num,
                 original_256=MP.original_256, original_128=MP.original_128,
                 original_64=MP.original_64, entity_embedding_size=MP.entity_embedding_size,
                 device=torch.device("cpu")):
        super().__init__()

        self.max_entity_size = max_entity_size
        self.dropout = nn.Dropout(dropout)
        self.embedd = nn.Linear(MP.embedding_size, original_128)
        self.transformer = Transformer(d_model=original_128, d_inner=original_256,
                                       n_layers=3, n_head=2, d_k=original_64,
                                       d_v=original_64,
                                       dropout=0.)  # make dropout=0 to make training and testing consistent
        self.conv1 = nn.Conv1d(original_128, entity_embedding_size, kernel_size=1, stride=1,
                               padding=0, bias=True)
        self.fc1 = nn.Linear(original_128, original_128)

        self.to(device)
    def forward(self, x):
        # x  [batch_size, entities_size, feature_size]

        batch_size = x.shape[0]
        entities_size = x.shape[1]

        # 统计每个batch，里面有几个有数据地实体，如果实体有数据地话tmp_y就是True，最后就是输出有几个true
        tmp_x = torch.mean(x, dim=2, keepdim=False)
        tmp_y = (tmp_x != 0)
        # entity_num, [batch_size entity_num]
        entity_num = torch.sum(tmp_y, dim=1, keepdim=False)

        # 实体数量可能超过我们限定地max_entities，所以要用mini来限制一下
        entity_num_numpy = np.minimum(self.max_entity_size - 2, entity_num.cpu().numpy())
        entity_num = torch.tensor(entity_num_numpy, dtype=entity_num.dtype, device=entity_num.device)

        # generate the mask for transformer
        mask = torch.arange(0, self.max_entity_size).float()
        mask = mask.repeat(batch_size, 1)

        device = next(self.parameters()).device
        mask = mask.to(device)
        mask = mask < entity_num.unsqueeze(dim=1)

        x = self.embedd(x)
        mask_seq_len = mask.shape[-1]
        tran_mask = mask.unsqueeze(1)

        # tran_mask: [batch_seq_size x max_entities x max_entities]
        tran_mask = tran_mask.repeat(1, mask_seq_len, 1)

        out = self.transformer(x, mask=tran_mask)

        entity_embeddings = F.relu(self.conv1(F.relu(out).transpose(1, 2))).transpose(1, 2)

        # 合并所有智能体的数据就是通过相加然后除以所有智能体的数量，也就是求平均
        masked_out = out * mask.unsqueeze(dim=2)
        z = masked_out.sum(dim=1, keepdim=False)
        z = z / entity_num.unsqueeze(dim=1)

        embedded_entity = F.relu(self.fc1(z))

        #entity_embeddings [batch_size, eneity_num, 128]
        # embedded_entity [batch_size, 128]
        return entity_embeddings, embedded_entity







if __name__ == '__main__':
   x = torch.zeros((64, 20, 163))
   # x[3][2][22] = 1
   print(x.shape)
   model = EntityEncoder()
   a, b= model.forward(x)
   print(a.shape)
   print(b.shape)

