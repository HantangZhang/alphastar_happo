""" A one line summary of the module or program
Copyright：©2011-2022 北京华如科技股份有限公司
This module provide configure file management service in i18n environment.
Authors: zhanghantang
DateTime:  2022/11/4 14:27
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.categorical as cate
from configs.hyper_parameters import Model_Parameters as MP
from configs.hyper_parameters import Action_Space_Parameters as AHP
from algorithm.model.alpha.mask_func import generate_location_mask

class LocationHead(nn.Module):

    def __init__(self,is_cnn=MP.is_cnn, autoregressive_embedding_size=MP.autoregressive_embedding_size,
                 original_256=MP.original_256, original_128=MP.original_128, original_64=MP.original_64,
                 original_32=MP.original_32, temperature=AHP.temperature, location_dim=AHP.location_dim,
                 device=torch.device("cpu")):
        super().__init__()

        self.use_improved_one = True
        self.is_cnn = is_cnn
        self.temperature = temperature
        self.autoregressive_embedding_size = autoregressive_embedding_size

        # # cnn
        # mmc = max_cnn_channel
        # self.ds_1 = nn.Conv2d(mmc + 4, mmc, kernel_size=1, stride=1,
        #                       padding=0, bias=True)
        #
        # self.film_blocks_num = 4
        # if not self.use_improved_one:
        #     self.film_net = FiLM(n_resblock=self.film_blocks_num,
        #                          conv_hidden=mmc,
        #                          gate_size=autoregressive_embedding_size)
        # else:
        #     self.film_net_mapskip = FiLMplusMapSkip(n_resblock=self.film_blocks_num,
        #                                             conv_hidden=mmc,
        #                                             gate_size=autoregressive_embedding_size)
        self.fc_1 = nn.Linear(autoregressive_embedding_size, original_256)
        self.fc_2 = nn.Linear(original_256, original_128)
        self.fc_3 = nn.Linear(original_128, original_64)
        self.fc_4 = nn.Linear(original_64, location_dim)

        self.softmax = nn.Softmax(dim=-1)

        self.to(device)


    def forward(self, autoregressive_embedding, level1_action):
        batch_size = level1_action.shape[0]
        x = self.fc_1(autoregressive_embedding)
        x = self.fc_2(x)
        x = self.fc_3(x)
        y = self.fc_4(x)

        location_mask = generate_location_mask(level1_action).to(level1_action.device)
        y = y.squeeze(-1)

        location_logits = y.masked_fill(~location_mask, -1e9)
        temperature = self.temperature
        location_logits = location_logits / temperature
        location_logits = self.softmax(location_logits)

        dist = cate.Categorical(probs=location_logits)
        location_id = dist.sample()
        location_id = location_id.reshape(batch_size, -1)
        location_probs = dist.log_prob(location_id).view(location_id.size(0), -1).sum(-1).unsqueeze(-1)

        return location_probs, location_id

    def evaluate_actions(self, autoregressive_embedding, level1_action, location_id, active_masks):
        x = self.fc_1(autoregressive_embedding)
        x = self.fc_2(x)
        x = self.fc_3(x)
        y = self.fc_4(x)

        location_mask = generate_location_mask(level1_action).to(level1_action.device)
        y = y.squeeze(-1)

        location_logits = y.masked_fill(~location_mask, -1e9)
        temperature = self.temperature
        location_logits = location_logits / temperature
        location_logits = self.softmax(location_logits)

        dist = cate.Categorical(probs=location_logits)
        location_probs = dist.log_prob(location_id).view(location_id.size(0), -1).sum(-1).unsqueeze(-1)

        if active_masks is not None:
            dist_entropy = (dist.entropy() * active_masks).sum() / active_masks.sum()
        else:
            dist_entropy = dist.entropy().mean()

        return location_probs, dist_entropy

if __name__ == '__main__':
    # batch_size = 10
    # autoregressive_embedding_size = 256
    # reshpe_channel = int(autoregressive_embedding_size / 64)
    # autoregressive_embedding = torch.ones((10, 256))
    # x = autoregressive_embedding.reshape(batch_size, -1, reshpe_channel, reshpe_channel)
    # print(x.shape)

    model = LocationHead()
    autoregressive_embedding = torch.ones((10, 256))
    level1_action = torch.ones((10, 1))
    active_masks = torch.zeros((10, 1))
    location_id = torch.ones((10, 1))
    x, y = model.forward(autoregressive_embedding, level1_action)
    print(x.shape)
    print(y.shape)
    res1, res2 = model.evaluate_actions(autoregressive_embedding, level1_action, location_id, active_masks)
    print(res1)
    print(res2)


