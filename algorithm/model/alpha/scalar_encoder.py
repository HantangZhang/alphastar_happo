import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.hyper_parameters import State_Space_Parameters as SSP
from configs.hyper_parameters import Model_Parameters as MP
from configs.hyper_parameters import Action_Space_Parameters as ASP
'''
agent_statistics
  player_id = 0
  minerals = 1
  vespene = 2
  food_used = 3
  food_cap = 4
  food_army = 5
  food_workers = 6
  idle_worker_count = 7
  army_count = 8
  warp_gate_count = 9
  larva_count = 10


'''

class ScalarEncoder(nn.Module):

    def __init__(self, n_statistics=SSP.scalar_stat_dim, original_64=MP.original_64,
                 level1_action_num=ASP.level1_action_dim, original_32=MP.original_32,
                 target_action_num=ASP.target_dim, location_action_num=ASP.location_dim,
                 original256=MP.original_256, scalar_context=MP.scalar_context_dim,
                 device=torch.device("cpu")):
        super().__init__()

        self.statistics_fc_emb = nn.Linear(n_statistics, original_64)
        self.statistics_fc = nn.Linear(int(n_statistics / 5), original_64)

        self.available_actions_fc_emb = nn.Linear(level1_action_num, original_32)
        self.available_actions_fc = nn.Linear(level1_action_num, original_32)

        self.last_levele1_action_fc_emb = nn.Linear(level1_action_num, original_32)
        self.last_levele1_action_fc = nn.Linear(level1_action_num, original_32)

        self.last_target_fc_emb = nn.Linear(target_action_num, original_32)
        self.last_target_fc = nn.Linear(target_action_num, original_32)

        self.last_location_fc_emb = nn.Linear(location_action_num, original_32)
        self.last_location_fc = nn.Linear(location_action_num, original_32)

        self.fc_1 = nn.Linear(MP.scalar_encoder_fc1_input, original256)  # with relu
        self.fc_2 = nn.Linear(MP.scalar_encoder_fc2_input, scalar_context)  # with relu

        self.to(device)

    def forward(self, obs):
        '''
        obs [batch_size, 5, 66]
        agent_statistics [batch_size, 5, agent_stat]
        available_actions [batch_size, 5, 6]
        last_level1_action [batch_size, 5, 6]
        last_target [batch_size, 5, 10]
        last_location [batch_size, 5, 18]

        embedded_scalar_list这个和core有关[agent_statistics, avaul_action(当前5个智能体能做的多有动作的和），其他同理] [batch_size, 摊平]
        scalar_context这个和每个智能体选动作有关
        [agent_statistics, available_actions, last_level1_action,last_target,last_location] [ batch_size, 5, 66]
        '''
        # [agent_statistics, available_actions, last_level1_action, last_target, last_location] = obs

        agent_statistics = obs[:, :, :30]
        available_actions = obs[:, :, 30:36]
        last_level1_action = obs[:, :, 36:42]
        last_target = obs[:, :, 42:52]
        last_location = obs[:, :, 52: 67]

        batch_size = agent_statistics.shape[0]
        embedded_scalar_list = []
        scalar_context_list = []

        embedded_agent_stat = agent_statistics.reshape(batch_size, -1)

        the_log_statistics_emb = torch.log(embedded_agent_stat + 1)
        the_log_statistics = torch.log(agent_statistics + 1,)
        x1 = F.relu(self.statistics_fc_emb(the_log_statistics_emb))
        x2 = F.relu(self.statistics_fc(the_log_statistics))
        embedded_scalar_list.append(x1)
        scalar_context_list.append(x2)

        embed_avail_action = torch.sum(available_actions, dim=1)
        x1 = F.relu(self.available_actions_fc_emb(embed_avail_action))
        x2 = F.relu(self.available_actions_fc(available_actions))
        embedded_scalar_list.append(x1)
        # The embedding is also added to `scalar_context`
        scalar_context_list.append(x2)

        # 三个动作，到底是用具体的数值，还是用onehot：决定用one hot,但为了维度统一形状是[batch_size, 6]，哪个动作被用了就在对应的位置+1
        # 改了现在处理5个智能体的问题就解决了
        embed_last_level1_action = torch.sum(last_level1_action, dim=1)
        x1 = F.relu(self.last_levele1_action_fc_emb(embed_last_level1_action))
        x2 = F.relu(self.last_levele1_action_fc(last_level1_action))
        embedded_scalar_list.append(x1)
        scalar_context_list.append(x2)

        embed_last_level1_action = torch.sum(last_target, dim=1)
        x1 = F.relu(self.last_target_fc_emb(embed_last_level1_action))
        x2 = F.relu(self.last_target_fc(last_target))
        embedded_scalar_list.append(x1)
        scalar_context_list.append(x2)

        embed_last_location = torch.sum(last_location, dim=1)
        x1 = F.relu(self.last_location_fc_emb(embed_last_location))
        x2 = F.relu(self.last_location_fc(last_location))
        embedded_scalar_list.append(x1)
        scalar_context_list.append(x2)

        embedded_scalar = torch.cat(embedded_scalar_list, dim=1)
        embedded_scalar_out = F.relu(self.fc_1(embedded_scalar))

        scalar_context = torch.cat(scalar_context_list, dim=2)
        scalar_context_out = F.relu(self.fc_2(scalar_context))

        # scalar_context_out在空战好像用处不大，后面考虑去掉 todo
        return embedded_scalar_out, scalar_context_out

if __name__ == '__main__':
    model = ScalarEncoder()
    batch_size = 3
    scalar_list = [[], [], [], [], []]
    # 代表利用rawobs额外计算出来的信息
    agent_statistics = torch.ones((batch_size, 5, 15))
    # 5个智能体，当前能选的一级动作的onehot编码
    available_actions = torch.zeros((batch_size, 5, 6))

    # 上一步的动作
    last_level1_action = torch.zeros((batch_size,5, 6))
    last_target = torch.zeros((batch_size,5, 10))
    last_location = torch.zeros((batch_size,5, 18))
    scalar_list[0] = agent_statistics
    scalar_list[1] = available_actions
    scalar_list[2] = last_level1_action
    scalar_list[3] = last_target
    scalar_list[4] = last_location

    a, b = model.forward(scalar_list)
    print(a.shape)
    print(b.shape)