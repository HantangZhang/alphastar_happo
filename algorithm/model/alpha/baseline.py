import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.hyper_parameters import State_Space_Parameters as SSP
from configs.hyper_parameters import Model_Parameters as MP
from algorithm.model.alpha.spatial_encoder import ResBlock1D
from utils.util import check
'''
class ScoreCumulative(enum.IntEnum):
  """Indices into the `score_cumulative` observation."""
  score = 0
  idle_production_time = 1  空闲的建筑物时间
  idle_worker_time = 2  空闲的农名时间
  total_value_units = 3 所有的单位
  total_value_structures = 4 所有的建筑
  killed_value_units = 5 杀的单位书
  killed_value_structures = 6
  collected_minerals = 7
  collected_vespene = 8
  collection_rate_minerals = 9
  collection_rate_vespene = 10
  spent_minerals = 11
  spent_vespene = 12

# 当前总局拿的分数
score = 0
# 击落无人机数 
killed_drone_count = 0
# 击落有人机数量
killed_manned_count = 0
# 使用导弹数
used_missile_count = 0
# 躲避导弹数
avoid_missile_count = 0
# 导弹命中率
good_shot_rate = 0
'''

class Baseline(nn.Module):

    def __init__(self, baseline_shape=SSP.baseline_shape[0], original_64=MP.original_64,
                 baseline_input=MP.baseline_input, original_256=MP.original_256,
                 n_resblocks=MP.n_resblocks, n_cumulatscore= SSP.n_cumulatscore,
                 original_32=MP.original_32,
                 device=torch.device("cpu")):
        super().__init__()
        self.statistics_fc = nn.Linear(baseline_shape - n_cumulatscore, original_64)
        self.cumulatscore_fc = nn.Linear(n_cumulatscore, original_32)

        self.embed_fc = nn.Linear(baseline_input, original_256)  # with relu
        self.resblock_stack = nn.ModuleList([
            ResBlock1D(inplanes=original_256, planes=original_256, seq_len=1)
            for _ in range(n_resblocks)])

        self.out_fc = nn.Linear(original_256, 1)
        self.to(device)
        self.tpdv = dict(dtype=torch.float32, device=device)

    def forward(self, core_output, baseline_state):

        agent_statistics = baseline_state[:, :30]
        cumulative_score = baseline_state[:, 30:]
        agent_statistics = check(agent_statistics).to(**self.tpdv)
        cumulative_score = check(cumulative_score).to(**self.tpdv)

        embedded_scalar_list = []

        the_log_statistics = torch.log(agent_statistics + 1)
        x = F.relu(self.statistics_fc(the_log_statistics))
        embedded_scalar_list.append(x)

        score_log_statistics = torch.log(cumulative_score + 1)
        x = F.relu(self.cumulatscore_fc(score_log_statistics))
        embedded_scalar_list.append(x)

        embedded_scalar = torch.cat(embedded_scalar_list, dim=1)

        action_type_input = torch.cat([core_output, embedded_scalar], dim=1)

        x = self.embed_fc(action_type_input)

        x = x.unsqueeze(-1)
        for resblock in self.resblock_stack:
            x = resblock(x)
        x = x.squeeze(-1)

        x = F.relu(x)

        baseline = self.out_fc(x)

        # 归一化
        out = (2.0 / np.pi) * torch.atan((np.pi / 2.0) * baseline)

        return out


if __name__ == '__main__':
    # todo
    a = 1

