""" A one line summary of the module or program
Copyright：©2011-2022 北京华如科技股份有限公司
This module provide configure file management service in i18n environment.
Authors: zhanghantang
DateTime:  2022/11/4 14:20
"""
import torch
import torch.nn as nn
from torch.optim import Adam

from algorithm.model.alpha.entity_encoder import EntityEncoder
from algorithm.model.alpha.core import Core
from algorithm.model.alpha.action_type_head import ActionTypeHead
from algorithm.model.alpha.location_head import LocationHead
from algorithm.model.alpha.selected_target_head import SelectedTargetHead
from algorithm.model.alpha.scalar_encoder import ScalarEncoder
from algorithm.model.alpha.baseline import Baseline
from utils.util import check

'''
1. 什么是glu
2. target的mask操作，目标只有10个己方单位，我考虑把所有导弹信息都加进去，例如变成10个飞机+10个导弹
3. ScoreCumulative
4. scalar_context对于一级动作选择的影响 已解决
5. prep_rollout()的实现
6. baseline_state的实现
7. scalar_context还没有按预先的分成5条分给5个actor
8. 初始化模型参数没有完成
9. guide里面的动态目标还没有实现

每次更新跑多少个步长参数的时候需要检查
algorithm_step
还有buffer当中
self.step = (self.step + 1) % (self.episode_length - 1) 确保最后一步的数据正确存入buffer当中

core

action_size = 564
All_Units_Size = 259
entities_size = 512
def action_logits_by_state(self, state, hidden_state = None, single_inference = False, obs = None):
        # batch_size = 1 if single_inference else None
        # sequence_length = 1 if single_inference else None
        batch_size = 16
        sequence_length = 8


如果是同一个网络，因为obs输入的信息不同，会走整个流程，输出不一样的动作
如果是多个决策网络，那么core到这里处理全局信息，后面每个智能体有自己的决策网络，他们输入的是core处理的全局信息，和自己的单独信息scalar_context
那么这里就需要研究一下glu还有没有用了:应该是没问题的
那么scalar_context就需要好好设计一下了

alphastar梯度更新直接参考learner里面的run就行
下面三个大块的任务：
参考alphastar的loss_funtion计算loss的计算
利用happo的方式编写trainer和policy
测通代码
'''


class AirEnocderModel(nn.Module):

    def __init__(self, device=torch.device("cpu")):
        super(AirEnocderModel, self).__init__()
        self.entity_encoder = EntityEncoder(device=device)
        self.core = Core(device=device)
        self.scalar_encoder = ScalarEncoder(device=device)

        self.tpdv = dict(dtype=torch.float32, device=device)

    def init_parameters(self):
        pass

    def forward(self, state, scalar_list, sequence_length=None, hidden_state=None):
        # embedded_scalar和embedded_entity都等于=[batch_size x embedded_size]
        state = check(state).to(**self.tpdv)
        scalar_list = check(scalar_list).to(**self.tpdv)

        entity_embeddings, embedded_entity = self.entity_encoder(state)
        embedded_scalar, scalar_context = self.scalar_encoder(scalar_list)

        core_output, hidden_state = self.core(embedded_scalar, embedded_entity, sequence_length, hidden_state)


        return core_output, scalar_context, entity_embeddings


class Actor(nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super(Actor, self).__init__()
        self.action_type_head = ActionTypeHead(device=device)
        self.target_unit_head = SelectedTargetHead(device=device)
        self.location_head = LocationHead(device=device)

        self.tpdv = dict(dtype=torch.float32, device=device)
        # 把模型放到指定device上

    def forward(self,core_output, scalar_context, entity_embeddings, level1_action_mask=None):

        level1_action_mask = check(level1_action_mask).to(**self.tpdv)


        level1_action_log_prob, level1_action, autoregressive_embedding = self.action_type_head(core_output, scalar_context,
                                                                                          level1_action_mask)

        target_jet_prob, target_jet = self.target_unit_head(autoregressive_embedding,
                                                                level1_action, entity_embeddings)
        location_prob, location_id = self.location_head(autoregressive_embedding, level1_action)


        return level1_action, level1_action_log_prob, location_id, location_prob, target_jet, target_jet_prob


    def evaluate_actions(self, core_output, scalar_context, entity_embeddings, level1_action_mask, level1_action, target_jet, location_id,  active_masks):
        level1_action_mask = check(level1_action_mask).to(**self.tpdv)
        level1_action = check(level1_action).to(**self.tpdv)
        active_masks = check(active_masks).to(**self.tpdv)
        target_jet = check(target_jet).to(**self.tpdv)
        location_id = check(location_id).to(**self.tpdv)

        level1_action_log_prob, level1_entropy, autoregressive_embedding = self.action_type_head.evaluate_actions(core_output, scalar_context,
                                                                                        level1_action,level1_action_mask, active_masks)

        target_probs, target_entropy = self.target_unit_head.evaluate_actions(autoregressive_embedding, level1_action,
                                                                              entity_embeddings, target_jet,
                                                                              active_masks)
        location_probs, location_entropy = self.location_head.evaluate_actions(autoregressive_embedding, level1_action, location_id, active_masks)

        return level1_action_log_prob, level1_entropy, target_probs, target_entropy, location_probs, location_entropy



class Critic(nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super(Critic, self).__init__()

        self.baseline = Baseline(device=device)
        self.to(device)

    def forward(self, core_output, baseline_state):

        value = self.baseline(core_output, baseline_state)

        return value
