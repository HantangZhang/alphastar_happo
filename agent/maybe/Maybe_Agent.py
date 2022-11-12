""" A one line summary of the module or program
Copyright：©2011-2022 北京华如科技股份有限公司
This module provide configure file management service in i18n environment.
Authors: zhanghantang
DateTime:  2022/11/2 9:53
"""
import numpy as np

from configs.hyper_parameters import Model_Parameters as MP
from envs.xsim_battle.xsimenv.env_cmd import CmdEnv


class MaybeAgent(object):

    def __init__(self, name, side, args):
        super(MaybeAgent, self).__init__()

        self.num_agents = args.num_agents

        self.hidden_size = MP.hidden_size

        self.pre_action = []

        self.agent_id = {0: 1, 1: 2, 2: 11, 3: 12, 4: 13}

        # action_wrapper

    def step(self, model_action):
        red_env_cmd = self.get_action(model_action)
        self.pre_action = red_env_cmd

        return red_env_cmd

    def reset(self):

        # 返回未处理的初始化obs信息
        state = np.ones((5, 20, 163), dtype=np.float32)
        obs = np.ones((5, 5, 67), dtype=np.float32)
        baseline_state = np.ones((5, 36), dtype=np.float32)
        level1_action_mask = np.ones((5, 6))
        return state, obs, baseline_state, level1_action_mask

    def get_action(self, model_action, red_obs):
        # todo
        # model_action shape(5,1)
        cmd_list = []
        for i in range(len(model_action)):
            for agent in range(5):
                cmd_list.append(CmdEnv.make_areapatrolparam(self.agent_id[agent], 0, 0, 9000, 200, 100, 300, 1, 6))
        return cmd_list

    def init_move(self, obs):
        cmd_list = []
        for value in self.agent_id.values():
            cmd_list.append(CmdEnv.make_areapatrolparam(value, 0, 0, 9000, 200, 100, 300, 1, 6))
        return cmd_list

    def process_entity_obs(self, raw_obs):
        # 处理实体的obs信息,处理后的信息可以用于处理scalarobs todo
        # 直接处理obs['platforminfos']，obs['trackinfos']和obs['missileinfos']的信息
        # 需要更新两个类，一个是每个实体自身的类myjet，另一个是整体的agent_state
        entity_obs = []
        # 每个飞机实体存在self.jet_dic中，更新这个字典
        my_obs_list = raw_obs['platforminfos']
        for my_obs in my_obs_list:
            self.jet_dic[my_obs['Name']].update_agent_info()

        # 利用更新的agent的信息来编写提取obs的逻辑 todo
        self.entity_obs_list.append(self.jet_dic[''].X)

        # 将处理后的信息用于更新agent_state类，包括列表和字典，字典里的数据用于规则函数计算 todo
        self.agent_state.entity_state = self.entity_obs_list

        # todo
        self.agent_state.entity_state_dic['distance'] = self.entity_obs_list[34]

    def process_scalar_obs(self):
        # 同process_entity_obs todo
        self.agent_state.scalar_state = self.scalar_obs_list

