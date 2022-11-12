""" A one line summary of the module or program
Copyright：©2011-2022 北京华如科技股份有限公司
This module provide configure file management service in i18n environment.
Authors: zhanghantang
DateTime:  2022/10/27 16:50
"""

import numpy as np
import copy
from envs.xsim_battle.xsimenv.env_cmd import CmdEnv
from envs.xsim_battle.utils_math import TSVector3

class MyJet(object):
    # 实例化每架智能体
    def __init__(self, data: dict):
        # 平台编号
        self.ID = data['ID']
        # x轴坐标(浮点型, 单位: 米, 下同)
        self.X = data['X']
        # y轴坐标
        self.Y = data['Y']
        # z轴坐标
        self.Z = data['Alt']
        # 航向(浮点型, 单位: 度, [0-360])
        self.Pitch = data['Pitch']
        # 横滚角
        self.Roll = data['Roll']
        # 航向, 即偏航角
        self.Heading = data['Heading']
        # 速度
        self.Speed = data['Speed']
        # 当前可用性
        self.Availability = data['Availability']
        # 类型
        self.Type = data['Type']
        # 仿真时间
        self.CurTime = data['CurTime']
        # 军别信息
        self.Identification = data['Identification']
        # 是否被锁定
        self.IsLocked = data['IsLocked']
        # 剩余弹药
        self.LeftWeapon = data['LeftWeapon']
        # 名字
        self.Name = data["Name"]

        # 坐标2d
        self.pos2d = {"X": self.X, "Y": self.Y}
        # 坐标3d
        self.pos3d = {"X": self.X, "Y": self.Y, "Z": self.Z}

        if self.Type == 1:
            self.num_avail_ms = 3
        else:
            self.num_avail_ms = 2

    def update_agent_info(self, agent_data):
        self.X = agent_data['X']

class EnemyJet(object):
    def __init__(self, agent):
        # 平台编号
        self.ID = agent['ID']
        # x轴坐标(浮点型, 单位: 米, 下同)
        self.X = agent['X']
        # y轴坐标
        self.Y = agent['Y']
        # z轴坐标
        self.Z = agent['Alt']
        # 航向(浮点型, 单位: 度, [0-360])
        self.Pitch = agent['Pitch']
        # 横滚角
        self.Roll = agent['Roll']
        # 航向, 即偏航角
        self.Heading = agent['Heading']
        # 速度
        self.Speed = agent['Speed']
        # 当前可用性
        self.Availability = agent['Availability']
        # 类型
        self.Type = agent['Type']
        # 仿真时间
        self.CurTime = agent['CurTime']
        # 军别信息
        self.Identification = agent['Identification']
        # 名字
        self.Name = agent["Name"]

        # 坐标
        self.pos2d_dic = {"X": self.X, "Y": self.Y}
        self.pos3d_dic = {"X": self.X, "Y": self.Y, "Z": self.Z}
        self.pos2d = np.array([self.X, self.Y])
        self.pos3d = np.array([self.X, self.Y, self.Z])
        # 被几发导弹瞄准等等
        self.num_locked_missile = 0

        self.attacked_missile_list = []
        self.previous_attacked_missile_list = []

        self.alive = False

        if self.Tyep == 1:
            self.EnemyLeftWeapon = 4
        else:
            self.EnemyLeftWeapon = 2

    def __eq__(self, other):
        return True if self.ID == other.ID else False

    def update_agent_info(self, agent):
        self.X = agent['X']
        self.Y = agent['Y']
        self.Z = agent['Alt']
        self.Pitch = agent['Pitch']
        self.Heading = agent['Heading']
        self.Roll = agent['Roll']
        self.Speed = agent['Speed']
        self.Availability = agent['Availability']
        self.Type = agent['Type']
        self.pos2d_dic = {"X": self.X, "Y": self.Y}
        self.pos3d_dic = {"X": self.X, "Y": self.Y, "Z": self.Z}
        self.pos2d = np.array([self.X, self.Y])
        self.pos3d = np.array([self.X, self.Y, self.Z])

        self.previous_attacked_missile_list = copy.copy(self.attacked_missile_list)
        # 清零，然后在decision——making里面更新
        self.attacked_missile_list = []

        self.alive = False

class AgentState(object):

    dead_jet_list = []




    def __init__(self, entity_obs=None, scalar_entity=None):
        super(AgentState, self).__init__()

        self.entity_state = entity_obs
        self.scalar_state = scalar_entity

        self.entity_state_dic = {}
        self.scalar_state_dic = {}

        self.dead_jet_list = []


        self._shape = None

    def _get_shape(self):
        pass

    def _tolist(self):
        return [self.entity_state, self.statistical_state]

    @property
    def shape(self):
        if self._shape is None:
            self._get_shape()

        return self._shape


    @property
    def device(self):
        return self.entity_state.device




