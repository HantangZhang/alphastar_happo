"""
@FileName：demo_agent.py
@Description：
@Author：
@Time：2021/6/17 上午9:21
@Department：AIStudio研发部
@Copyright：©2011-2021 北京华如科技股份有限公司
"""

from typing import List
import copy
import random
from env.env_cmd import CmdEnv


class AgentWrapper(object):

    def __init__(self, agent, side, **kwargs):
        self.agent = agent(side, {"side": side})

    def reset(self, side):
        self.agent.reset()

    def predict(self, obs_side, **kwargs) -> List[dict]:
        """ 步长处理
        此方法继承自基类中的step(self,sim_time, obs_red, **kwargs)
        选手通过重写此方法，去实现自己的策略逻辑，注意，这个方法每个步长都会被调用
        :return: 决策完毕的任务指令列表
        """  
        sim_time = obs_side['platforminfos'][0]['CurTime']
        # print(obs_side)
        try:
            cmd_list = self.agent.step(sim_time, obs_side)
        except Exception as e:
            print(e)
            print("AI get wrong")
            cmd_list = []
        # print("bbbbbbbbbbbbbbb")
        # print(cmd_list)
        return cmd_list

    def save(self, model_dir):
        pass

    def load(self, model_dir):
        pass

    def get_weights(self):
        return None

    def set_weights(self, model_dir):
        pass

