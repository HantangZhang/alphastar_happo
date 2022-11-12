""" A one line summary of the module or program
Copyright：©2011-2022 北京华如科技股份有限公司
This module provide configure file management service in i18n environment.
Authors: zhanghantang
DateTime:  2022/10/27 15:37
"""
import numpy as np
from gym.spaces import Discrete, Box


from agent_config import ADDRESS
from envs.xsim_battle.xsimenv.xsim_env import XSimEnv
from envs.xsim_battle.xsimenv.xsimenv_runner import EnvRunner
from envs.xsim_battle.agent_base import AgentState
from envs.xsim_battle.agent_base import MyJet, EnemyJet
from configs.hyper_parameters import Action_Space_Parameters as ASP
from configs.hyper_parameters import State_Space_Parameters as SSP
from agent.maybe.Maybe_Agent import MaybeAgent
from agent.yiteam.Yi_team import Yi_team

class Battle5v5Env():

    def __init__(self, args, env_id, time_ratio: int = 100, mode: str = 'host'):
        self.time_ratio = time_ratio
        self.env_id = env_id
        self.image = "xsim:v8.1"
        self.address = ADDRESS['ip'] + ':' + str(ADDRESS['port'] + env_id)
        self.mode = mode

        # agent数量为5，只能适应5v5场景
        self.num_agents = args.num_agents
        self.args = args

        # agent
        self.maybe = MaybeAgent('mayber', 'red', args)
        self.enemy_agent = Yi_team('blue', {"side": 'blue'})
        self.red_obs = None
        self.blue_obs = None

        self.observation_space = []
        self.share_observation_space = []
        self.action_space = []
        for i in range(self.num_agents):
            self.action_space.append(Discrete(ASP.level1_action_dim))
            self.observation_space.append(SSP.state_shape)
            self.share_observation_space.append(SSP.state_shape)

        self.agent_state = AgentState()
        self.my_jet_dic = {}
        self.enemy_jet_dic = {}

        # 统计信息
        self.battles_game = 0
        self.battles_win = 0

        self.episode_steps = 0
        self.episode_count = 0
        self.enable_algorithm_step = args.enable_algorithm_step



    def _launch(self):
        self.engine_env = XSimEnv(self.time_ratio, self.address, self.image, self.mode, local_test=True)
        self.engine_env.step([])

    def seed(self, seed):
        """Returns the random seed used by the environment."""
        self._seed = seed

    def step(self, model_action):
        if self.episode_steps == 1000:
            self.episode_steps = 0
        red_cmd_list = []
        mode = 0
        print(self.episode_steps, 2222222222222222)
        if self.episode_steps < self.enable_algorithm_step:
            red_cmd_list = self.maybe.init_move(self.red_obs)
        elif self.episode_steps >= self.enable_algorithm_step and self.episode_steps % 3 == 0:
            red_cmd_list = self.maybe.get_action(model_action, self.red_obs)
            mode = 2

        self.episode_steps += 1
        blue_cmd_list = self.enemy_agent.step(self.episode_steps, self.blue_obs)

        all_env_cmd = []
        all_env_cmd.extend(red_cmd_list)
        all_env_cmd.extend(blue_cmd_list)

        # 根据模型的动作，生成新的红蓝双方obs数据
        new_obs = self.engine_env.step(all_env_cmd)

        # 我方done数据判断
        dones = [False for _ in range(self.num_agents)]
        alive_agent = [plane['Name'] for plane in new_obs['red']['platforminfos']]
        if '红有人机' not in alive_agent:
            dones = [True, True, True, True, True]
        else:
            for p, agent_name in enumerate(self.my_jet_dic.keys()):
                if agent_name not in alive_agent:
                    dones[p] = True
        dones = np.array([dones])
        if mode == 2:
            # info数据 todo
            infos = [{'HelloWorld'} for _ in range(self.num_agents)]

            # 奖励计算
            rewards = self.compute_red_reward(new_obs, self.red_obs)

            # obs = self.agents['red'].get_obs(new_obs['red']) 等调用胡朝东规则将raw_obs处理成算法训练的obs todo
            obs = new_obs['red']
            # 暂不处理share——obs，以后通过写一个函数来处理 todo
            share_obs = new_obs['red']

            # 测试假数据
            available_actions = None
            obs = []
            share_obs = []
            available_actions = []
            for i in range(5):
                obs.append([1] * 163)
                share_obs.append([1] * 163)
                available_actions.append([0] * 6)
            rewards = [[self.compute_red_reward(new_obs, self.red_obs)]] * 5
            dones = np.zeros((5), dtype=bool)

            return obs, share_obs, rewards, dones, infos, available_actions
            # return obs, share_obs, np.array(rewards), np.array([dones]), np.array([infos]), available_actions

        return [[], [], [], np.array([dones]), [], []]

    def reset(self):
        self.episode_steps = 0
        if self.episode_count == 0:
            self._launch()

        self.engine_env.reset()
        obs = self.engine_env.step([])

        self.red_obs = obs['red']
        self.blue_obs = obs['blue']

        state, obs, baseline_state, level1_action_mask = self.maybe.reset()
        self.enemy_agent.reset()

        return state, obs, baseline_state, level1_action_mask

    def _init(self, raw_obs):
        # 初始化self.jet_list，将我方信息和敌方信息更新到
        my_obs_list = raw_obs['platforminfos']
        for obs_data in my_obs_list:
            if obs_data["Name"] not in self.jet_dic:
                jet = MyJet(obs_data)
                self.jet_dic[obs_data["Name"]] = jet

        enemy_obs_list = raw_obs['trackinfos']
        for obs_data in enemy_obs_list:
            if obs_data["Name"] not in self.jet_dic:
                jet = EnemyJet(obs_data)
                self.jet_dic[obs_data["Name"]] = jet

    def compute_red_reward(self, old_obs, new_obs):
        reward = 1

        return reward




