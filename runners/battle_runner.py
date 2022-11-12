""" A one line summary of the module or program
Copyright：©2011-2022 北京华如科技股份有限公司
This module provide configure file management service in i18n environment.
Authors: zhanghantang
DateTime:  2022/10/27 14:33
"""
import time
import torch
import numpy as np


from runners.base_runner import Runner
from algorithm.happo.happo_trainer import HAPPO
from algorithm.happo.happo_policy import HappoPolicy
from utils.replay_buffer import ReplayBuffer
from configs.hyper_parameters import Model_Parameters as MP

def _t2n(x):
    return x.detach().cpu().numpy()

class BattleRunner(Runner):
    def __init__(self, config):
        super(BattleRunner, self).__init__(config)

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']

        self.use_centralized_V = self.all_args.use_centralized_V

        # 运行相关
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.decision_step = self.all_args.decision_step
        self.enable_algorithm_step = self.all_args.enable_algorithm_step


        # 学习速率相关
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay

        # 神经网络相关
        self.hidden_size = MP.hidden_size

        # 保存
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        # self.model_dir = self.all_args.model_dir
        # self.run_dir = config["run_dir"]
        # self.log_dir = str(self.run_dir / 'logs')
        # if not os.path.exists(self.log_dir):
        #     os.makedirs(self.log_dir)
        # self.writter = SummaryWriter(self.log_dir)
        # self.save_dir = str(self.run_dir / 'models')
        # if not os.path.exists(self.save_dir):
        #     os.makedirs(self.save_dir)

        self.policy = []
        for agent_id in range(self.num_agents):
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]

            po = HappoPolicy(self.all_args,
                        self.envs.observation_space[agent_id],
                        share_observation_space,
                        self.envs.action_space[agent_id],
                        device=self.device)
            self.policy.append(po)

        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            # algorithm
            tr = HAPPO(self.all_args, self.policy[agent_id], device=self.device)
            # buffer
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else \
            self.envs.observation_space[agent_id]
            bu = ReplayBuffer(self.all_args,
                                       self.envs.observation_space[agent_id],
                                       share_observation_space,
                                       self.envs.action_space[agent_id])
            self.buffer.append(bu)
            self.trainer.append(tr)


    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        mode = 0

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            algorithm_step = 0
            for step in range(self.episode_length):
                # 分为三种mode
                # mode1：红由规则推进，蓝自由推进
                # mode2：红由算法推进，蓝自由推进
                # mode3：红空推，蓝自由推进
                print(step, 11111111111111111111)
                if step >= self.enable_algorithm_step and step % 3 == 0:
                    algorithm_step += 1
                    values, model_action, action_log_probs = self.collect(algorithm_step)
                    obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(model_action)
                    data = obs, share_obs, rewards, dones, infos, available_actions, \
                           values, model_action, action_log_probs
                    self.insert(data)
                else:
                    model_action = [[], [], [], [], []]
                    self.envs.step(model_action)

            self.compute()
            train_infos = self.train()


            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # # log information
            # if episode % self.log_interval == 0:
            #     end = time.time()
            #     print("\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
            #           .format(self.all_args.map_name,
            #                   episode,
            #                   episodes,
            #                   total_num_steps,
            #                   self.num_env_steps,
            #                   int(total_num_steps / (end - start))))
            #
            # if episode % self.eval_interval == 0 and self.use_eval:
            #     self.eval(total_num_steps)


    def warmup(self):
        obs, share_obs, available_actions = self.envs.reset()
        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs
        for agent_id in range(self.num_agents):
            print(self.buffer[agent_id].share_obs.shape)

            self.buffer[agent_id].share_obs[0] = share_obs[:, agent_id].copy()
            self.buffer[agent_id].obs[0] = obs[:, agent_id].copy()
            self.buffer[agent_id].available_actions[0] = available_actions[:, agent_id].copy()

    @torch.no_grad()
    def collect(self, step):
        value_collector = []
        action_collector = []
        action_log_prob_collector = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].masks[step],
                                                            self.buffer[agent_id].available_actions[step])
            value_collector.append(_t2n(value))
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
        # [self.envs, agents, dim]
        values = np.array(value_collector).transpose(1, 0, 2)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)

        return values, actions, action_log_probs


    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs = data

        dones_env = np.all(dones, axis=1)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        # active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # bad_masks = np.array(
        #     [[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in
        #      infos])

        if not self.use_centralized_V:
            share_obs = obs
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(share_obs[:, agent_id], obs[:, agent_id],
                                         actions[:, agent_id],
                                         action_log_probs[:, agent_id],
                                         values[:, agent_id], rewards[:, agent_id], masks[:, agent_id],
                                         # bad_masks[:, agent_id],active_masks[:, agent_id],
                                         available_actions[:, agent_id])

    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(self.buffer[agent_id].share_obs[-1],
                                                                  self.buffer[agent_id].masks[-1])
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)

    def train(self):
        train_infos = []
        factor = np.ones((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)

        for agent_id in torch.randperm(self.num_agents):
            self.trainer[agent_id].prep_training()
            self.buffer[agent_id].update_factor(factor)
            available_actions = None if self.buffer[agent_id].available_actions is None \
                else self.buffer[agent_id].available_actions[:-1].reshape(-1, *self.buffer[agent_id].available_actions.shape[2:])

            old_actions_logprob, _ = self.trainer[agent_id].policy.actor.evaluate_actions(
                self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])

            new_actions_logprob, _ = self.trainer[agent_id].policy.actor.evaluate_actions(
                self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))

            factor = factor * _t2n(
                torch.prod(torch.exp(new_actions_logprob - old_actions_logprob), dim=-1).reshape(self.episode_length,
                                                                                                 self.n_rollout_threads,
                                                                                                 1))
            train_infos.append(train_info)
            self.buffer[agent_id].after_update()

    def eval(self):
        pass