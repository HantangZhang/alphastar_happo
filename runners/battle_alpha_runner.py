
import time
import torch
import numpy as np


from runners.base_runner import Runner
from algorithm.happo.happo_trainer import HAPPO
from algorithm.happo.happo_policy import HappoPolicy
from utils.replay_buffer import ReplayBuffer
from configs.hyper_parameters import Model_Parameters as MP

from algorithm.model.alpha.airbattle_model import AirEnocderModel
from algorithm.happo.happo_alpha_policy import HappoAlphaPolicy
from algorithm.happo.happo_alpha_trainer import HappoAlphaTrainer
from utils.alpha_replay_buffer import AlphaReplayBuffer


def _t2n(x):
    return x.detach().cpu().numpy()

class BattleAlphaRunner(Runner):
    def __init__(self, config):
        super(BattleAlphaRunner, self).__init__(config)
        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']

        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.decision_step = self.all_args.decision_step
        self.enable_algorithm_step = self.all_args.enable_algorithm_step

        self.encoder_model = AirEnocderModel(device=self.device)

        self.policy = []
        for agent_id in range(self.num_agents):
            po = HappoAlphaPolicy(self.all_args, device=self.device)
            self.policy.append(po)

        self.trainer = []
        self.buffer = []
        # 注意这里，虽然每个policy不同，但是encoder是同一个，也就是agent0更新了encoder，那么agent1的trainer用的就是0更新完的encoder，以此类推
        for agent_id in range(self.num_agents):
            tr = HappoAlphaTrainer(self.all_args, self.encoder_model, self.policy[agent_id],device=self.device)
            bu = AlphaReplayBuffer(self.all_args, agent_id=agent_id)
            self.buffer.append(bu)
            self.trainer.append(tr)



    def run(self):
        self.warmup()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        engine_episode_length = self.episode_length * 3 + 201
        print('引擎推进步长为', engine_episode_length)
        for episode in range(episodes):
            # if self.use_linear_lr_decay:
            #     self.trainer.policy.lr_decay(episode, episodes)
            algorithm_step = 0
            # 循环episode_length步，但引擎最多智能存episode_length % 3的数据， 所以要么把推进的步长增加，要么把寸的维度降低
            # 200 66  600  (600 -200) // 3
            # 如果算法设置的episode_length为200， 那么buffer中的step +1 = 199应该就是最后一步
            print('开始第{}轮训练'.format(episode))
            for step in range(engine_episode_length + 1):
                # 分为三种mode
                # mode1：红由规则推进，蓝自由推进
                # mode2：红由算法推进，蓝自由推进
                # mode3：红空推，蓝自由推进
                if step >= self.enable_algorithm_step and step % 3 == 0:
                    print('进入算法推进引擎, 进入步长为', step, '算法更新步长为', algorithm_step)
                    values, l1_actions, l1_action_probs, locations, location_probs, targets, target_probs = self.collect(
                        algorithm_step)
                    model_action = [l1_actions, locations, targets]
                    # state [batch_size, 5, 20, 163]
                    # obs [batch_size, 5, 66]
                    # baseline_state [batch_size, 6+32*5]
                    state, obs, baseline_state, rewards, dones, infos, level1_action_mask = self.envs.step(model_action)
                    data = state, obs, baseline_state, rewards, dones, infos, level1_action_mask, l1_actions, l1_action_probs, \
                            locations, location_probs, targets, target_probs, values
                    self.insert(data)
                    algorithm_step += 1
                else:
                    model_action = [[], [], []]
                    self.envs.step(model_action)
            self.compute()
            train_infos = self.train()



    def warmup(self):
        state, obs, baseline_state, level1_action_mask = self.envs.reset()

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].state[0] = state[:, agent_id].copy()
            self.buffer[agent_id].obs[0] = obs[:, agent_id].copy()
            self.buffer[agent_id].baseline_state[0] = baseline_state[:, agent_id].copy()
            self.buffer[agent_id].level1_action_mask[0] = level1_action_mask[:, agent_id].copy()

    @torch.no_grad()
    def collect(self, step):
        value_collector = []
        l1_action_collector = []
        l1_action_prob_collector = []
        location_collector = []
        location_prob_collector = []
        target_collector = []
        target_prob_collector = []

        for agent_id in range(self.num_agents):
            # self.trainer[agent_id].prep_rollout()
            core_output, scalar_context, entity_embeddings = self.encoder_model(
                self.buffer[agent_id].state[step], self.buffer[agent_id].obs[step])
            value, l1_action, l1_action_prob, location, location_prob, target, target_prob = self.trainer[
                agent_id].policy.get_actions(
                core_output, scalar_context[:, agent_id, :], entity_embeddings, self.buffer[agent_id].baseline_state[step],
                self.buffer[agent_id].level1_action_mask[step]
            )
            value_collector.append(_t2n(value))
            l1_action_collector.append(_t2n(l1_action))
            l1_action_prob_collector.append(_t2n(l1_action_prob))
            location_collector.append(_t2n(location))
            location_prob_collector.append(_t2n(location_prob))
            target_collector.append(_t2n(target))
            target_prob_collector.append(_t2n(target_prob))

        values = np.array(value_collector).transpose(1, 0, 2)
        l1_actions = np.array(l1_action_collector).transpose(1, 0, 2)
        l1_actions_probs = np.array(l1_action_prob_collector).transpose(1, 0, 2)
        locations = np.array(location_collector).transpose(1, 0, 2)
        location_probs = np.array(location_prob_collector).transpose(1, 0, 2)
        targets = np.array(target_collector).transpose(1, 0, 2)
        target_prob = np.array(target_prob_collector).transpose(1, 0, 2)

        return values, l1_actions, l1_actions_probs, locations, location_probs, targets, target_prob

    def insert(self, data):
        # infor 和 done 目前可能不需要
        state, obs, baseline_state, rewards, dones, infos, level1_action_mask, l1_actions, l1_action_probs, \
        locations, location_probs, targets, target_probs, values = data

        dones_env = np.all(dones, axis=1)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # obs[:, agent_id] = [batch_size, 5, 67]
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(state[:, agent_id], obs[:, agent_id], baseline_state[:, agent_id],
                                         rewards[:, agent_id], values[:, agent_id],
                                         level1_action_mask[:, agent_id], l1_actions[:, agent_id],
                                         l1_action_probs[:, agent_id], locations[:, agent_id],
                                         location_probs[:, agent_id], targets[:, agent_id], target_probs[:, agent_id],
                                         masks[:, agent_id], active_masks[:, agent_id]
                                         )

    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            core_output, _, _ = self.encoder_model(
                self.buffer[agent_id].state[-1], self.buffer[agent_id].obs[-1])
            next_value = self.trainer[agent_id].policy.get_values(core_output, self.buffer[agent_id].baseline_state[-1])
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)


    def train(self):
        train_infos = []
        level1_factor = np.ones((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
        target_factor = np.ones((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
        location_factor = np.ones((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)

        for agent_id in torch.randperm(self.num_agents):
            # self.trainer[agent_id].prep_training()
            self.buffer[agent_id].update_factor(level1_factor, target_factor, location_factor)
            core_output, scalar_context, entity_embeddings = self.encoder_model(
                self.buffer[agent_id].state[:-1].reshape(-1, *self.buffer[agent_id].state.shape[2:]),
                self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]))

            # 所有的reshape这一行就是先将201变成200， 然后合并episode_length和n_rollout_threads
            # 例如(200, 32, 20, 163)变为了(6400, 20, 163)
            old_level1_action_log_prob, _, old_target_probs, _, old_location_probs, _ = self.trainer[
                agent_id].policy.actor.evaluate_actions(core_output, scalar_context[:, agent_id, :], entity_embeddings,
                self.buffer[agent_id].level1_action_mask[:-1].reshape(-1,*self.buffer[agent_id].level1_action_mask.shape[2:]),
                self.buffer[agent_id].l1_action.reshape(-1, *self.buffer[agent_id].l1_action.shape[2:]),
                self.buffer[agent_id].target.reshape(-1, *self.buffer[agent_id].target.shape[2:]),
                self.buffer[agent_id].location.reshape(-1, *self.buffer[agent_id].location.shape[2:]),
                self.buffer[agent_id].active_masks.reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))

            train_info = self.trainer[agent_id].train(self.buffer[agent_id])

            new_level1_action_log_prob, _, new_target_probs, _, new_location_probs, _ = self.trainer[
                agent_id].policy.actor.evaluate_actions(core_output, scalar_context[:, agent_id, :], entity_embeddings,
                self.buffer[agent_id].level1_action_mask[:-1].reshape(-1, * self.buffer[agent_id].level1_action_mask.shape[2:]),
                self.buffer[agent_id].l1_action.reshape(-1, *self.buffer[agent_id].l1_action.shape[2:]),
                self.buffer[agent_id].target.reshape(-1, *self.buffer[agent_id].target.shape[2:]),
                self.buffer[agent_id].location.reshape(-1, *self.buffer[agent_id].location.shape[2:]),
                self.buffer[agent_id].active_masks.reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))

            level1_factor = level1_factor * _t2n(
                torch.prod(torch.exp(new_level1_action_log_prob - old_level1_action_log_prob), dim=-1).reshape(
                    self.episode_length,
                    self.n_rollout_threads,
                    1))
            target_factor = target_factor * _t2n(
                torch.prod(torch.exp(new_target_probs - old_target_probs), dim=-1).reshape(
                    self.episode_length,
                    self.n_rollout_threads,
                    1))
            location_factor = location_factor * _t2n(
                torch.prod(torch.exp(new_location_probs - old_location_probs), dim=-1).reshape(
                    self.episode_length,
                    self.n_rollout_threads,
                    1))

            train_infos.append(train_info)
            self.buffer[agent_id].after_update()

if __name__ == '__main__':
    state_shape = [20, 163]
    obs_shape = [5, 66]
    state = np.zeros((201, 32, *state_shape))
    obs = np.zeros((201, 32, *obs_shape))
    print(state.shape)
    print(state[:-1].shape)
    print(state.shape[2:])
    print(state[:-1].reshape(-1, *state.shape[2:]).shape)


