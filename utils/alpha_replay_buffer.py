import numpy as np
import torch

from configs.hyper_parameters import State_Space_Parameters as SSP
from configs.hyper_parameters import Action_Space_Parameters as ASP

class AlphaReplayBuffer(object):

    def __init__(self, args, agent_id, state_shape=SSP.state_shape, obs_shape=SSP.obs_shape,
                 level1_action_space=ASP.level1_action_dim, baseline_shape=SSP.baseline_shape):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.num_agents = args.num_agents
        self.agent_id = agent_id

        self.state = np.zeros((self.episode_length + 1, self.n_rollout_threads, *state_shape), dtype=np.float32)
        # [2, 5, 66]
        self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, *obs_shape), dtype=np.float32)
        self.baseline_state = np.zeros((self.episode_length + 1, self.n_rollout_threads, *baseline_shape), dtype=np.float32)

        self.rewards = np.zeros((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)

        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        self.active_masks = np.ones_like(self.masks)


        self.value_preds = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        self.returns = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)

        self.level1_action_mask = np.ones((self.episode_length + 1, self.n_rollout_threads, level1_action_space),
                                         dtype=np.float32)

        self.l1_action = np.zeros((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
        self.l1_action_prob = np.zeros((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
        self.location = np.zeros((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
        self.location_prob = np.zeros((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
        self.target = np.zeros((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
        self.target_prob = np.zeros((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)

        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm

        self.step = 0

    def insert(self, state, obs, baseline_state, rewards, values,
                   level1_action_mask, l1_action,
                   l1_action_prob, location,
                   location_prob, target, target_prob, masks, active_masks=None):
        self.state[self.step + 1] = state.copy()
        self.obs[self.step + 1] = obs.copy()
        self.baseline_state[self.step + 1] = baseline_state.copy()
        self.rewards[self.step] = rewards.copy()
        self.value_preds[self.step + 1] = values.copy()

        self.level1_action_mask[self.step + 1] = level1_action_mask.copy()
        self.l1_action[self.step] = l1_action.copy()
        self.l1_action_prob[self.step] = l1_action_prob.copy()
        self.location[self.step] = location.copy()
        self.location_prob[self.step] = location_prob.copy()
        self.target[self.step] = target.copy()
        self.target_prob[self.step] = target_prob.copy()

        self.masks[self.step + 1] = masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()

        self.step = (self.step + 1) % (self.episode_length)


    def compute_returns(self, next_value, value_normalizer=None):

        if self._use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.shape[0])):
                if self._use_popart or self._use_valuenorm:
                    delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) * \
                            self.masks[step + 1] - value_normalizer.denormalize(self.value_preds[step])
                    gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                    self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                else:
                    delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                            self.value_preds[step]
                    gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                    self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.shape[0])):
                self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]

    def update_factor(self, factor1, factor2, factor3):
        self.level1_factor = factor1.copy()
        self.target_factor = factor2.copy()
        self.location_factor = factor3.copy()

    def after_update(self):
        self.state[0] = self.state[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.baseline_state[0] = self.baseline_state[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        self.level1_action_mask[0] = self.level1_action_mask[-1].copy()

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, n_rollout_threads * episode_length,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        '''
        state, obs, baseline_state, level1_action_mask, level1_action, target_jet, location_id, active_masks, old_level1_action_probs_batch, \
        , old_target_probs, , old_location_probs,, adv_targ, \
        level1_factor, target_factor, location_factor, value_preds_batch, return_batch'''
        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]
        state = self.state[:-1].reshape(-1, *self.state.shape[2:])
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        baseline_state = self.baseline_state[:-1].reshape(-1, *self.baseline_state.shape[2:])
        level1_action_mask = self.level1_action_mask.reshape(-1, *self.level1_action_mask.shape[2:])
        level1_action = self.l1_action.reshape(-1, 1)
        target_jet = self.target.reshape(-1, 1)
        location_id = self.location.reshape(-1, 1)
        active_masks = self.active_masks.reshape(-1, *self.active_masks.shape[2:])
        old_level1_action_probs = self.l1_action_prob.reshape(-1, 1)
        target_prob = self.target_prob.reshape(-1, 1)
        location_prob = self.location_prob.reshape(-1, 1)
        adv_targ = advantages.reshape(-1, 1)
        level1_factor = self.level1_factor.reshape(-1, 1)
        target_factor = self.target_factor.reshape(-1, 1)
        location_factor = self.location_factor.reshape(-1, 1)
        value_preds = self.value_preds.reshape(-1, 1)
        returns = self.returns.reshape(-1, 1)

        for indices in sampler:
            state_batch = state[indices]
            obs_batch = obs[indices]
            baseline_state_batch = baseline_state[indices]
            level1_action_mask_batch = level1_action_mask[indices]
            level1_action_batch = level1_action[indices]
            target_jet_batch = target_jet[indices]
            location_id_batch = location_id[indices]
            active_masks_batch = active_masks[indices]
            old_level1_action_probs_batch = old_level1_action_probs[indices]
            target_prob_batch = target_prob[indices]
            location_prob_batch = location_prob[indices]
            adv_targ_batch = adv_targ[indices]
            level1_factor_batch = level1_factor[indices]
            target_factor_batch = target_factor[indices]
            location_factor_batch = location_factor[indices]
            value_preds_batch = value_preds[indices]
            returns_batch = returns[indices]

            yield state_batch, obs_batch, baseline_state_batch, level1_action_mask_batch, level1_action_batch, target_jet_batch, location_id_batch, active_masks_batch, \
                  old_level1_action_probs_batch, target_prob_batch, location_prob_batch, adv_targ_batch, level1_factor_batch, \
                  target_factor_batch, location_factor_batch, value_preds_batch, returns_batch









if __name__ == '__main__':
    pass