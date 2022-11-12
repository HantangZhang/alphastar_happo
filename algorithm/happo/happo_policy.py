""" A one line summary of the module or program
Copyright：©2011-2022 北京华如科技股份有限公司
This module provide configure file management service in i18n environment.
Authors: zhanghantang
DateTime:  2022/10/31 11:20
"""
import torch
from algorithm.model.base.actor_critic import Actor, Critic
from utils.util import update_linear_schedule


class HappoPolicy:

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.args = args
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr

        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = Critic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, masks, available_actions=None,):
        actions, action_log_probs = self.actor(obs,
                                             masks,
                                             available_actions,
                                             )

        values = self.critic(cent_obs, masks)
        return values, actions, action_log_probs

    def get_values(self, cent_obs, masks):
        values, _ = self.critic(cent_obs, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, action, masks,
                         available_actions=None, active_masks=None):
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks)

        values = self.critic(cent_obs, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, masks, available_actions=None):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor(obs, masks, available_actions)
        return actions