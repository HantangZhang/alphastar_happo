""" A one line summary of the module or program
Copyright：©2011-2022 北京华如科技股份有限公司
This module provide configure file management service in i18n environment.
Authors: zhanghantang
DateTime:  2022/10/31 17:23
"""

import torch
import torch.nn as nn
from algorithm.model.base.base_model import MLPBase, ACTLayer, init
from utils.util import check
from configs.hyper_parameters import Model_Parameters as MP

class Actor(nn.Module):
    """
    Actor network class for HAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(Actor, self).__init__()

        self.hidden_size = MP.hidden_size
        self.args = args
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks

        self.model = MLPBase(args, obs_space)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args)

        self.to(device)
        self.tpdv = dict(dtype=torch.float32, device=device)


    def forward(self, obs, masks, available_actions=None):

        obs = check(obs).to(**self.tpdv)

        actor_features = self.model(obs)
        a = actor_features.shape
        actions, action_probs = self.act(actor_features, available_actions)

        return actions, action_probs

    def evaluate_actions(self, obs, action, masks, available_actions=None, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        actor_features = self.model(obs)

        action_probs, dist_entropy = self.act.evaluate_actions(actor_features, action, available_actions=available_actions)

        return action_probs, dist_entropy


class Critic(nn.Module):

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(Critic, self).__init__()
        self.hidden_size = MP.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.cent_obs_shape = cent_obs_space
        self.model = MLPBase(args, self.cent_obs_shape)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)



    def forward(self, cent_obs, masks):
        cent_obs = check(cent_obs).to(**self.tpdv)
        critic_features = self.model(cent_obs)

        values = self.v_out(critic_features)

        return values