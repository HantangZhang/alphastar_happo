""" A one line summary of the module or program
Copyright：©2011-2022 北京华如科技股份有限公司
This module provide configure file management service in i18n environment.
Authors: zhanghantang
DateTime:  2022/11/1 9:48
"""
import torch
import torch.nn as nn
import argparse
from configs.hyper_parameters import Model_Parameters as MP


class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        # self.fc_h = nn.Sequential(init_(
        #     nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        # self.fc2 = get_clones(self.fc_h, self._layer_N)
        self.fc2 = nn.ModuleList([nn.Sequential(init_(
            nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size)) for i in range(self._layer_N)])

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x


class MLPBase(nn.Module):
    def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):
        super(MLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        # self._stacked_frames = args.stacked_frames
        self._layer_N = MP.layer_N
        self.hidden_size = MP.hidden_size

        obs_dim = obs_shape[0]

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(obs_dim, self.hidden_size,
                              self._layer_N, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x

class ACTLayer(nn.Module):

    def __init__(self, action_space, inputs_dim, use_orthogonal, gain, args=None):
        super(ACTLayer, self).__init__()

        action_dim = action_space.n
        self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)

    def forward(self, x, available_actions=None):

        action_logits = self.action_out(x, available_actions)
        actions = action_logits.sample()
        action_log_probs = action_logits.log_probs(actions)

        return actions, action_log_probs

    def get_probs(self, x, available_actions=None):
        action_logits = self.action_out(x, available_actions)
        action_probs = action_logits.probs

        return action_probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):

        action_logits = self.action_out(x, available_actions)
        action_log_probs = action_logits.log_probs(action)
        if active_masks is not None:
            if self.action_type == "Discrete":
                dist_entropy = (action_logits.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum()
            else:
                dist_entropy = (action_logits.entropy() * active_masks).sum() / active_masks.sum()
        else:
            dist_entropy = action_logits.entropy().mean()

        return action_log_probs, dist_entropy






def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Categorical, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        available_actions = None
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)