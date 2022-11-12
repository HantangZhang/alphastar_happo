""" A one line summary of the module or program
Copyright：©2011-2022 北京华如科技股份有限公司
This module provide configure file management service in i18n environment.
Authors: zhanghantang
DateTime:  2022/10/28 13:03
"""
import numpy as np
# ValueError: could not broadcast input array from shape (2,5,163) into shape (2,163)
from utils.util import check
import torch
import torch.nn as nn




# probs = torch.zeros((3, 16))
# probs[0][3] = 23
# action[0] = 3
# x = torch.index_select(probs, 0, action)
# print(x)


class Model1(nn.Module):

    def __init__(self,):
        super(Model1, self).__init__()
        self.layer = nn.Linear(4, 4)

    def forward(self, x):
        out = self.layer(x)
        out2 = out * 2
        return out, out2


class Model2(nn.Module):

    def __init__(self,):
        super(Model2, self).__init__()
        self.layer = nn.Linear(4, 1)

    def forward(self, x):
        out = self.layer(x)
        return out


class Actor(nn.Module):

    def __init__(self):
        super(Actor, self).__init__()
        self.model1 = Model1()
        self.model2 = Model2()

        self.opt1 = torch.optim.Adam(self.model1.parameters())
        self.opt2 = torch.optim.Adam(self.model2.parameters())

    def forward(self, x1, x2):
        res1 = self.model1(x1)
        res2 = self.model2(x2)

        return res1, res2
model = Model1()
model2 = Model2()
x1 = torch.ones(((3, 4)), dtype=torch.float32)
x2 = torch.full((3, 4), 2, dtype=torch.float32)
res1, out2 = model(x1)
res2 = model2(out2)
print(res1)
print(res2)
print(model.layer.weight.grad)
loss = torch.sum(res2, dim=-1).mean()
loss.backward()
print(model.layer.weight.grad)
print(model2.layer.weight.grad)
# new_action1 = torch.full((3, 1), 10, dtype=torch.float32)
# new_action2 = torch.full((3, 1), 8, dtype=torch.float32)
# loss1 = new_action1 - res1
# loss2 = new_action1 -
# loss1 = torch.sum(loss1, dim=-1).mean()
# loss2 = torch.sum(loss2, dim=-1).mean()
# print(model.model1.layer.weight.grad)
# 第一种方式
# loss = loss1 + loss2
# loss.backward()
# print(model.model1.layer.weight.grad)
# print(model.model2.layer.weight.grad)
# 第一种方式
# loss1.backward()
# loss2.backward()
# print(model.model1.layer.weight.grad)
# print(model.model1.layer.weight.grad)


# model1 = Model1()
# model2 = Model2()
# opt1 = torch.optim.Adam(model1.parameters())
# opt2 = torch.optim.Adam(model2.parameters())
# x1 = torch.ones(((3, 4)), dtype=torch.float32)
# x2 = torch.full((3, 4), 2, dtype=torch.float32)
#
# action1 = model1(x1)
# action2 = model2(x2)
#
# new_action1 = torch.full((3, 1), 10, dtype=torch.float32)
# new_action2 = torch.full((3, 1), 8, dtype=torch.float32)
# loss1 = new_action1 - action1 + action2
# loss2 = new_action2 - action2
#
# loss1 = torch.sum(loss1, dim=-1).mean()
# loss2 = torch.sum(loss2, dim=-1).mean()
# print(model1.layer.weight.grad)
#
# # loss1.backward()
# # loss2.backward()
# loss = loss1 + loss2
# loss = torch.tensor([1.0], requires_grad=True)
# loss.backward()
# print(model1.layer.weight.grad)
# print(model2.layer.weight.grad)










# import math
# from envs.xsim_battle.utils_math import TSVector3
# import tensorflow as tf
# from gym.spaces import Box
# import torch
#
# x = np.array([[1, 1.5, 1.7, 2], [1.1, 2.3, 1.4, 1.8]])
# print(x.shape)
# action_dim = 4
#
# # initializer = tf.keras.initializers.orthogonal()
# # bias_initializer = tf.keras.initializers.constant()
# #
# # linear = tf.keras.layers.Dense(action_dim, kernel_initializer=initializer,
# #                                                 bias_initializer=bias_initializer)
# softmax = tf.keras.layers.Softmax(axis=-1)
#
# # x = linear(x)
# # print(x)
# x1 = tf.Variable(x, dtype=tf.float32)
# action_type_probs = softmax(x1)
# case1 = -tf.math.log(action_type_probs)
# case2 = -tf.reduce_sum(action_type_probs * tf.math.log(action_type_probs))
# print(case1)
# print(case2)

#
# action = tf.random.categorical(x1, 1, dtype=tf.int32)
# a = tf.range(action.shape[0])
# print(a.shape)
# print(tf.squeeze(action).shape)
# indices = tf.stack([tf.range(action.shape[0], dtype=tf.int32), action], axis=1)
# print(indices)
# action_probs = softmax(x1)
# print(action_probs)
# print(action)
# # # action_t = action_probs[:, a]
# action_t = tf.gather_nd(action_probs, indices)
# print(action_t)
#
# action = tf.random.categorical(action_type_probs, 1)
# action_probs = action_type_probs[0, action]
# print(action)
# print(action_type_probs)
# # print(action_type_probs[tf.squeeze(action_type)])
# # print(tf.squeeze(action_type))
# print(action_probs)

# x2 = torch.tensor(x)
# action_logit = torch.distributions.Categorical(x2)
# # action = action_logit.sample()
# # action_prob = action_logit.log_prob(action.squeeze(-1)).view(action.size(0), -1).sum(-1).unsqueeze(-1)
# # print(action)
# # print(action_prob)
# # print(action_logit.probs)
# print(action_logit.probs)
# res = action_logit.entropy()
# print(res)