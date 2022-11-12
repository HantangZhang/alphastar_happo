# import torch
# import numpy as np
# from algorithm.model.alpha.mask_func import generate_location_mask
# from envs.xsim_battle.macro_action.action_dict import ACTIONS_STAT

# mask = generate_location_mask((1))
# print(mask)
#
# level_action = torch.ones((3, 1))
# # mask = generate_location_mask(level_action)
# # print(mask)
# a = level_action.squeeze(-1)
# print(a.shape)
# for i in level_action:
#     mask = generate_location_mask(i)
#     print(mask)
import torch
import torch.nn as nn

class model(nn.Module):

    def __init__(self, device):
        super(model, self).__init__()

        self.linear = nn.Linear(10, 11)
        self.to(device)

    def forward(self, x):
        y = self.linear(x)

        return x


x = torch.zeros((2, 10))
print(x.device)
device = 'cuda:0'
mo = model(device)
x = x.to('cuda:0')
print(next(mo.parameters()).device, 11111)
y = mo(x)
print(y)


