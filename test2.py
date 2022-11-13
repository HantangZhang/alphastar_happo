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
import random
dic = {}
for _ in range(1000):
    a = random.randint(1, 28)
    if a not in dic:
        dic[a] = 1
    else:
        dic[a] += 1

print(dic)
