import torch
from envs.xsim_battle.macro_action.action_dict import ACTIONS_STAT
from envs.xsim_battle.agent_base import AgentState
from configs.hyper_parameters import Action_Space_Parameters as AHP
from configs.hyper_parameters import State_Space_Parameters as SSP
# 可选的目标是10个，但是entity的目标是20个
TARGET_UNITS_TYPES_MASK = torch.zeros(AHP.level1_action_dim, SSP.eneity_num)
LOCATION_MASK = torch.zeros(AHP.level1_action_dim, AHP.location_dim)
for action in range(AHP.level1_action_dim):
    action_stat = ACTIONS_STAT.get(action, None)
    location_list = action_stat['location_id']
    LOCATION_MASK[action, location_list] = 1
# 暂时写成这样，到时候直接从hp中调
action_size = 5
dead_jet_list = []

def generate_target_mask(level1_actions):
    mask = torch.zeros(level1_actions.shape[0], SSP.eneity_num)
    for i, action in enumerate(level1_actions):
        action = int(action)
        action_stat = ACTIONS_STAT.get(action, None)
        target_list = action_stat['target_id']
        for jet_id in target_list:
            if jet_id in AgentState.dead_jet_list:
                target_list.remove(jet_id)
        action_stat['target_id'] = target_list
        TARGET_UNITS_TYPES_MASK[action, target_list] = 1
        mask[i] = TARGET_UNITS_TYPES_MASK[action]
    return mask.bool()


# def generate_target_mask(level1_action):
#     # 生成mask之前，应该先更新这个lavel1 action的action_stat，然后再更新TARGET_UNITS_TYPES_MASK，最后直接取mask就行
#     # 因为要取mask之前一定要调整，不取也就可以不用调整，所以直接让调整的函数返回mask就行
#     level1_action = int(level1_action)
#     action_stat  = ACTIONS_STAT.get(level1_action, None)
#     target_list = adjust_action_stat(action_stat)
#     mask = adjust_target_mask(level1_action, target_list)
#     mask = mask.bool()
#     return mask
#
# # def test(level1_action):
#
# def adjust_action_stat(action_stat):
#     # 应该在每次决策前只更新当前选择的这个动作的target和location就行，其他动作因为用不到更新了徒增计算量
#     # location好像没有需要调整的地方
#     target_list = action_stat['target_id']
#     for jet_id in target_list:
#         if jet_id in AgentState.dead_jet_list:
#             target_list.remove(jet_id)
#     action_stat['target_id'] = target_list
#     return target_list
#
# def adjust_target_mask(level1_action, target_list):
#     TARGET_UNITS_TYPES_MASK[level1_action, target_list] = 1
#     return TARGET_UNITS_TYPES_MASK[level1_action]


def generate_location_mask(level1_actions):
    mask = torch.zeros(level1_actions.shape[0], AHP.location_dim)
    for i, action in enumerate(level1_actions):
        mask[i] = LOCATION_MASK[int(action)]
    return mask.bool()





if __name__ == '__main__':
    # level1_action = 2
    # AgentState.dead_jet_list = [1, 8]
    # mask = generate_target_mask(level1_action)
    # print(mask)
    #
    # print(LOCATION_MASK)
    level1_action = torch.zeros((10, 1))
    level1_action[4] = 3
    mask = generate_target_mask(level1_action)
    print(mask.shape)
    mask2 = generate_location_mask(level1_action)
    print(mask2.shape)