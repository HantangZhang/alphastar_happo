""" A one line summary of the module or program
Copyright：©2011-2022 北京华如科技股份有限公司
This module provide configure file management service in i18n environment.
Authors: zhanghantang
DateTime:  2022/10/27 14:36
"""
import numpy as np
from envs.xsim_battle.utils_math import TSVector3
from envs.xsim_battle.macro_action.macro_config import Macro_Action_Agent_Parameters as MA

# 最大开火距离
## 所有宏动作的信息应该统一写成一个config文件，想想这个config怎么写

# 攻击动作可执行判断函数
def check_attack(my_jet, enemy_jets: list):
    """
    判断当前能否选择攻击这个动作
    条件：
    1. 弹药是否充足
    2. 开火距离是否满足
    3. 开火角度是否满足
    :param my_jet: 当前智能体实例
    :param enemy_jets: 所有敌机列表，列表每个元素是敌机的实例
    :return: bool，True为可以执行
    """

    can_attack_enemy = []
    fire = True

    # 弹药是否充足
    if my_jet.num_avail_ms <= 0:
        fire = False

    # 开火距离是否满足
    for enemy_jet in enemy_jets:
        dis = TSVector3.distance(my_jet.pos3d, enemy_jet.pos3d)
        if dis < MA.max_attack_distance:
            can_attack_enemy.append(enemy_jet)
    if len(can_attack_enemy) == 0:
        fire = False

    # 开火角度是否满足
    for enemy_jet in can_attack_enemy:
        delta = enemy_jet.pos2d - my_jet.pos2d
        theta = 90 - my_jet.Heading * 180 / np.pi
        theta = reg_rad(theta * np.pi / 180)
        delta2 = np.matmul(np.array([[np.cos(theta), np.sin(theta)],
                                     [-np.sin(theta), np.cos(theta)]]), delta.T)
        deg = dir2rad(delta2) * 180 / np.pi

        if deg > ATTACK_RANGE_ANGLE:
            can_attack_enemy.remove(enemy_jet)

# 躲避动作可执行判断函数
def check_evade():
    #

# 逃逸动作可执行判断函数
def check_escape():
    pass

# 掩护动作可执行判断函数
def check_cover():
    pass

# 接进动作可执行判断函数
def check_flank():
    pass

# 制导动作可执行判断函数
def check_guide():
    pass

if __name__ == '__main__':
    # 开火条件判断测试