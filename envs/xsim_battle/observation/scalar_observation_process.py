""" A one line summary of the module or program
Copyright：©2011-2022 北京华如科技股份有限公司
This module provide configure file management service in i18n environment.
Authors: zhanghantang
DateTime:  2022/10/27 17:28
"""

[agent_statistics, home_race, away_race, upgrades,

 enemy_upgrades,
 time, available_actions,
 unit_counts_bow,

         mmr, units_buildings,
 effects,
 upgrade,
 beginning_build_order, last_delay, last_action_type,
         last_repeat_queued] = scalar_list

agent_statistics
# 红方剩余武器数量
red_left_missile = 0
# 蓝方剩余武器数量
blue_left_missile = 0
# 允许发射的最大武器数量
missile_cap = 0
# 当前正在飞行的导弹数量
missile_flying= 0
# 红方剩余飞机数
red_jet_count = 0
# 蓝方剩余飞机数
blue_jet_count = 0
# 距离

time
# 当前时间步长

available_actions
# 当前可执行的1级动作
available_actions = [[0, 1, 0, 1, 1], [1, 1, 0 ,1, 1]]

last_action_type
last_target
last_location






