""" A one line summary of the module or program
Copyright：©2011-2022 北京华如科技股份有限公司
This module provide configure file management service in i18n environment.
Authors: zhanghantang
DateTime:  2022/10/27 17:59
"""
from collections import namedtuple

'''
维护宏动作常量参数

max_attack_distance：智能体最大开火距离
attack_angle_lower：智能体开火角度上界
attack_angle_upper：智能体开火角度下界
'''


MacroActionAgentParameters = namedtuple('MacroAgentParameters', ['max_attack_distance',
                                                                 'attack_angle_lower',
                                                                 'attack_angle_upper',
                                                                 ])


Macro_Action_Agent_Parameters = MacroActionAgentParameters(max_attack_distance=40e3,
                                                           attack_angle_lower=30,
                                                           attack_angle_upper=-30
                                                           )


print(Macro_Action_Agent_Parameters.max_attack_range)