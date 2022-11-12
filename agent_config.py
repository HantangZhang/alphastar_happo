"""
@FileName：config.py
@Description：
@Author：wubinxing
@Time：2021/5/9 下午8:08
@Department：AIStudio研发部
@Copyright：©2011-2021 北京华如科技股份有限公司
"""
# from agent.red_agent import RebAgent
# from agent.blue_agent import BlueAgent
# from agent.baiduagent.blue_agent import BlueAgent as huaru_BlueAgent
#from agent.Yi_swarm.Yi_swarm import Yi_swarm
# from baiduagent.baidu import RedRLAgent as baidu_RedRLAgent
# from baiduagent.baidu import BlueRLAgent as baidu_BlueRLAgent
#from agent.niuniu.niuniu_v2 import MyAgent as BlueAgent
#from agent.Cm5.Cm5 import Cm5 as Cm5_Blue
# 是否启用host模式,host仅支持单个xsim
# from agent.AerialGhost.AerialGhost import AerialGhostAgent
#from agent.DominoLab.agent.DominoLab.DL_agent import DLAgent as DominoLab

from agent.maybe.Maybe_Agent import MaybeAgent
from agent.yiteam.Yi_team import Yi_team
ISHOST = True

# 为态势显示工具域分组ID  1-1000
HostID = 7

IMAGE = "xsim:v8.1"

# 加速比 1-100
TimeRatio = 80

# 范围:0-100 生成的回放个数 (RTMNum + 2),后续的回放会把之前的覆盖掉.
RTMNum = 100

config = {
    "episode_time": 100, # 训练次数
    "step_time": 1, # 想定步长
    'agents': {
            'red': MaybeAgent,
            'blue': Yi_team
              }
}

# 进程数量
POOL_NUM = 10

# 启动XSIM的数量
XSIM_NUM = 6


ADDRESS = {
    "ip": "192.168.70.205",
    "port": 24
}
