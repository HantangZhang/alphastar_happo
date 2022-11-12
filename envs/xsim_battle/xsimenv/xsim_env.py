""" A one line summary of the module or program
Copyright：©2011-2022 北京华如科技股份有限公司
This module provide configure file management service in i18n environment.
Authors: zhanghantang
DateTime:  2022/10/27 15:47
"""
from envs.xsim_battle.xsimenv.xsim_manager import XSimManager
from envs.xsim_battle.xsimenv.communication_service import CommunicationService

class XSimEnv(object):
    """
        仿真环境类
        对于用户来说，如果想要与xsim环境连接，只需要实例化一个XSimEnv实例即可
        - 通过 step(action)->obs 将任务指令发送至xsim引擎，然后推进引擎，同时，引擎返回执行action后的observation
        - 通过 reset()重置xsim环境
        - 通过 close()关闭xsim环境
    @Examples:
        添加使用示例
        # 创建xsim环境
        xsim_env = XSimEnv()
        # 推进环境
        obs = xsim_env.step(action)
        # 重置环境
        env_reset_state = xsim_env.reset()
        # 关闭环境
        env_close_state = xsim_env.close()
    @Author：wubinxing
    """

    def __init__(self, time_ratio: int, address: str, image_name='xsim:v1.0', mode: str = 'host', local_test=False):
        """
        初始化函数
        @param domain_id: 服务域名
        @author:wubinxing
        @create data:2021/05/10 15.00
        @change date:
        """
        self.local_test = local_test
        if local_test == False:
            # xsim引擎控制器
            self.xsim_manager = XSimManager(time_ratio, address, image_name, mode)
            # 与xsim引擎交互通信服务
            self.communication_service = CommunicationService(self.xsim_manager.address)

    def __del__(self):
        if self.local_test == False:
            self.xsim_manager.close_env()

    def step(self, action: list) -> dict:
        """
        用户与xsim环境交互核心函数。通过step控制引擎的推进。
        @param action: 用户要执行的任务指令列表，任务指令可以通过EnvCmd任务指令组包辅助类进行辅助组包。
        @return: xsim在执行完毕action后当前的态势信息
        @author:wubinxing
        @create data:2021/05/10 15.00
        @change date:
        """
        try:
            if self.local_test == True:
                raw_obs = {'blue': {'platforminfos': [{'Name': '蓝有人机', 'Identification': '蓝方', 'ID': 6, 'Type': 1, 'Availability': 1.0, 'X': 103804.79204641584, 'Y': -6317.694582573702, 'Lon': 2.023967634528888, 'Lat': 0.2607737876164057, 'Alt': 9341.289153143764, 'Heading': -1.728289625507187, 'Pitch': -0.004014324449108708, 'Roll': 0.0, 'Speed': 299.99999999999983, 'CurTime': 135.0, 'AccMag': 0.0, 'NormalG': 0.0, 'IsLocked': False, 'Status': 32719, 'LeftWeapon': 4}, {'Name': '蓝无人机1', 'Identification': '蓝方', 'ID': 14, 'Type': 2, 'Availability': 1.0, 'X': 95110.30765678418, 'Y': 10133.566573454584, 'Lon': 2.0225678455787337, 'Lat': 0.2633578021129436, 'Alt': 9328.62517170515, 'Heading': -1.4361696467369145, 'Pitch': -0.0043330915887404, 'Roll': 0.0, 'Speed': 299.99999999999915, 'CurTime': 135.0, 'AccMag': 0.0, 'NormalG': 0.0, 'IsLocked': False, 'Status': 32719, 'LeftWeapon': 2}, {'Name': '蓝无人机2', 'Identification': '蓝方', 'ID': 15, 'Type': 2, 'Availability': 1.0, 'X': 85181.91957894305, 'Y': -10618.764450461675, 'Lon': 2.0209440807860504, 'Lat': 0.26011127408666596, 'Alt': 9312.52674551867, 'Heading': -1.7101829017949808, 'Pitch': -0.004740927450187415, 'Roll': 0.0, 'Speed': 300.0, 'CurTime': 135.0, 'AccMag': 0.0, 'NormalG': 0.0, 'IsLocked': False, 'Status': 32719, 'LeftWeapon': 2}, {'Name': '蓝无人机3', 'Identification': '蓝方', 'ID': 16, 'Type': 2, 'Availability': 1.0, 'X': 85182.14487080467, 'Y': 10616.291056055412, 'Lon': 2.0209564424743096, 'Lat': 0.26343935501807353, 'Alt': 9312.527544723824, 'Heading': -1.4242186703822, 'Pitch': -0.004740898122349362, 'Roll': 0.0, 'Speed': 300.0, 'CurTime': 135.0, 'AccMag': 0.0, 'NormalG': 0.0, 'IsLocked': False, 'Status': 32719, 'LeftWeapon': 2}, {'Name': '蓝无人机4', 'Identification': '蓝方', 'ID': 17, 'Type': 2, 'Availability': 1.0, 'X': 95110.15777021706, 'Y': -10135.814947996962, 'Lon': 2.0225546855258045, 'Lat': 0.2601810803355522, 'Alt': 9328.624511473812, 'Heading': -1.6974056109852216, 'Pitch': -0.004333108148264193, 'Roll': 0.0, 'Speed': 299.99999999999915, 'CurTime': 135.0, 'AccMag': 0.0, 'NormalG': 0.0, 'IsLocked': False, 'Status': 32719, 'LeftWeapon': 2}], 'trackinfos': [{'Name': '红有人机', 'Identification': '红方', 'ID': 1, 'Type': 1, 'Availability': 1.0, 'X': -98563.15052217754, 'Y': -10064.363341005464, 'Lon': 1.9911436548629613, 'Lat': 0.26019020092051537, 'Alt': 9804.414961034432, 'Heading': 1.9221423265274198, 'Pitch': -0.21662617145620414, 'Roll': 0.0, 'Speed': 360.20726499999967, 'CurTime': 135.0, 'IsLocked': False}, {'Name': '红无人机1', 'Identification': '红方', 'ID': 2, 'Type': 2, 'Availability': 1.0, 'X': -103436.53442845897, 'Y': 83797.9906167541, 'Lon': 1.990283974996565, 'Lat': 0.27489793764881587, 'Alt': 9249.51067796629, 'Heading': 1.8157827983696802, 'Pitch': 3.435733278926532e-06, 'Roll': 0.0, 'Speed': 217.6519699999999, 'CurTime': 135.0, 'IsLocked': False}, {'Name': '红无人机2', 'Identification': '红方', 'ID': 11, 'Type': 2, 'Availability': 1.0, 'X': -105191.69002630748, 'Y': 42086.52493770076, 'Lon': 1.990024410743685, 'Lat': 0.26836095940137644, 'Alt': 7432.184429466724, 'Heading': 1.8157831979886694, 'Pitch': 3.6790013518361598e-06, 'Roll': 0.0, 'Speed': 235.30393999999959, 'CurTime': 135.0, 'IsLocked': False}, {'Name': '红无人机3', 'Identification': '红方', 'ID': 12, 'Type': 2, 'Availability': 1.0, 'X': -104640.57619048604, 'Y': -34433.342826732194, 'Lon': 1.9901734263734174, 'Lat': 0.25636664595739433, 'Alt': 9189.496847754344, 'Heading': 1.815777438003948, 'Pitch': 4.674865803032765e-06, 'Roll': 0.0, 'Speed': 296.0773399999999, 'CurTime': 135.0, 'IsLocked': False}, {'Name': '红无人机4', 'Identification': '红方', 'ID': 13, 'Type': 2, 'Availability': 1.0, 'X': -106248.58564998471, 'Y': -69659.39690713931, 'Lon': 1.9899395998622706, 'Lat': 0.2508457830893316, 'Alt': 9980.011746574193, 'Heading': 1.8157244540440143, 'Pitch': -2.982340376764517e-06, 'Roll': 0.0, 'Speed': 217.65197000000043, 'CurTime': 135.0, 'IsLocked': False}], 'missileinfos': []}, 'red': {'platforminfos': [{'Name': '红有人机', 'Identification': '红方', 'ID': 1, 'Type': 1, 'Availability': 1.0, 'X': -98563.15052217754, 'Y': -10064.363341005464, 'Lon': 1.9911436548629613, 'Lat': 0.26019020092051537, 'Alt': 9804.414961034432, 'Heading': 1.9221423265274198, 'Pitch': -0.21662617145620414, 'Roll': 0.0, 'Speed': 360.20726499999967, 'CurTime': 135.0, 'AccMag': 9.806649999999998, 'NormalG': 0.0, 'IsLocked': False, 'Status': 32719, 'LeftWeapon': 4, 'Z': 9804.414961034432}, {'Name': '红无人机1', 'Identification': '红方', 'ID': 2, 'Type': 2, 'Availability': 1.0, 'X': -103436.53442845897, 'Y': 83797.9906167541, 'Lon': 1.990283974996565, 'Lat': 0.27489793764881587, 'Alt': 9249.51067796629, 'Heading': 1.8157827983696802, 'Pitch': 3.435733278926532e-06, 'Roll': 0.0, 'Speed': 217.6519699999999, 'CurTime': 135.0, 'AccMag': 19.6133, 'NormalG': 0.0, 'IsLocked': False, 'Status': 32719, 'LeftWeapon': 2, 'Z': 9249.51067796629}, {'Name': '红无人机2', 'Identification': '红方', 'ID': 11, 'Type': 2, 'Availability': 1.0, 'X': -105191.69002630748, 'Y': 42086.52493770076, 'Lon': 1.990024410743685, 'Lat': 0.26836095940137644, 'Alt': 7432.184429466724, 'Heading': 1.8157831979886694, 'Pitch': 3.6790013518361598e-06, 'Roll': 0.0, 'Speed': 235.30393999999959, 'CurTime': 135.0, 'AccMag': 19.6133, 'NormalG': 0.0, 'IsLocked': False, 'Status': 32719, 'LeftWeapon': 2, 'Z': 7432.184429466724}, {'Name': '红无人机3', 'Identification': '红方', 'ID': 12, 'Type': 2, 'Availability': 1.0, 'X': -104640.57619048604, 'Y': -34433.342826732194, 'Lon': 1.9901734263734174, 'Lat': 0.25636664595739433, 'Alt': 9189.496847754344, 'Heading': 1.815777438003948, 'Pitch': 4.674865803032765e-06, 'Roll': 0.0, 'Speed': 296.0773399999999, 'CurTime': 135.0, 'AccMag': 19.6133, 'NormalG': 0.0, 'IsLocked': False, 'Status': 32719, 'LeftWeapon': 2, 'Z': 9189.496847754344}, {'Name': '红无人机4', 'Identification': '红方', 'ID': 13, 'Type': 2, 'Availability': 1.0, 'X': -106248.58564998471, 'Y': -69659.39690713931, 'Lon': 1.9899395998622706, 'Lat': 0.2508457830893316, 'Alt': 9980.011746574193, 'Heading': 1.8157244540440143, 'Pitch': -2.982340376764517e-06, 'Roll': 0.0, 'Speed': 217.65197000000043, 'CurTime': 135.0, 'AccMag': 19.613300000000002, 'NormalG': 0.0, 'IsLocked': False, 'Status': 32719, 'LeftWeapon': 2, 'Z': 9980.011746574193}], 'trackinfos': [{'Name': '蓝有人机', 'Identification': '蓝方', 'ID': 6, 'Type': 1, 'Availability': 1.0, 'X': 103804.79204641584, 'Y': -6317.694582573702, 'Lon': 2.023967634528888, 'Lat': 0.2607737876164057, 'Alt': 9341.289153143764, 'Heading': -1.728289625507187, 'Pitch': -0.004014324449108708, 'Roll': 0.0, 'Speed': 299.99999999999983, 'CurTime': 135.0, 'IsLocked': False, 'Z': 9341.289153143764, 'LeftWeapon': 4}, {'Name': '蓝无人机1', 'Identification': '蓝方', 'ID': 14, 'Type': 2, 'Availability': 1.0, 'X': 95110.30765678418, 'Y': 10133.566573454584, 'Lon': 2.0225678455787337, 'Lat': 0.2633578021129436, 'Alt': 9328.62517170515, 'Heading': -1.4361696467369145, 'Pitch': -0.0043330915887404, 'Roll': 0.0, 'Speed': 299.99999999999915, 'CurTime': 135.0, 'IsLocked': False, 'Z': 9328.62517170515, 'LeftWeapon': 2}, {'Name': '蓝无人机2', 'Identification': '蓝方', 'ID': 15, 'Type': 2, 'Availability': 1.0, 'X': 85181.91957894305, 'Y': -10618.764450461675, 'Lon': 2.0209440807860504, 'Lat': 0.26011127408666596, 'Alt': 9312.52674551867, 'Heading': -1.7101829017949808, 'Pitch': -0.004740927450187415, 'Roll': 0.0, 'Speed': 300.0, 'CurTime': 135.0, 'IsLocked': False, 'Z': 9312.52674551867, 'LeftWeapon': 2}, {'Name': '蓝无人机3', 'Identification': '蓝方', 'ID': 16, 'Type': 2, 'Availability': 1.0, 'X': 85182.14487080467, 'Y': 10616.291056055412, 'Lon': 2.0209564424743096, 'Lat': 0.26343935501807353, 'Alt': 9312.527544723824, 'Heading': -1.4242186703822, 'Pitch': -0.004740898122349362, 'Roll': 0.0, 'Speed': 300.0, 'CurTime': 135.0, 'IsLocked': False, 'Z': 9312.527544723824, 'LeftWeapon': 2}, {'Name': '蓝无人机4', 'Identification': '蓝方', 'ID': 17, 'Type': 2, 'Availability': 1.0, 'X': 95110.15777021706, 'Y': -10135.814947996962, 'Lon': 2.0225546855258045, 'Lat': 0.2601810803355522, 'Alt': 9328.624511473812, 'Heading': -1.6974056109852216, 'Pitch': -0.004333108148264193, 'Roll': 0.0, 'Speed': 299.99999999999915, 'CurTime': 135.0, 'IsLocked': False, 'Z': 9328.624511473812, 'LeftWeapon': 2}], 'missileinfos': []}, 'sim_time': 135.0, 'xsim_tag': 60000}

                return raw_obs
            obs = self.communication_service.step(action)
            #print("引擎推进一步!!!!")
            return obs
        except Exception as e:
            print(e)
        # return self.communication_service.step(action)

    def reset(self):
        """
        重置训练环境
        @return: 环境重置状态：成功或者失败
        @author:wubinxing
        @create data:2021/05/10 15.00
        @change date:
        """
        # return self.communication_service.reset()

        # Xsim战场环境重置
        if self.local_test == False:


            self.communication_service.reset()
            obs = self.communication_service.step([])

            while obs["sim_time"] > 10:
                obs = self.step([])

    def end(self):
        """
        重置训练环境
        @return: 环境重置状态：成功或者失败
        @author:wubinxing
        @create data:2021/05/10 15.00
        @change date:
        """
        return self.communication_service.end()

    def close(self) -> bool:
        """
        关闭训练环境
        @return: 环境关闭状态：成功或者失败
        @author:wubinxing
        @create data:2021/05/10 15.00
        @change date:
        """
        self.xsim_manager.close_env()
        self.communication_service.close()
        return True

    def close_env(self):
        self.xsim_manager.close_env()

    def restar_env(self):
        self.xsim_manager.restar_env()