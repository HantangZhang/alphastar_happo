#-*-coding:utf-8-*-
"""
@FileName：cmd_env.py
@Description：
@Author：qiwenhao
@Time：2021/5/24 19:22
@Department：AIStudio研发部
@Copyright：©2011-2021 北京华如科技股份有限公司
@Project：601
"""
from typing import List
ACCMAG = 1.0

class CmdEnv(object):
    """
        指令字典组包类
        用户通过此类进行指令数据的组包
    @Examples:
        添加使用示例
    @Author：qiwenhao
    """

    def entity_info(self, id) -> dict:
        from envs.xsim_battle.xsimenv.observation_processor import _OBSINIT
        _OBSINIT = {'blue': {'platforminfos': [{'Name': '蓝有人机', 'Identification': '蓝方', 'ID': 6, 'Type': 1, 'Availability': 1.0, 'X': 103804.79204641584, 'Y': -6317.694582573702, 'Lon': 2.023967634528888, 'Lat': 0.2607737876164057, 'Alt': 9341.289153143764, 'Heading': -1.728289625507187, 'Pitch': -0.004014324449108708, 'Roll': 0.0, 'Speed': 299.99999999999983, 'CurTime': 135.0, 'AccMag': 0.0, 'NormalG': 0.0, 'IsLocked': False, 'Status': 32719, 'LeftWeapon': 4}, {'Name': '蓝无人机1', 'Identification': '蓝方', 'ID': 14, 'Type': 2, 'Availability': 1.0, 'X': 95110.30765678418, 'Y': 10133.566573454584, 'Lon': 2.0225678455787337, 'Lat': 0.2633578021129436, 'Alt': 9328.62517170515, 'Heading': -1.4361696467369145, 'Pitch': -0.0043330915887404, 'Roll': 0.0, 'Speed': 299.99999999999915, 'CurTime': 135.0, 'AccMag': 0.0, 'NormalG': 0.0, 'IsLocked': False, 'Status': 32719, 'LeftWeapon': 2}, {'Name': '蓝无人机2', 'Identification': '蓝方', 'ID': 15, 'Type': 2, 'Availability': 1.0, 'X': 85181.91957894305, 'Y': -10618.764450461675, 'Lon': 2.0209440807860504, 'Lat': 0.26011127408666596, 'Alt': 9312.52674551867, 'Heading': -1.7101829017949808, 'Pitch': -0.004740927450187415, 'Roll': 0.0, 'Speed': 300.0, 'CurTime': 135.0, 'AccMag': 0.0, 'NormalG': 0.0, 'IsLocked': False, 'Status': 32719, 'LeftWeapon': 2}, {'Name': '蓝无人机3', 'Identification': '蓝方', 'ID': 16, 'Type': 2, 'Availability': 1.0, 'X': 85182.14487080467, 'Y': 10616.291056055412, 'Lon': 2.0209564424743096, 'Lat': 0.26343935501807353, 'Alt': 9312.527544723824, 'Heading': -1.4242186703822, 'Pitch': -0.004740898122349362, 'Roll': 0.0, 'Speed': 300.0, 'CurTime': 135.0, 'AccMag': 0.0, 'NormalG': 0.0, 'IsLocked': False, 'Status': 32719, 'LeftWeapon': 2}, {'Name': '蓝无人机4', 'Identification': '蓝方', 'ID': 17, 'Type': 2, 'Availability': 1.0, 'X': 95110.15777021706, 'Y': -10135.814947996962, 'Lon': 2.0225546855258045, 'Lat': 0.2601810803355522, 'Alt': 9328.624511473812, 'Heading': -1.6974056109852216, 'Pitch': -0.004333108148264193, 'Roll': 0.0, 'Speed': 299.99999999999915, 'CurTime': 135.0, 'AccMag': 0.0, 'NormalG': 0.0, 'IsLocked': False, 'Status': 32719, 'LeftWeapon': 2}], 'trackinfos': [{'Name': '红有人机', 'Identification': '红方', 'ID': 1, 'Type': 1, 'Availability': 1.0, 'X': -98563.15052217754, 'Y': -10064.363341005464, 'Lon': 1.9911436548629613, 'Lat': 0.26019020092051537, 'Alt': 9804.414961034432, 'Heading': 1.9221423265274198, 'Pitch': -0.21662617145620414, 'Roll': 0.0, 'Speed': 360.20726499999967, 'CurTime': 135.0, 'IsLocked': False}, {'Name': '红无人机1', 'Identification': '红方', 'ID': 2, 'Type': 2, 'Availability': 1.0, 'X': -103436.53442845897, 'Y': 83797.9906167541, 'Lon': 1.990283974996565, 'Lat': 0.27489793764881587, 'Alt': 9249.51067796629, 'Heading': 1.8157827983696802, 'Pitch': 3.435733278926532e-06, 'Roll': 0.0, 'Speed': 217.6519699999999, 'CurTime': 135.0, 'IsLocked': False}, {'Name': '红无人机2', 'Identification': '红方', 'ID': 11, 'Type': 2, 'Availability': 1.0, 'X': -105191.69002630748, 'Y': 42086.52493770076, 'Lon': 1.990024410743685, 'Lat': 0.26836095940137644, 'Alt': 7432.184429466724, 'Heading': 1.8157831979886694, 'Pitch': 3.6790013518361598e-06, 'Roll': 0.0, 'Speed': 235.30393999999959, 'CurTime': 135.0, 'IsLocked': False}, {'Name': '红无人机3', 'Identification': '红方', 'ID': 12, 'Type': 2, 'Availability': 1.0, 'X': -104640.57619048604, 'Y': -34433.342826732194, 'Lon': 1.9901734263734174, 'Lat': 0.25636664595739433, 'Alt': 9189.496847754344, 'Heading': 1.815777438003948, 'Pitch': 4.674865803032765e-06, 'Roll': 0.0, 'Speed': 296.0773399999999, 'CurTime': 135.0, 'IsLocked': False}, {'Name': '红无人机4', 'Identification': '红方', 'ID': 13, 'Type': 2, 'Availability': 1.0, 'X': -106248.58564998471, 'Y': -69659.39690713931, 'Lon': 1.9899395998622706, 'Lat': 0.2508457830893316, 'Alt': 9980.011746574193, 'Heading': 1.8157244540440143, 'Pitch': -2.982340376764517e-06, 'Roll': 0.0, 'Speed': 217.65197000000043, 'CurTime': 135.0, 'IsLocked': False}], 'missileinfos': []}, 'red': {'platforminfos': [{'Name': '红有人机', 'Identification': '红方', 'ID': 1, 'Type': 1, 'Availability': 1.0, 'X': -98563.15052217754, 'Y': -10064.363341005464, 'Lon': 1.9911436548629613, 'Lat': 0.26019020092051537, 'Alt': 9804.414961034432, 'Heading': 1.9221423265274198, 'Pitch': -0.21662617145620414, 'Roll': 0.0, 'Speed': 360.20726499999967, 'CurTime': 135.0, 'AccMag': 9.806649999999998, 'NormalG': 0.0, 'IsLocked': False, 'Status': 32719, 'LeftWeapon': 4, 'Z': 9804.414961034432}, {'Name': '红无人机1', 'Identification': '红方', 'ID': 2, 'Type': 2, 'Availability': 1.0, 'X': -103436.53442845897, 'Y': 83797.9906167541, 'Lon': 1.990283974996565, 'Lat': 0.27489793764881587, 'Alt': 9249.51067796629, 'Heading': 1.8157827983696802, 'Pitch': 3.435733278926532e-06, 'Roll': 0.0, 'Speed': 217.6519699999999, 'CurTime': 135.0, 'AccMag': 19.6133, 'NormalG': 0.0, 'IsLocked': False, 'Status': 32719, 'LeftWeapon': 2, 'Z': 9249.51067796629}, {'Name': '红无人机2', 'Identification': '红方', 'ID': 11, 'Type': 2, 'Availability': 1.0, 'X': -105191.69002630748, 'Y': 42086.52493770076, 'Lon': 1.990024410743685, 'Lat': 0.26836095940137644, 'Alt': 7432.184429466724, 'Heading': 1.8157831979886694, 'Pitch': 3.6790013518361598e-06, 'Roll': 0.0, 'Speed': 235.30393999999959, 'CurTime': 135.0, 'AccMag': 19.6133, 'NormalG': 0.0, 'IsLocked': False, 'Status': 32719, 'LeftWeapon': 2, 'Z': 7432.184429466724}, {'Name': '红无人机3', 'Identification': '红方', 'ID': 12, 'Type': 2, 'Availability': 1.0, 'X': -104640.57619048604, 'Y': -34433.342826732194, 'Lon': 1.9901734263734174, 'Lat': 0.25636664595739433, 'Alt': 9189.496847754344, 'Heading': 1.815777438003948, 'Pitch': 4.674865803032765e-06, 'Roll': 0.0, 'Speed': 296.0773399999999, 'CurTime': 135.0, 'AccMag': 19.6133, 'NormalG': 0.0, 'IsLocked': False, 'Status': 32719, 'LeftWeapon': 2, 'Z': 9189.496847754344}, {'Name': '红无人机4', 'Identification': '红方', 'ID': 13, 'Type': 2, 'Availability': 1.0, 'X': -106248.58564998471, 'Y': -69659.39690713931, 'Lon': 1.9899395998622706, 'Lat': 0.2508457830893316, 'Alt': 9980.011746574193, 'Heading': 1.8157244540440143, 'Pitch': -2.982340376764517e-06, 'Roll': 0.0, 'Speed': 217.65197000000043, 'CurTime': 135.0, 'AccMag': 19.613300000000002, 'NormalG': 0.0, 'IsLocked': False, 'Status': 32719, 'LeftWeapon': 2, 'Z': 9980.011746574193}], 'trackinfos': [{'Name': '蓝有人机', 'Identification': '蓝方', 'ID': 6, 'Type': 1, 'Availability': 1.0, 'X': 103804.79204641584, 'Y': -6317.694582573702, 'Lon': 2.023967634528888, 'Lat': 0.2607737876164057, 'Alt': 9341.289153143764, 'Heading': -1.728289625507187, 'Pitch': -0.004014324449108708, 'Roll': 0.0, 'Speed': 299.99999999999983, 'CurTime': 135.0, 'IsLocked': False, 'Z': 9341.289153143764, 'LeftWeapon': 4}, {'Name': '蓝无人机1', 'Identification': '蓝方', 'ID': 14, 'Type': 2, 'Availability': 1.0, 'X': 95110.30765678418, 'Y': 10133.566573454584, 'Lon': 2.0225678455787337, 'Lat': 0.2633578021129436, 'Alt': 9328.62517170515, 'Heading': -1.4361696467369145, 'Pitch': -0.0043330915887404, 'Roll': 0.0, 'Speed': 299.99999999999915, 'CurTime': 135.0, 'IsLocked': False, 'Z': 9328.62517170515, 'LeftWeapon': 2}, {'Name': '蓝无人机2', 'Identification': '蓝方', 'ID': 15, 'Type': 2, 'Availability': 1.0, 'X': 85181.91957894305, 'Y': -10618.764450461675, 'Lon': 2.0209440807860504, 'Lat': 0.26011127408666596, 'Alt': 9312.52674551867, 'Heading': -1.7101829017949808, 'Pitch': -0.004740927450187415, 'Roll': 0.0, 'Speed': 300.0, 'CurTime': 135.0, 'IsLocked': False, 'Z': 9312.52674551867, 'LeftWeapon': 2}, {'Name': '蓝无人机3', 'Identification': '蓝方', 'ID': 16, 'Type': 2, 'Availability': 1.0, 'X': 85182.14487080467, 'Y': 10616.291056055412, 'Lon': 2.0209564424743096, 'Lat': 0.26343935501807353, 'Alt': 9312.527544723824, 'Heading': -1.4242186703822, 'Pitch': -0.004740898122349362, 'Roll': 0.0, 'Speed': 300.0, 'CurTime': 135.0, 'IsLocked': False, 'Z': 9312.527544723824, 'LeftWeapon': 2}, {'Name': '蓝无人机4', 'Identification': '蓝方', 'ID': 17, 'Type': 2, 'Availability': 1.0, 'X': 95110.15777021706, 'Y': -10135.814947996962, 'Lon': 2.0225546855258045, 'Lat': 0.2601810803355522, 'Alt': 9328.624511473812, 'Heading': -1.6974056109852216, 'Pitch': -0.004333108148264193, 'Roll': 0.0, 'Speed': 299.99999999999915, 'CurTime': 135.0, 'IsLocked': False, 'Z': 9328.624511473812, 'LeftWeapon': 2}], 'missileinfos': []}, 'sim_time': 135.0, 'xsim_tag': 60000}

        if _OBSINIT is None:
            raise ServerError("无法正确获取战场初始信息，请检查是否正确连接XSIM.")
        for Identification, info in _OBSINIT.items():
            for platforminfo in info["platforminfos"]:
                if platforminfo["ID"] == id:
                    name = "有人机" if platforminfo["Type"] == 1 else "无人机"
                    return dict(Identification=Identification, entity_name=name)

        raise ModuleNotFoundError("ID:{}, Module中未查询到此实体ID.".format(id))

    @staticmethod
    def make_entityinitinfo(receiver: int,
                            x: float,
                            y: float,
                            z: float,
                            init_speed: float,
                            init_heading: float
                            ) -> dict:
        """
        初始化实体指令组包函数
        :param receiver: Handle ID
        :param x: 初始位置为战场x坐标
        :param y: 初始位置为战场y坐标
        :param z: 初始位置为战场z坐标
        :param init_speed: 初始速度(单位：米/秒，有人机取值范围：[150,400]，无人机取值范围：[100,300])
        :param init_heading: 初始朝向(单位：度，取值范围[0,360]，与正北方向的夹角)
        :return: 完成组包后的数据，格式为dict：Key为主题名称，Value为数据
        """

        owner_info = CmdEnv().entity_info(receiver)

        if owner_info["Identification"] == "red":
            assert x >= -150000 and x <= -125000, "X超出红方初始战场位置，红方战场初始位置范围为[-125000, -150000]."
            assert y >= -150000 and y <= 150000

        if owner_info["Identification"] == "blue":
            assert x >= 125000 and x <= 150000, "X超出蓝方初始战场位置，红方战场初始位置范围为[125000, 150000]."
            assert y >= -150000 and y <= 150000

        if owner_info["entity_name"] == "有人机":
            assert init_speed <= 400 and init_speed >= 150, "有人机初始速度的取值范围：[150, 400], 单位：米/秒."
            assert z >= 9000 and z <= 10000, "Z坐标值超过限制高度,有人机高度限制[9000,10000]."

        if owner_info["entity_name"] == "无人机":
            assert init_speed <= 300 and init_speed >= 100, "无人机初始速度的取值范围：[100,300]，单位：米/秒."
            assert z >= 9000 and z <= 10000, "Z坐标值超过限制高度，无人机高度限制[9000,10000]."

        assert init_heading <= 360 and init_heading >= 0, "初始朝向的取值范围[0,360]，与正北方向的夹角，单位：度"

        action = dict(
            HandleID=receiver,
            Receiver=receiver,
            InitPos=dict(X=x, Y=y, Z=z),
            InitSpeed=init_speed,
            InitHeading=init_heading
        )

        return dict(CmdInitEntityControl=action)

    @staticmethod
    def make_linepatrolparam(receiver: int,
                             coord_list: List[dict],
                             cmd_speed: float,
                             cmd_accmag: float,
                             cmd_g: float) -> dict:
        '''
        航线巡逻控制指令
        :param receiver: Handle ID
        :param coord_list: 路径点坐标列表 -> [{"x": 500, "y": 400, "z": 2000}, {"x": 600, "y": 500, "z": 3000}]
                           区域x，y不得超过作战区域,有人机高度限制[2000,15000]，无人机高度限制[2000,10000]
        :param cmd_speed: 指令速度
        :param cmd_accmag: 指令加速度
        :param cmd_g: 指令过载
        :return: 完成组包后的数据，格式为dict：Key为主题名称，Value为数据
        '''

        owner_info = CmdEnv().entity_info(receiver)

        if owner_info["entity_name"] == "有人机":
            assert cmd_speed >= 150 and cmd_speed <= 400, "指令速度,有人机取值范围：[150, 400]."
            assert cmd_accmag >= 0 and cmd_accmag <= ACCMAG, "指令加速度,有人机取值范围：[0, ACCMAG]."
            assert cmd_g >= 0 and cmd_g <= 6, "指令过载,有人机取值范围：[0, 6]."

        if owner_info["entity_name"] == "无人机":
            assert cmd_speed >= 100 and cmd_speed <= 300, "指令速度,无人机取值范围：[100,300]."
            assert cmd_accmag >= 0 and cmd_accmag <= 2 * ACCMAG, "指令加速度,无人机取值范围：[0, 2 * ACCMAG]."
            assert cmd_g >= 0 and cmd_g <= 12, "指令过载，无人机取值范围：[0,12]."

        action = {}
        action["HandleID"] = receiver
        action["Receiver"] = receiver
        for coord in coord_list:
            assert coord.get("X", -1) != -1, \
                "KeyError： 未找到坐标x的值，请检查coord_list参数，参数示例：coord_list=[{'X': 500, 'Y': 400, 'Z': 2000}, {'X': 600, 'Y': 500, 'Z': 3000}]"
            assert coord.get("Y", -1) != -1, \
                "KeyError： 未找到坐标y的值，请检查coord_list参数，参数示例：coord_list=[{'X': 500, 'Y': 400, 'Z': 2000}, {'X': 600, 'Y': 500, 'Z': 3000}]"
            assert coord.get("Z", -1) != -1, \
                "KeyError： 未找到坐标z的值，请检查coord_list参数，参数示例：coord_list=[{'X': 500, 'Y': 400, 'Z': 2000}, {'X': 600, 'Y': 500, 'Z': 3000}]"

            if owner_info["entity_name"] == "有人机":
                assert coord.get("Z") >= 2000 and coord.get("Z") <= 15000, "Z坐标值超过限制高度,有人机高度限制[2000,15000]."

            if owner_info["entity_name"] == "无人机":
                assert coord.get("Z") >= 2000 and coord.get("Z") <= 10000, "Z坐标值超过限制高度，无人机高度限制[2000,10000]."

        action["CoordList"] = coord_list

        action["CmdSpeed"] = cmd_speed
        action["CmdAccMag"] = cmd_accmag
        action["CmdG"] = cmd_g

        return dict(CmdLinePatrolControl=action)

    @staticmethod
    def make_areapatrolparam(receiver: int,
                             x: float,
                             y: float,
                             z: float,
                             area_length: float,
                             area_width: float,
                             cmd_speed: float,
                             cmd_accmag: float,
                             cmd_g: float
                             ) -> dict:
        '''
        区域巡逻控制指令
        :param receiver: Handle ID
        :param x: 区域中心坐标x坐标
        :param y: 区域中心坐标y坐标
        :param z: 区域中心坐标z坐标
        :param area_length: 区域长
        :param area_width: 区域宽
        :param cmd_speed: 指令速度
        :param cmd_accmag: 指令加速度
        :param cmd_g: 指令过载
        :return: 完成组包后的数据，格式为dict：Key为主题名称，Value为数据
        '''

        owner_info = CmdEnv().entity_info(receiver)

        if owner_info["entity_name"] == "有人机":
            assert z >= 2000 and z <= 15000, "Z坐标值超过限制高度,有人机高度限制[2000,15000]."
            assert cmd_speed >= 150 and cmd_speed <= 400, "指令速度,有人机取值范围：[150, 400]."
            assert cmd_accmag >= 0 and cmd_accmag <= ACCMAG, "指令加速度,有人机取值范围：[0, ACCMAG]."
            assert cmd_g >= 0 and cmd_g <= 6, "指令过载,有人机取值范围：[0, 6]."

        if owner_info["entity_name"] == "无人机":
            assert z >= 2000 and z <= 10000, "Z坐标值超过限制高度，无人机高度限制[2000,10000]."
            assert cmd_speed >= 100 and cmd_speed <= 300, "指令速度,无人机取值范围：[100,300]."
            assert cmd_accmag >= 0 and cmd_accmag <= 2 * ACCMAG, "指令加速度,无人机取值范围：[0, 2 * ACCMAG]."
            assert cmd_g >= 0 and cmd_g <= 12, "指令过载，无人机取值范围：[0,12]."

        area_length_limit = abs(area_length) / 2.0 + abs(x)
        area_width_limit = abs(area_width) / 2.0 + abs(y)
        assert area_length_limit <= 150000 and area_width_limit <= 150000, "区域长度或者宽度超过限制，经度方向边长，abs(CenterCoord.x +- AreaLength/2) <= 150000，纬度方向边长，abs(CenterCoord.y +- AreaWidth/2) <= 150000"

        action = dict(
            HandleID=receiver,
            Receiver=receiver,
            CenterCoord=dict(X=x, Y=y, Z=z),
            AreaLength=area_length,
            AreaWidth=area_width,
            CmdSpeed=cmd_speed,
            CmdAccMag=cmd_accmag,
            CmdG=cmd_g
        )

        return dict(CmdAreaPatrolControl=action)

    @staticmethod
    def make_motioncmdparam(receiver: int,
                            update_motiontype: int,
                            cmd_speed: float,
                            cmd_accmag: float,
                            cmd_g: float
                            ) -> dict:
        '''
        机动参数调整控制指令
        :param receiver: Handle ID
        :param update_motiontype: 调整机动参数,可实现组合赋值，例如：CMDSPPED | CMDACCMAG
        :param cmd_speed: 指令速度
        :param cmd_accmag: 指令加速度
        :param cmd_g: 指令过载
        :return: 完成组包后的数据，格式为dict：Key为主题名称，Value为数据
        '''

        owner_info = CmdEnv().entity_info(receiver)

        if owner_info["entity_name"] == "有人机":
            assert cmd_speed >= 150 and cmd_speed <= 400, "指令速度,有人机取值范围：[150, 400]."
            assert cmd_accmag >= 0 and cmd_accmag <= ACCMAG, "指令加速度,有人机取值范围：[0, ACCMAG]."
            assert cmd_g >= 0 and cmd_g <= 6, "指令过载,有人机取值范围：[0, 6]."

        if owner_info["entity_name"] == "无人机":
            assert cmd_speed >= 100 and cmd_speed <= 300, "指令速度,无人机取值范围：[100,300]."
            assert cmd_accmag >= 0 and cmd_accmag <= 2 * ACCMAG, "指令加速度,无人机取值范围：[0, 2 * ACCMAG]."
            assert cmd_g >= 0 and cmd_g <= 12, "指令过载，无人机取值范围：[0,12]."

        action = dict(
            HandleID=receiver,
            Receiver=receiver,
            UpdateMotionType=update_motiontype,
            CmdSpeed=cmd_speed,
            CmdAccMag=cmd_accmag,
            CmdG=cmd_g
        )

        return dict(CmdChangeMotionControl=action)

    @staticmethod
    def make_followparam(receiver: int,
                         tgt_id: int,
                         cmd_speed: float,
                         cmd_accmag: float,
                         cmd_g: float
                         ) -> dict:
        '''
        跟随目标指令
        :param receiver: Handle ID
        :param tgt_id: 目标ID,友方敌方均可
        :param cmd_speed: 指令速度
        :param cmd_accmag: 指令加速度
        :param cmd_g: 指令过载
        :return: 完成组包后的数据，格式为dict：Key为主题名称，Value为数据
        '''

        owner_info = CmdEnv().entity_info(receiver)

        if owner_info["entity_name"] == "有人机":
            assert cmd_speed >= 150 and cmd_speed <= 400, "指令速度,有人机取值范围：[150, 400]."
            assert cmd_accmag >= 0 and cmd_accmag <= ACCMAG, "指令加速度,有人机取值范围：[0, ACCMAG]."
            assert cmd_g >= 0 and cmd_g <= 6, "指令过载,有人机取值范围：[0, 6]."

        if owner_info["entity_name"] == "无人机":
            assert cmd_speed >= 100 and cmd_speed <= 300, "指令速度,无人机取值范围：[100,300]."
            assert cmd_accmag >= 0 and cmd_accmag <= 2 * ACCMAG, "指令加速度,无人机取值范围：[0, 2 * ACCMAG]."
            assert cmd_g >= 0 and cmd_g <= 12, "指令过载，无人机取值范围：[0,12]."

        action = dict(
            HandleID=receiver,
            Receiver=receiver,
            TgtID=tgt_id,
            CmdSpeed=cmd_speed,
            CmdAccMag=cmd_accmag,
            CmdG=cmd_g
        )

        return dict(CmdTargetFollowControl=action)

    @staticmethod
    def make_attackparam(receiver: int,
                         tgt_id: int,
                         fire_range: float
                         ) -> dict:
        '''
        打击目标指令
        :param tgt_id: 目标ID
        :param fire_range: 开火范围，最大探测范围的百分比，取值范围[0, 1]
        :return: 完成组包后的数据，格式为dict：Key为主题名称，Value为数据
        '''
        owner_info = CmdEnv().entity_info(receiver)
        # assert owner_info["entity_name"] == "有人机" or owner_info["entity_name"] == "无人机", "目标ID不正确，无法执行打击指令。"


        assert fire_range >= 0 and fire_range <= 1, "开火范围超出限制，最大探测范围的百分比，取值范围[0, 1]"

        action = dict(
            HandleID=receiver,
            Receiver=receiver,
            TgtID=tgt_id,
            Range=fire_range
        )

        return dict(CmdAttackControl=action)


class ServerError(Exception):
    pass
