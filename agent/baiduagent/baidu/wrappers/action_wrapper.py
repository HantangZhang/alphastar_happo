import math
import numpy as np
from baiduagent.baidu.rl_utils import sort_uavs, get_aircraft_and_uavs, get_aircraft, get_closest_enemy_plane
from mat.envs.battle5v5.env_cmd import CmdEnv

class ActionWrapper(object):
    def __init__(self):

        # plane type(1->aircraft, 2->uav): value
        self.min_speed = {1: 150, 2: 100} 
        self.max_speed = {1: 400, 2: 300} 
        #self.max_accmag = {1: 9.8 * 1.0, 2: 9.8 * 2.0} # obs
        self.max_accmag = {1: 1.0, 2: 2.0} # action !!! 
        self.max_cmd_g = {1: 6, 2: 12}
        self.min_z = {1: 2000, 2: 2000} 
        self.max_z = {1: 15000, 2: 10000} 

        self.target_r = 10000 # 10km
        self.adjust_height = 500

        # self.target_r = 400 * 3 # 10km
        # self.adjust_height = 20 * 3

    def get_action_new(self, model_action, last_obs):
        for key in last_obs.keys():
            for item in last_obs[key]:
                if "Alt" not in item.keys():
                    continue
                item["Z"] = item["Alt"]
        model_action = model_action[0]
        name_list = ["有人机", "无人机1", "无人机2", "无人机3", "无人机4" ]
        Steering_angle = [-1, 0, 1] #左转,直行,右转
        adjust_height_angle = [-1, 0, 1] #下降,平飞,上升
        action_len = len(Steering_angle) * len(adjust_height_angle)  #动作的长度由引导角度*调整角度共同决定
        is_launch = [False, True]
        cmd_list = []
        weapon_ind_list = []

        enemy_aircraft, enemy_uavs = get_aircraft_and_uavs(last_obs['trackinfos'])
        enemy_all_planes = enemy_aircraft + enemy_uavs  

        for ind_self, self_plane_info in enumerate(last_obs['platforminfos']):

            assert name_list[ind_self] in self_plane_info["Name"]
            self_is_live = self_plane_info['ID'] != 0 and self_plane_info['Availability'] > 0.0001

            # self feature
            if self_is_live:
                self_action_np = model_action[ind_self]
                # action_ind_max = np.argmax(self_action_np)
                action_ind_max = self_action_np[0]

                #显示每个飞机的初始参数

                # print("model_action", model_action)
                # print("self_action_np", self_action_np)
                # print("action_ind_max", action_ind_max)
                # print("Steering_angle[action_ind_max%3]", Steering_angle[action_ind_max%3])
                # print("(action_ind_max%action_len) // 3", (action_ind_max%action_len) // 3)
                # print("is_launch[is_launch[0 if action_ind_max < action_len else 1]]", is_launch[0 if action_ind_max < action_len else 1])

                plane_type = self_plane_info['Type']
                theta = Steering_angle[action_ind_max % 3] * 2 / 18 * math.pi  # (-pi, pi)
                x = self.target_r * math.sin(self_plane_info["Heading"] + theta)
                y = self.target_r * math.cos(self_plane_info["Heading"] + theta)

                z = self.adjust_height * adjust_height_angle[(action_ind_max % action_len) // 3]

                speed = self.max_speed[plane_type]
                acc = self.max_accmag[plane_type]
                g = self.max_cmd_g[plane_type]
                # print('action={},theta={},x={},y={},z={}'.format(action_ind_max,theta,x,y,z))
                route_list = [{"X": self_plane_info['X'] + x, "Y": self_plane_info['Y'] + y, "Z":
                    np.clip(self_plane_info['Z'] + z, self.min_z[plane_type], self.max_z[plane_type])
                               }, ]
                cmd_list.append(CmdEnv.make_linepatrolparam(self_plane_info['ID'], route_list, speed, acc, g))
                # print("xzy cmd", CmdEnv.make_linepatrolparam(self_plane_info['ID'], route_list, speed, acc, g))
                if is_launch[0 if action_ind_max < action_len else 1]:
                    # launch a missile to attack closest enemy plane
                    target_enemy, distance = get_closest_enemy_plane(self_plane_info, enemy_all_planes)
                    if target_enemy:
                        # weapon_ind_list.append(len(cmd_list))
                        # if distance <= 80000: #当最近的目标离己方飞机的间距小于80000时
                        # CmdEnv.make_attackparam 指令，可以将令飞机攻击指定实体，参数依次为 实体ID，目标ID ，最大射程的百分比（0-1，达到这个距离比就会打击）
                        cmd_list.append(CmdEnv.make_attackparam(self_plane_info['ID'], target_enemy['ID'], 1))
                        # print("battle cmd", CmdEnv.make_attackparam(self_plane_info['ID'], target_enemy['ID'], 1))

        
        return cmd_list

    def get_action(self, model_action, last_obs):
        self_aircraft, self_uavs = get_aircraft_and_uavs(last_obs['platforminfos'])
        sort_uavs(self_uavs) # NOTE


        enemy_aircraft, enemy_uavs = get_aircraft_and_uavs(last_obs['trackinfos'])
        enemy_all_planes = enemy_aircraft + enemy_uavs

        cmd_list = []
        offset = 0  
        for plane_info in self_aircraft + self_uavs:
            plane_type = plane_info['Type']
            theta = model_action[offset + 0] * math.pi # (-pi, pi)
            x = self.target_r * math.sin(theta)
            y = self.target_r * math.cos(theta)

            z = self.adjust_height * model_action[offset + 1]

            speed = self._map_action(model_action[offset + 2], self.min_speed[plane_type], self.max_speed[plane_type])
            acc = self._map_action(model_action[offset + 3], 0, self.max_accmag[plane_type])
            g = self._map_action(model_action[offset + 4], 0, self.max_cmd_g[plane_type])

            route_list = [{"X": plane_info['X'] + x, "Y": plane_info['Y'] + y, "Z": 
                np.clip(plane_info['Z'] + z, self.min_z[plane_type], self.max_z[plane_type])
                }, ]
            cmd_list.append(CmdEnv.make_linepatrolparam(plane_info['ID'], route_list, speed, acc, g))

            if model_action[offset + 5] > 0:
                # launch a missile to attack closest enemy plane
                target_enemy, distance = get_closest_enemy_plane(plane_info, enemy_all_planes)
                if target_enemy:
                    #if distance <= 80000: #当最近的目标离己方飞机的间距小于80000时
                    # CmdEnv.make_attackparam 指令，可以将令飞机攻击指定实体，参数依次为 实体ID，目标ID ，最大射程的百分比（0-1，达到这个距离比就会打击）
                    cmd_list.append(CmdEnv.make_attackparam(plane_info['ID'], target_enemy['ID'], 1))

            offset += 6 # 6 dims for each plane
        
        return cmd_list

    def reset(self):
        pass
    
    def _map_action(self, rate, min_val, max_val):
        assert -1 - 1e-6 <= rate <= 1 + 1e-6, rate
        return min_val + ((max_val - min_val) / 2.0) * (rate + 1.0)


