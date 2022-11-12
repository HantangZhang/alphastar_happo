import gym
import numpy as np
import copy
import math
from collections import defaultdict
from utils.utils_math import TSVector3
from baiduagent.baidu.rl_utils import sort_uavs, get_aircraft_and_uavs, get_closest_enemy_plane

class FeaturesExtractorV2(object):
    def __init__(self, max_timesteps=20 * 60):
        # reset counter variables
        self.launched_missiles_set = set([])
        self.plane_launch_missiles_cnt = defaultdict(int)
        self.steps_cnt = 0

        self.max_timesteps = max_timesteps

        self.uavs_num = 4 # 4 uavs
        self.planes_num = 5 # 1 manned + 4 uavs

        self.xy_norm = 150 * 1000.0 # 150 km 
        self.z_norm = 15000.0 # 15 km 
        self.distance_norm = math.sqrt((self.xy_norm * 2) ** 2.0 + (self.xy_norm * 2) ** 2.0 + (self.z_norm * 2) ** 2.0) # 1000: Z range
        
        # plane type(1->aircraft, 2->uav, 3->missile): value
        self.min_speed = {1: 150, 2: 100} 
        self.max_speed = {1: 400, 2: 300, 3: 1000} 
        self.max_accmag = {1: 9.8 * 1.0, 2: 9.8 * 2.0} # obs
        self.max_cmd_g = {1: 6, 2: 12}

        self.missiles_max_feat_num = 2 # for each plane, only considering 2 closest missiles

        self.missile_padding_feats = [0.0, 1.0, 0.0, 1.0, 1.0, 1.0]
        self._default_closest_enemy_feats = [1.0, 1.0, 1.0, 1.0, 0.0]
        self._default_relative_feats = [1.0, 1.0, 1.0, 1.0]

    def get_features(self, obs):
        
        # count launched missiles of planes
        for missile_info in obs['missileinfos']:
            if missile_info['ID'] not in self.launched_missiles_set:
                self.launched_missiles_set.add(missile_info['ID'])
                self.plane_launch_missiles_cnt[missile_info['LauncherID']] += 1
        features = []
        # reset count variables
        self.self_left_weapons_sum = 0
        self.enemy_left_weapons_sum = 0

        time_feat = (self.max_timesteps - self.steps_cnt) / self.max_timesteps
        features.append(time_feat)
        
        # self information
        self_aircraft, self_uavs = get_aircraft_and_uavs(obs['platforminfos'])

        sort_uavs(self_uavs)

        # self aircraft infos
        if len(self_aircraft) == 1:
            plane_info = self_aircraft[0]
            features.extend(self._extract_self_plane_feats(obs, plane_info))
        elif len(self_aircraft) == 0:
            features.extend(self._get_self_plane_padding_feats())
        else:
            assert False
        
        # self uavs infos
        for plane_info in self_uavs:
            features.extend(self._extract_self_plane_feats(obs, plane_info))
        features.extend(self._get_self_plane_padding_feats() * (self.uavs_num - len(self_uavs)))
        
        
        # enemy information
        enemy_aircraft, enemy_uavs = get_aircraft_and_uavs(obs['trackinfos']) 

        sort_uavs(enemy_uavs)

        # enemy aircraft
        if len(enemy_aircraft) == 1:
            plane_info = enemy_aircraft[0]
            features.extend(self._extract_enemy_plane_feats(obs, plane_info))
        elif len(enemy_aircraft) == 0:
            features.extend(self._get_enemy_plane_padding_feats())
        else:
            assert False
        
        # enemy uavs
        for plane_info in enemy_uavs:
            features.extend(self._extract_enemy_plane_feats(obs, plane_info))
        features.extend(self._get_enemy_plane_padding_feats() * (self.uavs_num - len(enemy_uavs)))
        

        self_all_planes = self_aircraft + self_uavs
        enemy_all_planes = enemy_aircraft + enemy_uavs
        if len(enemy_all_planes) == 0:
            features.extend(self._default_closest_enemy_feats * self.planes_num)
        else:
            for plane in self_all_planes:
                features.extend(self._get_closest_enemy_feats(plane, enemy_all_planes))

            features.extend(self._default_closest_enemy_feats * (self.planes_num - len(self_all_planes)))
        
        # added in v2
        for self_plane in self_all_planes:
            for enemy_plane in enemy_all_planes:
                features.extend(self._get_relative_feats(self_plane, enemy_plane))

            features.extend(self._default_relative_feats * (self.planes_num - len(enemy_all_planes)))
        features.extend(self._default_relative_feats * ((self.planes_num - len(self_all_planes)) * self.planes_num))

        
        # added in v2
        assert 0 <=self.self_left_weapons_sum <= 12 # 4 + 2 * 4
        assert 0 <=self.enemy_left_weapons_sum <= 12 # 4 + 2 * 4
        features.append(self.self_left_weapons_sum / 12.0)
        features.append(self.enemy_left_weapons_sum / 12.0)

        return np.array(features, dtype='float32')

    def _get_avail(self, plane_info, closest_enemy_dis):
        plane_is_live = plane_info['ID'] != 0 and plane_info['Availability'] > 0.0001
        if plane_is_live:
            avail = [1] * 18
            if plane_info["LeftWeapon"] == 0 or closest_enemy_dis > 80000:
                avail = [1]*9 + [0]*9
        else:
            avail = [0]*18
            avail[4] = 1

        return np.array(avail)

    def get_features_new(self, obs):
        all_features = []
        ava = []
        # print(obs)
        # count launched missiles of planes
        for missile_info in obs['missileinfos']:
            if missile_info['ID'] not in self.launched_missiles_set:
                self.launched_missiles_set.add(missile_info['ID'])
                self.plane_launch_missiles_cnt[missile_info['LauncherID']] += 1
                # print("self.plane_launch_missiles_cnt[missile_info['LauncherID']]")
                # print(self.plane_launch_missiles_cnt[missile_info['LauncherID']])
        
        time_feat = (self.max_timesteps - self.steps_cnt) / self.max_timesteps
        # print("time_feat", time_feat)
        # # self information
        # self_aircraft, self_uavs = get_aircraft_and_uavs(obs['platforminfos'])
        # sort_uavs(self_uavs)
        
        # print("obs")
        # print(obs)

        # enemy information
        enemy_aircraft, enemy_uavs = get_aircraft_and_uavs(obs['trackinfos']) 
        enemy_all_planes = enemy_aircraft + enemy_uavs

        name_list = ["有人机", "无人机1", "无人机2", "无人机3", "无人机4" ]
        # print(obs['platforminfos'])
        for ind_self, self_plane_info in enumerate(obs['platforminfos']):
            # print(self_plane_info)
            assert name_list[ind_self] in self_plane_info["Name"]
            self_is_live = self_plane_info['ID'] != 0 and self_plane_info['Availability'] > 0.0001
            if self_is_live:
                _, closest_enemy_dis = get_closest_enemy_plane(self_plane_info, enemy_all_planes)
                # _, closest_enemy_dis = get_closest_enemy_plane(self_plane_info, enemy_aircraft)
            else:
                closest_enemy_dis = float('inf')

            # reset count variables
            self.self_left_weapons_sum = 0
            self.enemy_left_weapons_sum = 0
            
            features = []
            features.append(time_feat)
            # self feature
            if self_is_live:
                features.extend(self._extract_self_plane_feats(obs, self_plane_info))
            else:
                features.extend(self._get_self_plane_padding_feats())
            

            for ind_enem, enemy_plane_info in enumerate(obs['trackinfos']):
                assert name_list[ind_enem] in enemy_plane_info["Name"]
                enemy_is_live = enemy_plane_info['ID'] != 0 and enemy_plane_info['Availability'] > 0.0001
                # entity feature
                if enemy_is_live:
                    features.extend(self._extract_enemy_plane_feats(obs, enemy_plane_info))
                else:
                    features.extend(self._get_enemy_plane_padding_feats())

                # # closest 0 or 1
                # if enemy_is_live and self_is_live:
                #     features.extend(self._get_closest_enemy_feats(self_plane_info, [enemy_plane_info]))
                # else:
                #     features.extend(self._default_closest_enemy_feats)

                # relative distance
                if enemy_is_live and self_is_live:
                    features.extend(self._get_relative_feats(self_plane_info, enemy_plane_info))

                else:
                    features.extend(self._default_relative_feats)

            
            # added in v2
            assert 0 <=self.self_left_weapons_sum <= 12 # 4 + 2 * 4
            assert 0 <=self.enemy_left_weapons_sum <= 12 # 4 + 2 * 4
            features.append(self.self_left_weapons_sum / 12.0)
            features.append(self.enemy_left_weapons_sum / 12.0)

            all_features.append(np.array(features, dtype='float32'))
            ava.append(self._get_avail(self_plane_info, closest_enemy_dis))
            
        return all_features, ava
    
    def _get_closest_enemy_feats(self, plane, enemy_all_planes):
        min_idx = 0
        min_distance = float('inf')
        for i, enemy_plane in enumerate(enemy_all_planes):
            distance = TSVector3.distance(plane, enemy_plane)
            if distance < min_distance:
                min_distance = distance
                min_idx = i

        closest_enemy = enemy_all_planes[min_idx]

        feats = []
        
        feats.append(min_distance / self.distance_norm)
        feats.append((closest_enemy['X'] - plane['X']) / self.xy_norm)
        feats.append((closest_enemy['Y'] - plane['Y']) / self.xy_norm)
        feats.append((closest_enemy['Z'] - plane['Z']) / self.z_norm)
        
        # added in v2
        feats.append(int(closest_enemy['LeftWeapon'] > 0))
    
        return feats

    def _get_relative_feats(self, self_plane, enemy_plane):

        feats = []
        
        distance = TSVector3.distance(self_plane, enemy_plane)
        # print(distance)
        # print(self_plane)
        # print(enemy_plane)
        feats.append(distance / self.distance_norm)
        feats.append((enemy_plane['X'] - self_plane['X']) / self.xy_norm)
        feats.append((enemy_plane['Y'] - self_plane['Y']) / self.xy_norm)
        feats.append((enemy_plane['Z'] - self_plane['Z']) / self.z_norm)
    
        return feats


    def _get_enemy_plane_padding_feats(self):
        feats = []

        #feats.append(plane_info['Availability'])
        feats.append(0)
        #feats.append(plane_info['X'] / self.xy_norm)
        feats.append(-2.0)
        #feats.append(plane_info['Y'] / self.xy_norm)
        feats.append(-2.0)
        #feats.append(plane_info['Z'] / self.z_norm)
        feats.append(0.0)
        #feats.append(plane_info['Heading'])
        feats.append(0.0)
        #feats.append(plane_info['Pitch'])
        feats.append(0.0)
        #feats.append(plane_info['Roll'])
        feats.append(0.0)
        #feats.append((plane_info['Speed'] - self.min_speed[plane_type]) / (self.max_speed[plane_type] - self.min_speed[plane_type]))
        feats.append(0.0)

        #feats.append(plane_info['AccMag'] / self.max_accmag[plane_type])
        #feats.append(0.0)
        #feats.append(plane_info['NormalG'] / self.max_cmd_g[plane_type])
        #feats.append(0.0)

        #feats.append(plane_info['LeftWeapon'])
        feats.append(0.0)
        #feats.append(int(plane_info['LeftWeapon'] > 0))
        feats.append(0.0)
        #feats.append(int(plane_info['IsLocked']))
        feats.append(0.0)

        feats.extend(self.missile_padding_feats * self.missiles_max_feat_num)

        return feats


    def _get_self_plane_padding_feats(self):
        feats = []

        #feats.append(plane_info['Availability'])
        feats.append(0)
        #feats.append(plane_info['X'] / self.xy_norm)
        feats.append(-2.0)
        #feats.append(plane_info['Y'] / self.xy_norm)
        feats.append(-2.0)
        #feats.append(plane_info['Z'] / self.z_norm)
        feats.append(0.0)
        #feats.append(plane_info['Heading'])
        feats.append(0.0)
        #feats.append(plane_info['Pitch'])
        feats.append(0.0)
        #feats.append(plane_info['Roll'])
        feats.append(0.0)
        #feats.append((plane_info['Speed'] - self.min_speed[plane_type]) / (self.max_speed[plane_type] - self.min_speed[plane_type]))
        feats.append(0.0)
        #feats.append(plane_info['AccMag'])
        feats.append(0.0)
        #feats.append(plane_info['NormalG'] / self.max_cmd_g[plane_type])
        feats.append(0.0)
        #feats.append(plane_info['LeftWeapon'])
        feats.append(0.0)
        #feats.append(int(plane_info['LeftWeapon'] > 0))
        feats.append(0.0)
        #feats.append(int(plane_info['IsLocked']))
        feats.append(0.0)


        feats.extend(self.missile_padding_feats * self.missiles_max_feat_num)

        return feats

    def _extract_self_plane_feats(self, obs, plane_info):
        # print(plane_info)
        feats = []
        plane_type = plane_info['Type']

        feats.append(plane_info['Availability'])
        feats.append(plane_info['X'] / self.xy_norm)
        feats.append(plane_info['Y'] / self.xy_norm)
        feats.append(plane_info['Z'] / self.z_norm)
        feats.append(plane_info['Heading'])
        feats.append(plane_info['Pitch'])
        feats.append(plane_info['Roll'])
        feats.append((plane_info['Speed'] - self.min_speed[plane_type]) / (self.max_speed[plane_type] - self.min_speed[plane_type]))
        feats.append(plane_info['AccMag'] / self.max_accmag[plane_type])
        feats.append(plane_info['NormalG'] / self.max_cmd_g[plane_type])
        feats.append(plane_info['LeftWeapon'])
        feats.append(int(plane_info['LeftWeapon'] > 0))
        feats.append(int(plane_info['IsLocked']))

        feats.extend(self._get_missile_feats(obs, plane_info))

        self.self_left_weapons_sum += plane_info['LeftWeapon']

        return feats

    def _extract_enemy_plane_feats(self, obs, plane_info):
        # print(plane_info)
        feats = []
        plane_type = plane_info['Type']

        feats.append(plane_info['Availability'])
        feats.append(plane_info['X'] / self.xy_norm)
        feats.append(plane_info['Y'] / self.xy_norm)
        feats.append(plane_info['Z'] / self.z_norm)
        feats.append(plane_info['Heading'])
        feats.append(plane_info['Pitch'])
        feats.append(plane_info['Roll'])
        feats.append((plane_info['Speed'] - self.min_speed[plane_type]) / (self.max_speed[plane_type] - self.min_speed[plane_type]))
        #feats.append(plane_info['AccMag'] / self.max_accmag[plane_type])
        #feats.append(plane_info['NormalG'] / self.max_cmd_g[plane_type])

        max_weapon = 4 if plane_type == 1 else 2
        left_weapon = max_weapon - self.plane_launch_missiles_cnt[plane_info['ID']]
        assert left_weapon >= 0, "plane_type: {} max_weapon: {} left_weapon: {}".format(plane_type, max_weapon, left_weapon)
        feats.append(left_weapon)
        feats.append(int(left_weapon > 0))

        feats.append(int(plane_info['IsLocked']))

        feats.extend(self._get_missile_feats(obs, plane_info))

        plane_info['LeftWeapon'] = left_weapon
        self.enemy_left_weapons_sum += left_weapon
        # self.enemy_left_weapons_sum += 0

        return feats


    def _get_missile_feats(self, obs, plane_info):
        missiles = []
        for missile_info in obs['missileinfos']:
            if missile_info['ID'] != 0 and missile_info['Availability'] > 0.0001: # available missile
                if missile_info['EngageTargetID'] == plane_info['ID']:
                    missile_info['Z'] = missile_info['Alt']
                    distance = TSVector3.distance(missile_info, plane_info)
                    attack_missile = copy.deepcopy(missile_info)
                    attack_missile['distance'] = distance
                    missiles.append(attack_missile)

        missiles.sort(key=lambda x: x['distance'])
        num = min(len(missiles), self.missiles_max_feat_num)
        feats = []
        missile_type = 3
        for i in range(num):
            missile_info = missiles[i]
            feats.append(missile_info['Availability']) 
            feats.append(missile_info['distance'] / self.distance_norm) 
            feats.append(missile_info['Speed'] / self.max_speed[missile_type]) 
            feats.append((missile_info['X'] - plane_info['X']) / self.xy_norm)
            feats.append((missile_info['Y'] - plane_info['Y']) / self.xy_norm)
            feats.append((missile_info['Z'] - plane_info['Z']) / self.z_norm)

        feats = feats + self.missile_padding_feats * (self.missiles_max_feat_num - num)

        return feats
    
    
    def get_obs(self, obs):
        self.steps_cnt += 1

        features = self.get_features_new(obs)
        return features
    
    def reset(self):
        # reset counter variables
        self.launched_missiles_set = set([])
        self.plane_launch_missiles_cnt = defaultdict(int)
        self.steps_cnt = 0
