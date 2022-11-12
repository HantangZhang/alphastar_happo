from utils.utils_math import TSVector3

def sort_uavs(uavs):
    uavs.sort(key=lambda x: (x['Y'], x['X'])) # uav is homogeneous

def get_aircraft_and_uavs(planes_list):
    aircraft = []
    uavs = []
    for plane_info in planes_list:
        if plane_info['ID'] != 0 and plane_info['Availability'] > 0.0001: 
            plane_info["Z"] = plane_info["Alt"]     # 飞机的 Alt 即为飞机的当前高度
            if plane_info['Type'] == 1:           # 所有类型为 1 的飞机是 有人机
                aircraft.append(plane_info) # 将有人机保存下来 一般情况，每方有人机只有1架
            if plane_info['Type'] == 2:           # 所有类型为 2 的飞机是 无人机
                uavs.append(plane_info)  # 将无人机保存下来 一般情况，每方无人机只有4架
    return aircraft, uavs

def get_aircraft(planes_list):
    aircraft = []
    for plane_info in planes_list:
        if plane_info['ID'] != 0 and plane_info['Availability'] > 0.0001: 
            plane_info["Z"] = plane_info["Alt"]     # 飞机的 Alt 即为飞机的当前高度
            if plane_info['Type'] == 1:           # 所有类型为 1 的飞机是 有人机
                aircraft.append(plane_info) # 将有人机保存下来 一般情况，每方有人机只有1架
    return aircraft

def get_closest_enemy_plane(plane, enemy_all_planes):
    min_idx = -1
    min_distance = float('inf')
    for i, enemy_plane in enumerate(enemy_all_planes):
        if enemy_plane['ID'] != 0 and enemy_plane['Availability'] > 0.0001: 
            # print("self_plane")
            # print(plane)
            # print("enemy_plane")
            # print(enemy_plane)
            distance = TSVector3.distance(plane, enemy_plane)
            if distance < min_distance:
                min_distance = distance
                min_idx = i

    if min_idx != -1:
        return enemy_all_planes[min_idx], min_distance
    
    return None, min_distance
