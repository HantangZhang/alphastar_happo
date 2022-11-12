import copy
import math

class MirrorObsAction(object):
    def __init__(self):
        """
        Mirror obs and action when self.side is blue.
        """
        self.side = None

    def reset(self, side):
        self.side = side 

    def get_obs(self, obs):
        obs = self._map_obs(obs)
        return obs

    def get_action(self, action):
        action = self._map_action(action)
        return action

    def _map_obs(self, obs):
        if self.side == 'red':
            return obs
        elif self.side == 'blue':
            mirror_obs = self._get_mirror_obs(obs)
            return mirror_obs
        else:
            assert False

    def _map_action(self, action):
        if self.side == 'red':
            return action
        elif self.side == 'blue':
            mirror_action = self._get_mirror_action(action)
            return mirror_action
        else:
            assert False
    
    def _get_mirror_action(self, action):
        for cmd in action:
            assert isinstance(cmd, dict)
            assert len(cmd) == 1
            key = list(cmd.keys())[0]
            if key == "CmdLinePatrolControl":
                route_list = cmd[key]['CoordList']
                assert len(route_list) == 1
                route_list[0]['X'] = -route_list[0]['X']
                route_list[0]['Y'] = -route_list[0]['Y']
            elif key == "CmdAttackControl":
                pass
            else:
                assert False

        return action

    def _get_mirror_obs(self, obs):
        mirror_obs = copy.deepcopy(obs)
        # print("mirror_obs", mirror_obs)

        # self information
        for plane in mirror_obs['platforminfos']:
            self._mirror_plane(plane)
        for plane in mirror_obs['trackinfos']:
            self._mirror_plane(plane)
        for missile in mirror_obs['missileinfos']:
            self._mirror_plane(missile)

        return mirror_obs

    def _mirror_plane(self, plane):
        # symmetry based on (0, 0) 
        plane['X'] = -plane['X'] 
        plane['Y'] = -plane['Y'] 
        if 0.0 <= plane['Heading'] <= math.pi:
            plane['Heading'] = -math.pi + plane['Heading']
        elif -math.pi <= plane['Heading'] <= 0.0:
            plane['Heading'] = math.pi + plane['Heading']
        else:
            assert False

