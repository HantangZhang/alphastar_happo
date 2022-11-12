import time
import wandb
import numpy as np
from functools import reduce
import torch
import os
from baiduagent.baidu.wrappers.action_wrapper import ActionWrapper
from baiduagent.baidu.wrappers.mirror_wrapper import MirrorObsAction
from baiduagent.baidu.wrappers.features_extractor_v2 import FeaturesExtractorV2
import torch
import os
import pickle
import copy

CUR_PATH = os.path.dirname(os.path.realpath('__file__'))
# import sys
# sys.path.insert(0, "/home/work/multi_rl_seq/mat")

def _t2n(x):
    return x.detach().cpu().numpy()


class BaiduAgent(object):
    """seq2seq mat agent for predict."""
    def __init__(self, name, side):
        super(BaiduAgent, self).__init__()

        # with open("/home/ds/data/ma_transformer_0911_v2/baiduagent/baidu/model_config.pickle", "rb") as config_pkl:
        #     config = pickle.load(config_pkl)   #加载配置文件



        #self.all_args = config["all_args"]
        #self.algorithm_name = self.all_args.algorithm_name
        self.algorithm_name = 'mat'
        self.num_agents = 5#config["num_agents"]
        self.recurrent_N = 1#self.all_args.recurrent_N
        self.hidden_size = 64#self.all_args.hidden_size

        # # cuda
        # if self.all_args.cuda and torch.cuda.is_available():
        #     print("choose to use gpu...")
        #     device = torch.device("cuda:0")
        # else:
        #     print("choose to use cpu...")
        #     device = torch.device("cpu")
        


        self.side = side["side"]
        self.eval_rnn_states = np.zeros((1, self.num_agents, self.recurrent_N,
                                    self.hidden_size), dtype=np.float32)
        self.eval_masks = np.ones((1, self.num_agents, 1), dtype=np.float32)
        self.pre_actions = []
        self.accu_repe_num = 0

        # env wrappers 
        self.features_extractor = FeaturesExtractorV2()
        self.mirror_obs_action = MirrorObsAction() #当Agent属于蓝方时,将obs修改成相反的
        self.action_wrapper = ActionWrapper()
        self.mirror_obs_action.reset(self.side) #当agent为蓝方时,将动作修改为相反的

    @torch.no_grad()
    def step(self, cur_time, obs, is_repe=True, repe_num=3):
        """predict actions."""
        # print("obs")
        # print(obs)

        def get_z(sub_obs): #把Alt改成z便于之后的action_wrapper
            for key in sub_obs.keys():
                for item in sub_obs[key]:
                    if "Alt" not in item.keys():
                        continue
                    item["Z"] = item["Alt"]
        get_z(obs)

        mirror_obs = self.mirror_obs_action.get_obs(obs)
        train_obs, train_share_obs, ava = self._get_features(mirror_obs)

        return train_obs, train_share_obs, ava

    '''
        # self.accu_repe_num = cur_time - 1
        if self.accu_repe_num % repe_num == 0:
            # print("exe")
            # print(obs['platforminfos'][0]['CurTime'])
            # print(self.accu_repe_num)
            # print(obs)

            mirror_obs = self.mirror_obs_action.get_obs(obs)
            # # print("mirror_obs")
            # # print(mirror_obs)
            train_obs, train_share_obs, ava = self._get_features(mirror_obs)
            # eval_obs, eval_share_obs, ava = self._get_features(obs)

            

                
            eval_actions, eval_rnn_states = self.policy.act(np.concatenate(eval_share_obs),np.concatenate(eval_obs),
                                        np.concatenate(self.eval_rnn_states),
                                        np.concatenate(self.eval_masks),
                                        np.concatenate(ava),
                                        deterministic=True)
                                        
            # print("eval_actions", eval_actions)
            eval_actions = np.array(np.split(_t2n(eval_actions), 1))
            self.eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), 1))
            # print("eval_actions", eval_actions)
            # print("baidu_self_actions_index", eval_actions)
            # env_cmd = self.action_wrapper.get_action_new(eval_actions[0], obs)
            env_cmd = self.action_wrapper.get_action_new(eval_actions[0], mirror_obs)
            # print("env", env_cmd)
            env_cmd = self.mirror_obs_action.get_action(env_cmd) #下达指令后,如果是红方则不需变动指令,如果是蓝方则需要转化成镜像指令
            self.pre_actions = copy.deepcopy(env_cmd)

            self.accu_repe_num += 1
            return env_cmd
            
            
        else:
            # print(" no exe")
            # print(obs['platforminfos'][0]['CurTime'])
            # print(self.accu_repe_num)
        # if is_repe == True and self.accu_repe_num % repe_num != 0:
            self.accu_repe_num += 1
            for sub_self_action in self.pre_actions:
                if "CmdAttackControl" in sub_self_action.keys():
                    # print("self.pre_actions", self.pre_actions)
                    # print("env_cmd", env_cmd)
                    self.pre_actions.remove(sub_self_action)
                    # print("self.pre_actions", self.pre_actions)
                    # print("env_cmd", env_cmd)

            return self.pre_actions
            '''

    def reset(self):

        self.accu_repe_num = 0
        self.eval_rnn_states = np.zeros((1, self.num_agents, self.recurrent_N,
                                    self.hidden_size), dtype=np.float32)
        self.pre_actions = []
        self.eval_masks = np.ones((1, self.num_agents, 1), dtype=np.float32)

        self.features_extractor.reset()
        self.mirror_obs_action.reset(self.side)
        self.action_wrapper.reset()

    def _get_features(self, raw_obs):
        # 把死亡的战斗机信息补全
        name_list = ["有人机", "无人机1", "无人机2", "无人机3", "无人机4"]
        for key in raw_obs.keys():
            if key == "missileinfos" or (key != "missileinfos" and len(raw_obs[key]) == 5):
                continue
            tmp = ["", "", "", "", ""]
            cur_fighter_set = set([])
            for item in raw_obs[key]:
                cur_fighter_set.add(item["Name"][1:])
                ind_name = name_list.index(item["Name"][1:])
                tmp[ind_name] = item
            for dead_name in list(set(name_list) - cur_fighter_set):
                # print(set(name_list) - cur_fighter_set)
                ind_name = name_list.index(dead_name)
                # print(dead_name)
                # print(ind_name)
                tmp[ind_name] = {"Name":dead_name, "ID":"dead", "Availability":0}
            raw_obs[key] = tmp
            # if len(set(name_list) - cur_fighter_set) > 1:
            #     print(raw_obs[key])

        # print(obs)
        features, ava = self.features_extractor.get_obs(raw_obs)
        # for feature in features:
        #     print(feature.shape)

        # scale, offset = self.scaler.get()
        # scale[0] = 1.0  # don't scale time step feature
        # offset[0] = 0.0  # don't offset time step feature

        # features = features.reshape((1, -1))
        # scaled_features = [(fea - offset) * scale  for fea in features] # center and scale observations
        state = features.copy()
        return features, state, ava

    def load(self, model_dir):
        """load policy's networks from a saved model."""
        self.policy.restore(model_dir)