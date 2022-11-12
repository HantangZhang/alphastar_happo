""" A one line summary of the module or program
Copyright：©2011-2022 北京华如科技股份有限公司
This module provide configure file management service in i18n environment.
Authors: zhanghantang
DateTime:  2022/10/27 14:34
"""
import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.append("../../")
print(sys.path)
from agent_config import ADDRESS
from agent_config import config as agent_config
from configs.config import get_config
from envs.xsim_battle.battle_env import Battle5v5Env
from envs.xsim_battle.alpha_battle_env import AlphaBattleEnv
from envs.xsimenv_wrapper import ShareDummyVecEnv, ShareSubprocVecEnv
# from runners.battle_runner import BattleRunner as Runner
from runners.battle_alpha_runner import BattleAlphaRunner as Runner

'''
宏动作：
1.条件函数编写 ***
2.location计算函数编写 ***
3.其他函数编写 *

运行相关：
1. config参数文件编写 *
2. 其他运行(训练资源tensorflow适配，模型保存与读取等） **

环境相关：
1. Battle5v5Env文件编写 *
2. obs信息处理，entity信息处理和scalar信息处理（要适配在Battle5v5Env文件编写当中，并且处理后的信息直接用在宏动作计算上）**
3. 奖励函数编写 *

算法相关
1. 编写base_runner(负责调用训练函数）和trainer编写 *
2. 编写battle_runner和buffer （收集和存放数据，和运行主函数) **
3. 神经网络模型（简单mlp和alphastar都要有，先用mlp测试）****
4. happo的policy编写（直接用就行，但有些地方要改成tensorflow）*


其他待开发问题：
1. share_obs信息如何设计
2. maybe初始移动策略编写

修改：
1. 百度是3个步长一次决策，三条obs，改为3种mode模式

'''


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "battle":
                env = AlphaBattleEnv(all_args, rank)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "battle":
                env = AlphaBattleEnv(all_args, rank)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--num_agents', type=int, default=5, help="the number of agent")
    all_args = parser.parse_known_args(args)[0]

    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    print("all config: ", all_args)
    if all_args.seed_specify:
        all_args.seed = all_args.runing_id
    else:
        all_args.seed = np.random.randint(1000, 10000)
    print("seed is :", all_args.seed)

    # 待适配tensorflow todo
    # if all_args.cuda and torch.cuda.is_available():
    #     print("choose to use gpu...")
    #     device = torch.device("cuda:0")
    #     限制torch在cpu上使用的线程数
    #     torch.set_num_threads(all_args.n_training_threads)
    #     if all_args.cuda_deterministic:
    #         torch.backends.cudnn.benchmark = False
    #         torch.backends.cudnn.deterministic = True
    # else:
    #     print("choose to use cpu...")
    #     device = torch.device("cpu")
    #     torch.set_num_threads(all_args.n_training_threads)

    # run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
    #                    0] + "/results") / all_args.env_name / all_args.map_name / all_args.algorithm_name / all_args.experiment_name / str(
    #     all_args.seed)
    # if not run_dir.exists():
    #     os.makedirs(str(run_dir))
    #
    # if not run_dir.exists():
    #     curr_run = 'run1'
    # else:
    #     exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
    #                      str(folder.name).startswith('run')]
    #     if len(exst_run_nums) == 0:
    #         curr_run = 'run1'
    #     else:
    #         curr_run = 'run%i' % (max(exst_run_nums) + 1)
    # run_dir = run_dir / curr_run
    # if not run_dir.exists():
    #     os.makedirs(str(run_dir))
    #
    # setproctitle.setproctitle(
    #     str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
    #         all_args.user_name))


    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = 5

    # device = 'cpu'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        # "run_dir": run_dir
    }

    runner = Runner(config)
    runner.run()


if __name__ == "__main__":

    main(sys.argv[1:])
