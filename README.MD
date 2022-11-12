



### 观测空间
观测空间处理代码编写在/envs/xsim_battle/observation目录下
1. 实体obs信息处理在Battle5v5Env.process_entity_obs函数里编写
2. scalar信息处理在Battle5v5Env.process_scalar_obs函数里编写


### 动作空间
宏动作代码编写在/envs/xsim_battle/macro_action目录下
1. 宏动作条件判断函数编写在action_condition.py文件下
2. 宏动作位置点计算函数编写在location_calculate.py文件下
3. 其他功能的函数暂编写在other_function.py文件下

### 奖励函数
奖励函数编写在/envs/xsim_battle/reward.py文件里

### 环境处理
环境处理主入口在battle5v5_env.py文件下



### 现存问题
1. 动作空间仅适配离散动作空间，暂未配置连续动作空间