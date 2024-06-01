import numpy as np
from mushroom_rl.core import Core, Agent
from olympic_mujoco.environments.loco_env_base import LocoEnvBase
from olympic_mujoco.enums.enums import AlgorithmType

# 创建环境
env = LocoEnvBase.make("UnitreeH1.walk")
env.set_algorithm_type(AlgorithmType.IMITATION_LEARNING)

# 加载智能体
# VAIL算法训练得到的模型
# D:\Github-Workspaces\olympics-mujoco\logs_2\env_id___UnitreeH1.walk__vail\2\agent_epoch_393_J_983.155679.msh
# logs_2/env_id___UnitreeH1.walk__vail/2/agent_epoch_393_J_983.155679.msh

# GAIL算法训练得到的模型
# D:\Github-Workspaces\olympics-mujoco\logs_2\env_id___UnitreeH1.walk__gail\2\agent_epoch_397_J_870.328114.msh
# logs_2/env_id___UnitreeH1.walk__gail/2/agent_epoch_397_J_870.328114.msh

agent_path="logs_2/env_id___UnitreeH1.walk__vail/2/agent_epoch_393_J_983.155679.msh"
# agent_path="logs_2/env_id___UnitreeH1.walk__gail/2/agent_epoch_397_J_870.328114.msh"

agent = Agent.load(agent_path)

# 创建核心对象
core = Core(agent, env)

def evaluate_(n_episodes,n_steps_per_episode=500,render=True):
    for k in range(n_episodes):
        # 重置环境
        state = env.reset()

        print("k = ",k)
        for z in range(n_steps_per_episode):
            # 使用智能体选择动作
            action = agent.draw_action(state)
            # 执行动作，获取下一个状态和奖励
            next_state, reward, done, info = env.step(action)

            # observation_keys = env.get_all_observation_keys()
            # print("observation_keys = ",observation_keys)

            # 更新状态
            state = next_state
            # 渲染环境
            if render:
                env.render()
        
# 调用评估函数
evaluate_(n_episodes=3)
