import numpy as np
import os
import sys
sys.path.append(os.getcwd())

from olympic_mujoco.environments.loco_env_base import LocoEnvBase

import gymnasium as gym

# 获取所有已注册的 Gym 环境的名称
env_names = list(gym.envs.registry.keys())

# 打印环境列表
for env_name in env_names:
    print(env_name)

# def experiment(seed=0):
#     # np.random.seed(seed)

#     # mdp = LocoEnv.make("TestHumanoid.run.perfect")
#     # mdp = LocoEnvBase.make("UnitreeH1.walk.perfect")
#     # mdp = LocoEnvBase.make("StickFigureA1.run.real")
#     # mdp = LocoEnvBase.make("Jvrc.run.real")
#     # mdp = LocoEnvBase.make("StickFigureA3.run.real")
#     mdp = LocoEnvBase.make("Talos.walk.real")


#     mdp.play_trajectory_from_velocity(n_episodes=3, n_steps_per_episode=500)


#     # mdp.play_trajectory(n_episodes=3, n_steps_per_episode=500)
#     # print(mdp.get_all_task_names())
    
#     # mdp.test()

# if __name__ == '__main__':
#     experiment()
#     # test_gym()