import numpy as np
import os
import sys
sys.path.append(os.getcwd())
import olympic_mujoco
import gymnasium as gym


# env = gym.make("OlympicMujoco", env_name="StickFigureA1.run")
env = gym.make("OlympicMujoco", env_name="StickFigureA3.run")

# env = gym.make("OlympicMujoco", env_name="UnitreeH1.run")
# env = gym.make("OlympicMujoco", env_name="Jvrc.run")


# print(type(env))

# 假设 env 是一个 OrderEnforcing 包装器类的实例
# print(type(env.unwrapped))  # 获取原始未被包装的环境


action_dim = env.action_space.shape[0]

env.reset()
env.render()
absorbing = False
i = 0

while True:
    if i == 1000 or absorbing:
        env.reset()
        i = 0
    action = np.random.randn(action_dim)
    # print(action)
    nstate, _, absorbing, _,  _ = env.step(action)

    env.render()
    i += 1