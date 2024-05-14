import os
import time
import argparse
import torch
import pickle
import mujoco
import numpy as np
import transforms3d as tf3
import sys
sys.path.append(os.getcwd())

from olympic_mujoco.environments.loco_env_base import LocoEnvBase

def print_reward(ep_rewards):
    mean_rewards = {k:[] for k in ep_rewards[-1].keys()}
    print('*********************************')
    for key in mean_rewards.keys():
        l = [step[key] for step in ep_rewards]
        mean_rewards[key] = sum(l)/len(l)
        print(key, ': ', mean_rewards[key])
        #total_rewards = [r for step in ep_rewards for r in step.values()]
    print('*********************************')
    print("mean per step reward: ", sum(mean_rewards.values()))

def run(env, policy):
    observation = env.test_reset()
    env.render()
    viewer = env.viewer
    viewer._paused = False
    done = False
    ts, end_ts = 0, 2000
    ep_rewards = []

    while (ts < end_ts) and (done == False):
        if hasattr(env, 'frame_skip'):
            start = time.time()

        with torch.no_grad():
            action = policy.forward(torch.Tensor(observation), deterministic=True).detach().numpy()

        observation, _, done, info = env.step(action.copy())
        ep_rewards.append(info)

        env.render()

        if hasattr(env, 'frame_skip'):
            end = time.time()
            sim_dt = env.robot.client.sim_dt()
            delaytime = max(0, env.frame_skip / (1/sim_dt) - (end-start))
            time.sleep(delaytime)
        ts+=1

    print("Episode finished after {} timesteps".format(ts))
    print_reward(ep_rewards)
    env.close()

def main():
    _path="/home/wzx/new-Github-Workspaces/olympics-mujoco/olympic_mujoco/trained/a3_walk/actor.pt"

    path_to_actor = ""
    path_to_pkl = ""
    if os.path.isfile(_path) and _path.endswith(".pt"):
        path_to_actor = _path
        path_to_pkl = os.path.join(os.path.dirname(_path), "experiment.pkl")
    if os.path.isdir(_path):
        path_to_actor = os.path.join(_path, "actor.pt")
        path_to_pkl = os.path.join(_path, "experiment.pkl")

    # load experiment args
    run_args = pickle.load(open(path_to_pkl, "rb"))
    # load trained policy
    policy = torch.load(path_to_actor)
    policy.eval()

    env = LocoEnvBase.make("StickFigureA1.run.real")

    run(env, policy)
    print("-----------------------------------------")

if __name__=='__main__':
    main()
