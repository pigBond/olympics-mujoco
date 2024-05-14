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
    # _path="/home/wzx/test-workspace/LearningHumanoidWalking/trained/jvrc_stepper/actor.pt"
    _path="/home/wzx/new-Github-Workspaces/olympics-mujoco/olympic_mujoco/trained/test_test_5/actor.pt"

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

    env = LocoEnvBase.make("Jvrc.run.real")

    # path_to_actor =  /home/wzx/test-workspace/LearningHumanoidWalking/trained/jvrc_stepper/actor.pt
    # run_args =  Namespace(anneal=1.0, clip=0.2, continued=None, entropy_coeff=0.0, env='jvrc_step', epochs=3, eps=1e-05, eval_freq=100, gamma=0.99, input_norm_steps=100000, lam=0.95, logdir='stepper_logs/log_jvrc_full', lr=0.0001, max_grad_norm=0.05, max_traj_len=400, minibatch_size=64, mirror_coeff=0.4, n_itr=20000, no_mirror=False, num_procs=24, num_steps=5096, seed=0, std_dev=-1.5, use_gae=True)
    # policy =  Gaussian_FF_Actor(
    #     (actor_layers): ModuleList(
    #         (0): Linear(in_features=41, out_features=256, bias=True)
    #         (1): Linear(in_features=256, out_features=256, bias=True)
    #     )
    #     (means): Linear(in_features=256, out_features=12, bias=True)
    #     )


    run(env, policy)
    print("-----------------------------------------")

if __name__=='__main__':
    main()
