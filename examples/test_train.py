import os
import sys
import argparse
import ray
from functools import partial

import numpy as np
import torch
import pickle

import sys
sys.path.append(os.getcwd())

from olympic_mujoco.environments.loco_env_base import LocoEnvBase

from rl.algos.ppo import PPO
from rl.policies.actor import Gaussian_FF_Actor
from rl.policies.critic import FF_V
from rl.envs.normalize import get_normalization_params
from rl.envs.wrappers import SymmetricEnv

def import_env():
    import olympic_mujoco.environments.real_humanoid_robots.Jvrc as Env
    # import olympic_mujoco.environments.real_humanoid_robots.StickFigureA3 as Env
    return Env

def run_experiment(args):
    # import the correct environment
    Env = import_env()
    # Env = LocoEnvBase.make("Jvrc.run.real")
    # Env = make_env()

    # wrapper function for creating parallelized envs
    env_fn = partial(Env)

    if not args.no_mirror:
        try:
            print("Wrapping in SymmetricEnv.")
            env_fn = partial(SymmetricEnv, env_fn,
                             mirrored_obs=env_fn().robot.mirrored_obs,
                             mirrored_act=env_fn().robot.mirrored_acts,
                             clock_inds=env_fn().robot.clock_inds)
        except AttributeError as e:
            print("Warning! Cannot use SymmetricEnv.", e)
    obs_dim = env_fn().observation_space.shape[0]
    action_dim = env_fn().action_space.shape[0]

    # Set up Parallelism
    os.environ['OMP_NUM_THREADS'] = '1'
    if not ray.is_initialized():
        ray.init(num_cpus=args.num_procs)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 查看gpu信息
    # 检查CUDA是否可用
    if_cuda = torch.cuda.is_available()
    print("if_cuda=", if_cuda)

    # 如果CUDA可用，获取GPU数量
    if if_cuda:
        gpu_count = torch.cuda.device_count()
        print("gpu_count=", gpu_count)

        # 获取并打印每个GPU的型号
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.continued:
        print("-----------------------继续训练之前的模型-----------------------")
        path_to_actor = ""
        path_to_pkl = ""
        if os.path.isfile(args.continued) and args.continued.endswith(".pt"):
            path_to_actor = args.continued
        if os.path.isdir(args.continued):
            path_to_actor = os.path.join(args.continued, "actor.pt")
        path_to_critic = path_to_actor.split('actor')[0]+'critic'+path_to_actor.split('actor')[1]
        policy = torch.load(path_to_actor)
        critic = torch.load(path_to_critic)
    else:
        print("-----------------------训练新的模型-----------------------")
        policy = Gaussian_FF_Actor(obs_dim, action_dim, fixed_std=np.exp(args.std_dev), bounded=False)
        critic = FF_V(obs_dim)
        # policy = Gaussian_FF_Actor_device(obs_dim, action_dim, fixed_std=np.exp(args.std_dev), bounded=False, device=device).to(device)
        # critic = FF_V(obs_dim).to(device)

        # with torch.no_grad():
        #     policy.obs_mean, policy.obs_std = map(torch.Tensor,
        #                                           get_normalization_params(iter=args.input_norm_steps,
        #                                                                    noise_std=1,
        #                                                                    policy=policy,
        #                                                                    env_fn=env_fn,
        #                                                                    procs=args.num_procs))
        critic.obs_mean = policy.obs_mean
        critic.obs_std = policy.obs_std

    policy.train()
    critic.train()

    # dump hyperparameters
    os.makedirs(args.logdir, exist_ok=True)
    pkl_path = os.path.join(args.logdir, "experiment.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(args, f)

    algo = PPO(args=vars(args), save_path=args.logdir)
    algo.train(env_fn, policy, critic, args.n_itr, anneal_rate=args.anneal)

if __name__ == "__main__":
    # --env: 环境名称，必填参数，默认无（必须提供），类型字符串（str）
    # --seed: 随机种子，可选参数，默认0，类型整数（int）
    # --logdir: 日志目录，可选参数，默认"./logs_dir/"，类型字符串（str）
    # --input_norm_steps: 输入标准化步骤，可选参数，默认100000，类型整数（int）
    # --n_itr: 学习算法迭代次数，可选参数，默认20000，类型整数（int）
    # --lr: Adam学习率，可选参数，默认1e-4，类型浮点数（float）
    # --eps: Adam epsilon（数值稳定性），可选参数，默认1e-5，类型浮点数（float）
    # --lam: 广义优势估计折扣，可选参数，默认0.95，类型浮点数（float）
    # --gamma: MDP折扣，可选参数，默认0.99，类型浮点数（float）
    # --anneal: 探索标准差的退火率，可选参数，默认未激活（False），类型布尔值（bool）
    # --std_dev: 探索标准差指数，可选参数，默认-1.5，类型整数（int）
    # --entropy_coeff: 熵正则化系数，可选参数，默认0.0，类型浮点数（float）
    # --clip: PPO代理损失剪辑参数，可选参数，默认0.2，类型浮点数（float）
    # --minibatch_size: PPO更新小批量大小，可选参数，默认64，类型整数（int）
    # --epochs: 每个PPO更新的优化纪元数，可选参数，默认3，类型整数（int）
    # --use_gae: 是否使用GAE计算回报，可选参数，默认True，类型布尔值（bool）
    # --num_procs: 训练的线程数，可选参数，默认12，类型整数（int）
    # --max_grad_norm: 梯度裁剪值，可选参数，默认0.05，类型浮点数（float）
    # --max_traj_len: 最大轨迹长度，可选参数，默认400，类型整数（int）
    # --no_mirror: 是否使用SymmetricEnv，可选参数，默认未激活（False），类型布尔值（bool）
    # --mirror_coeff: 对称损失权重，可选参数，默认0.4，类型浮点数（float）
    # --eval_freq: 评估频率，可选参数，默认100，类型整数（int）
    # --continued: 预训练权重路径，可选参数，默认None，类型字符串（str）
    parser = argparse.ArgumentParser()

    if sys.argv[1] != 'train':
        raise Exception("Invalid usage.")

    sys.argv.remove(sys.argv[1])
    parser.add_argument("--seed", default=0, type=int)                        # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--logdir", type=str, default="./logs_dir/")          # Where to log diagnostics to
    parser.add_argument("--input_norm_steps", type=int, default=100000)
    parser.add_argument("--n_itr", type=int, default=20000, help="Number of iterations of the learning algorithm")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate") # Xie
    parser.add_argument("--eps", type=float, default=1e-5, help="Adam epsilon (for numerical stability)")
    parser.add_argument("--lam", type=float, default=0.95, help="Generalized advantage estimate discount")
    parser.add_argument("--gamma", type=float, default=0.99, help="MDP discount")
    parser.add_argument("--anneal", default=1.0, action='store_true', help="anneal rate for stddev")
    parser.add_argument("--std_dev", type=int, default=-1.5, help="exponent of exploration std_dev")
    parser.add_argument("--entropy_coeff", type=float, default=0.0, help="Coefficient for entropy regularization")
    parser.add_argument("--clip", type=float, default=0.2, help="Clipping parameter for PPO surrogate loss")
    parser.add_argument("--minibatch_size", type=int, default=64, help="Batch size for PPO updates")
    parser.add_argument("--epochs", type=int, default=3, help="Number of optimization epochs per PPO update") #Xie
    parser.add_argument("--use_gae", type=bool, default=True,help="Whether or not to calculate returns using Generalized Advantage Estimation")
    parser.add_argument("--num_procs", type=int, default=12, help="Number of threads to train on")
    parser.add_argument("--max_grad_norm", type=float, default=0.05, help="Value to clip gradients at.")
    parser.add_argument("--max_traj_len", type=int, default=400, help="Max episode horizon")
    parser.add_argument("--no_mirror", required=False, action="store_true", help="to use SymmetricEnv")
    parser.add_argument("--mirror_coeff", required=False, default=0.4, type=float, help="weight for mirror loss")
    parser.add_argument("--eval_freq", required=False, default=100, type=int, help="Frequency of performing evaluation")
    parser.add_argument("--continued", required=False, default=None, type=str, help="path to pretrained weights")
    args = parser.parse_args()

    run_experiment(args)
