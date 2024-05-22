import os
import numpy as np
from mushroom_rl.core import Core, Agent
from olympic_mujoco.environments.loco_env_base import LocoEnvBase
from olympic_mujoco.enums.enums import AlgorithmType

def save_episode_data(episode_data, episode_number,folder_path,episode_str):
    """ 保存episode数据到指定文件夹中的npz文件 """
    # 确保文件夹存在
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for key, value in episode_data.items():
        if isinstance(value, list):
            episode_data[key] = np.array(value)
    
    # 构建文件名
    file_name = episode_str + str(episode_number) + '.npz'
    file_path = os.path.join(folder_path, file_name)

    np.savez(file_path, **episode_data)

def smooth_data(data, window_size=5):
    """ 对数据应用移动平均平滑 """
    smoothed_data = []
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        end = i + 1
        window = data[start:end]
        smoothed_value = sum(window) / len(window)
        smoothed_data.append(smoothed_value)
    return smoothed_data

def update_and_store_data(data_dict, observation_keys, values):
    # 确保每个键对应的值是一个列表，并将新的值添加到列表中
    for key, value in zip(observation_keys, values):
        if key not in data_dict:
            data_dict[key] = []  # 如果键不存在，创建一个新列表
        data_dict[key].append(value)  # 将值添加到列表中
    return data_dict

# 评估智能体并保存数据
def evaluate_and_save(agent,env,algorithm_str,folder_path,n_episodes,n_steps_per_episode,render):
    # 创建核心对象
    core = Core(agent, env)
    for k in range(n_episodes):
        # 重置环境
        state = env.reset()
        episode_data = {}
        # 获取观察值的名称
        observation_keys = env.get_all_observation_keys()

        # 初始化数据存储结构
        for key in observation_keys:
            episode_data[key] = []

        for z in range(100):
            # 使用智能体选择动作
            action = agent.draw_action(state)
            # 执行动作，获取下一个状态和奖励
            next_state, reward, done, info = env.step(action)

            # 获取关节位置、速度
            joint_pos = list(env._get_joint_pos())
            joint_vel = list(env._get_joint_vel())
            combined_array = joint_pos + joint_vel

            values = env.test_test()
            for tmp in range(5):
                updated_data_dict = update_and_store_data(episode_data, observation_keys, combined_array)
            
            # 更新状态
            state = next_state
            # 渲染环境
            if render:
                env.render()
        print("k = ",k)

        save_episode_data(episode_data,k,folder_path,algorithm_str+"_unprocessed_")
        # 平滑数据
        for key in episode_data:
            episode_data[key] = smooth_data(episode_data[key])
        save_episode_data(episode_data,k,folder_path,algorithm_str+"_processed_")

if __name__ == '__main__':

    folder_path="saved_npz"
    n_episodes_=1
    n_steps_per_episode_=500
    render_=False

    # 创建环境
    env = LocoEnvBase.make("UnitreeH1.walk")
    env.set_algorithm_type(AlgorithmType.IMITATION_LEARNING)

    # VAIL算法训练得到的模型
    agent_path_vail="logs_2/env_id___UnitreeH1.walk__vail/2/agent_epoch_393_J_983.155679.msh"
    agent_vail = Agent.load(agent_path_vail)

    # GAIL算法训练得到的模型
    agent_path_gail="logs_2/env_id___UnitreeH1.walk__gail/2/agent_epoch_397_J_870.328114.msh"
    agent_gail = Agent.load(agent_path_gail)

    evaluate_and_save(agent_vail,env,"vail",folder_path,n_episodes_,n_steps_per_episode_,render_)

    evaluate_and_save(agent_gail,env,"gail",folder_path,n_episodes_,n_steps_per_episode_,render_)






