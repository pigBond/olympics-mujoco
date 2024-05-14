import numpy as np
import os
import sys
sys.path.append(os.getcwd())

from olympic_mujoco.environments.loco_env_base import LocoEnvBase

def show_menu(options):
    print("\n请选择一个演示理想行走轨迹模型:")
    for index, option in enumerate(options, start=1):
        print(f"{index}. {option}")
    print(f"{len(options) + 1}. 退出")


    # mdp = LocoEnvBase.make("UnitreeH1.walk.perfect")
    # mdp = LocoEnvBase.make("StickFigureA1.run.real")
    # mdp = LocoEnvBase.make("Jvrc.run.real")
    # mdp = LocoEnvBase.make("StickFigureA3.run.real")
    mdp = LocoEnvBase.make("Talos.walk.real")


def execute_task(option):
    # 这里可以根据选项执行不同的任务
    print(f"执行 {option} 中...")
    task_str=""
    if option == "unitree h1 walk":
        task_str="UnitreeH1.walk.real"
        print("Executing unitree h1 walk...")
    elif option == "unitree h1 run":
        task_str="UnitreeH1.run.real"
        print("Executing unitree h1 run...")
    elif option == "atlas walk":
        task_str="Atlas.walk.real"
        print("Executing atlas walk...")
    elif option == "talos walk":
        task_str="Talos.walk.real"
        print("Executing talos walk...")
    
    print("task_str = ",task_str)
    mdp = LocoEnvBase.make(task_str)
    mdp.play_trajectory_from_velocity(n_episodes=3, n_steps_per_episode=500)

def main():
    # 初始化选项列表
    options = ["unitree h1 walk", "unitree h1 run", "atlas walk" , "talos walk"]

    while True:
        show_menu(options)
        try:
            choice = int(input("请输入你的选择（1-{}）: ".format(len(options) + 1)))
            if choice == len(options) + 1:
                print("退出程序。")
                break
            elif 1 <= choice <= len(options):
                # 执行对应的任务
                execute_task(options[choice - 1])
                break
            else:
                print("无效的选择，请重新输入。")
        except ValueError:
            print("请输入一个有效的数字。")

if __name__ == "__main__":
    main()
