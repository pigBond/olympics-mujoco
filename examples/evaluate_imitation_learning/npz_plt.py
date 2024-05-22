import os
import numpy as np
import matplotlib.pyplot as plt

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # SimHei是Windows系统中的一种中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 定义一个函数来加载并输出npz文件的内容
def load_and_compare_npz(file_paths,save_path,labels, colors):
    # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 加载所有npz文件
    data_dict = {}
    for file_path in file_paths:
        data = np.load(file_path)
        data_dict[file_path] = data
    
    # 获取所有文件中的所有key
    all_keys = set()
    for data in data_dict.values():
        all_keys.update(data.files)
    
    # 找出所有文件中相同的key
    common_keys = all_keys.copy()
    for keys in data_dict.values():
        common_keys &= set(keys)
    
    # 打印并比较相同key的数据
    for key in common_keys:
        # 绘制图线进行对比
        plt.figure(figsize=(10, 5))
        for file_path, data in data_dict.items():
            if key in data:
                plt.plot(data[key], label=labels[file_paths.index(file_path)], color=colors[file_paths.index(file_path)])
        plt.title(f'Comparison of {key}')
        plt.legend(loc='upper right')
        # plt.show()
        plt.savefig(f'{save_path}/{key}.png')  # 保存图像到指定路径
        plt.close()  # 关闭当前活动图像，避免下一个图像覆盖上一个图像
        print(f"Image for {key} has been saved.")  # 打印保存成功的消息

# 使用函数加载并比较npz文件的内容
save_path = 'saved_npz/comparison_plt'  # 指定保存图像的文件夹路径
file_paths = [
    "olympic_mujoco/datasets/humanoids/real/mini_datasets/02-constspeed_UnitreeH1.npz",
    "saved_npz/vail_unprocessed_0.npz",
    "saved_npz/vail_processed_0.npz",
    "saved_npz/gail_unprocessed_0.npz",
    "saved_npz/gail_processed_0.npz",
]
labels = ['UnitreeH1 理想行走轨迹', 'UnitreeH1 VAIL算法未整形','UnitreeH1 VAIL算法滤波整形' ,'UnitreeH1 GAIL算法未整形','UnitreeH1 GAIL算法滤波整形']
colors = ['blue', 'lightcoral', 'red', 'lightgreen', 'green']
load_and_compare_npz(file_paths,save_path, labels, colors)