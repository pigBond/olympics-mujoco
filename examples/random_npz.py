import numpy as np

# Unitree H1
# 关节名称列表
# joint_names = ['pelvis_tx', 'pelvis_tz', 'pelvis_ty', 'pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 
#                'back_bkz', 'l_arm_shy', 'l_arm_shx', 'l_arm_shz', 'left_elbow', 'r_arm_shy', 'r_arm_shx',
#                'r_arm_shz', 'right_elbow', 'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r',
#                'ankle_angle_r', 'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', 'ankle_angle_l']


# StickFigure A1
# joint_names = [
# 'pelvis_tx', 'pelvis_tz', 'pelvis_ty',
# 'right_hip_x', 'right_hip_z', 'right_hip_y', 'right_knee', 'right_ankle_x', 'right_ankle_y',
# 'left_hip_x', 'left_hip_z', 'left_hip_y', 'left_knee', 'left_ankle_x', 'left_ankle_y',
# 'right_shoulder1', 'right_shoulder2', 'right_elbow',
# 'left_shoulder1', 'left_shoulder2', 'left_elbow'
# ]

# Jvrc
joint_names = [
    # right leg
    'R_HIP_P', 'R_HIP_R', 'R_HIP_Y', 'R_KNEE', 'R_ANKLE_R', 'R_ANKLE_P',
    # left leg
    'L_HIP_P', 'L_HIP_R', 'L_HIP_Y', 'L_KNEE', 'L_ANKLE_R', 'L_ANKLE_P',
    # waist
    'WAIST_Y', 'WAIST_P', 'WAIST_R',
    # right arm
    'R_SHOULDER_P', 'R_SHOULDER_R', 'R_SHOULDER_Y', 'R_ELBOW_P', 'R_ELBOW_Y', 'R_WRIST_R', 'R_WRIST_Y', 'R_UTHUMB',
    # left arm
    'L_SHOULDER_P', 'L_SHOULDER_R', 'L_SHOULDER_Y', 'L_ELBOW_P', 'L_ELBOW_Y', 'L_WRIST_R', 'L_WRIST_Y', 'L_UTHUMB'
]


# 设置轨迹长度和时间步长
traj_length = 1000

# 生成随机关节位置和速度数据
traj_data = {}
for joint in joint_names:
    traj_data[f'q_{joint}'] = np.random.uniform(low=-np.pi, high=np.pi, size=(traj_length, 1))
    traj_data[f'dq_{joint}'] = np.random.uniform(low=-1.0, high=1.0, size=(traj_length, 1))

# 保存为npz文件
np.savez('random_stick_jvrc.npz', **traj_data)
