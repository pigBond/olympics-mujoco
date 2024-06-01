import os
import numpy as np
import transforms3d as tf3
import collections

from pathlib import Path
from copy import deepcopy
from dm_control import mjcf

from mushroom_rl.utils.running_stats import *
from mushroom_rl.utils.mujoco import *

from olympic_mujoco.environments.base_robot.base_humanoid_robot import BaseHumanoidRobot
from olympic_mujoco.utils import check_validity_task_mode_dataset
from olympic_mujoco.environments.loco_env_base import ValidTaskConf

from olympic_mujoco.interfaces.mujoco_robot_interface import MujocoRobotInterface
from olympic_mujoco.enums.enums import AlgorithmType
from olympic_mujoco.tasks import walking_task
from olympic_mujoco.environments import robot

"""
# ---right leg

R_HIP_P         1
R_HIP_R         2
R_HIP_Y         3
R_KNEE          4
R_ANKLE_R       5
R_ANKLE_P       6 

# ---left leg  

L_HIP_P         7
L_HIP_R         8
L_HIP_Y         9
L_KNEE          10
L_ANKLE_R       11
L_ANKLE_P       12

# ---waist   

WAIST_Y         13  
WAIST_P         14
WAIST_R         15

# ---head

NECK_Y          16
NECK_R          17
NECK_P          18

# ---right arm

R_SHOULDER_P    19
R_SHOULDER_R    20
R_SHOULDER_Y    21
R_ELBOW_P       22
R_ELBOW_Y       23
R_WRIST_R       24
R_WRIST_Y       25
R_UTHUMB        26

# ---left arm

L_SHOULDER_P    27
L_SHOULDER_R    28
L_SHOULDER_Y    29
L_ELBOW_P       30
L_ELBOW_Y       31
L_WRIST_R       32
L_WRIST_Y       33
L_UTHUMB        34

"""

class Jvrc(BaseHumanoidRobot):

    valid_task_confs = ValidTaskConf(
        tasks=["walk", "run", "test"], data_types=["real", "perfect"]
    )

    def __init__(self, disable_arms=True, disable_back_joint=False, **kwargs):
        """
        Constructor.

        """
        # print("Jvrc init")

        #TODO：上面的是完整的jvrc机器人模型xml文件，需要配合_delete_from_xml_handle使用，但是可能是因为模型差异，不能进行良好的行走
        # 下面的是jvrc简易模型xml文件，不需要配合_delete_from_xml_handle使用，但是存在问题，不能正常显示皮肤且只显示机器人的下半身
        # 这里简单先加一个bool用于区分这两个模型
        
        train_about=True
        # 为true表明是与强化学习训练相关的 ,对应的是不完整的模型,即jvrc_step中的
        
        if train_about:
            xml_path = (
                Path(__file__).resolve().parent.parent
                / "data"
                / "jvrc_step"
                / "jvrc1.xml"
            ).as_posix()
        else:
            xml_path = (
                Path(__file__).resolve().parent.parent
                / "data"
                / "jvrc_mj_description"
                / "xml"
                / "jvrc1.xml"
            ).as_posix()


        action_spec = self._get_action_specification()

        observation_spec = self._get_observation_specification()

        collision_groups = [
            ("floor", ["floor"]),
            ("foot_r", ["R_FOOT"]),
            ("foot_l", ["L_FOOT"]),
        ]

        self._hidable_obs = ("positions", "velocities", "foot_forces", "weight")

        # --- Modify the xml, the action_spec, and the observation_spec if needed ---
        self._disable_arms = disable_arms
        self._disable_back_joint = disable_back_joint

        xml_handles = mjcf.from_path(xml_path)

        if not train_about:
            # 禁用手臂的逻辑
            # 分为两部分 1.使能手臂使得其不会触碰到机器人的其他身体部位
            # if disable_arms:
            #     xml_handle = self._reorient_arms(xml_handle) # reposition the arm

            # 2.将手臂等无关机器人行走的joint,motor,equ从观测空间中移除
            joints_to_remove, motors_to_remove, equ_constr_to_remove = self._get_xml_modifications()
            xml_handles = self._delete_from_xml_handle(xml_handles, joints_to_remove,
                                                            motors_to_remove, equ_constr_to_remove)

        super().__init__(
            xml_handles, action_spec, observation_spec, collision_groups, **kwargs
        )

        self._initialize_observation_space()

        
    #---------------------------------------------------------------------------------------------------------
    #----------------------------------------- 观测空间的处理 --------------------------------------------------
    #---------------------------------------------------------------------------------------------------------
    # 用于根据算法和任务初始化观测空间
    # 分为两部分，一部分是设置观测空间的一系列参数，用于强化学习
    # 另一部分是用于设定观测空间都有哪些东西，即 _get_observation_specification
    # 但是考虑到模仿学习和强化学习的观测空间差别较大，所以这里在强化学习相关的操作中，先不使用 模仿学习的规范，这个存在一定的问题
    def _initialize_observation_space(self):
        # print("根据算法和任务初始化观测空间")
        if self._algorithm_type == AlgorithmType.REINFORCEMENT_LEARNING:
            # print("强化学习")
            
            sim_dt = 0.0025  # 仿真步长设置为0.0025秒
            control_dt = 0.025  # 控制步长设置为0.025秒
            frame_skip = (control_dt/sim_dt)  # 计算需要跳过的帧数，以保持仿真与控制步长的一致性

            pdgains = np.zeros((12, 2))
            coeff = 0.5

            kp_values=[200, 200, 200, 250, 80, 80,200, 200, 200, 250, 80, 80]
            kd_values=[20, 20, 20, 25, 8, 8,20, 20, 20, 25, 8, 8]

            # kp_values = [
            #     200, 200, 200, 250, 80, 80,  # Right leg (R_HIP_P, R_HIP_R, R_HIP_Y, R_KNEE, R_ANKLE_R, R_ANKLE_P)
            #     200, 200, 200, 250, 80, 80,  # Left leg (L_HIP_P, L_HIP_R, L_HIP_Y, L_KNEE, L_ANKLE_R, L_ANKLE_P)
            #     100, 100, 100,  # Waist (WAIST_Y, WAIST_P, WAIST_R) - Assuming similar but lower values than legs
            #     60, 60, 60,  # Head (NECK_Y, NECK_R, NECK_P) - Assuming lower values due to smaller, lighter structure
            #     150, 150, 150, 120, 50, 50, 50, 30,  # Right arm (Shoulder, Elbow, Wrist, Thumb) - Values between legs and head
            #     150, 150, 150, 120, 50, 50, 50, 30,  # Left arm (Shoulder, Elbow, Wrist, Thumb) - Mirror of right arm
            # ]
            # kd_values = [
            #     20, 20, 20, 25, 8, 8,
            #     20, 20, 20, 25, 8, 8,
            #     10, 10, 10,
            #     6, 6, 6,
            #     15, 15, 15, 12, 5, 5, 5, 3, 
            #     15, 15, 15, 12, 5, 5, 5, 3,  
            # ]

            pdgains.T[0] = coeff * np.array(kp_values)
            pdgains.T[1] = coeff * np.array(kd_values)

            # list of desired actuators
            # RHIP_P 0, RHIP_R 1, RHIP_Y 2, RKNEE 3, RANKLE_R 4, RANKLE_P 5
            # LHIP_P 6, LHIP_R 7, LHIP_Y 8, LKNEE 9, LANKLE_R 10, LANKLE_P 11
            self.actuators = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

            # set up interface
            self.interface = MujocoRobotInterface(self._model, self._data, 'R_ANKLE_P_S', 'L_ANKLE_P_S')

            # set up task
            self.task = walking_task.WalkingTask(client=self.interface,
                                                dt=control_dt,
                                                neutral_foot_orient=np.array([1, 0, 0, 0]),
                                                root_body='PELVIS_S',
                                                lfoot_body='L_ANKLE_P_S',
                                                rfoot_body='R_ANKLE_P_S',
                                                head_body='NECK_P_S',
            )
            # set goal height
            self.task._goal_height_ref = 0.80
            self.task._total_duration = 1.1
            self.task._swing_duration = 0.75
            self.task._stance_duration = 0.35

            self.robot = robot.JVRC(pdgains.T, control_dt, self.actuators, self.interface)

            # define indices for action and obs mirror fns
            base_mir_obs = [0.1, -1, 2, -3,              # root orient
                            -4, 5, -6,                   # root ang vel
                            13, -14, -15, 16, -17, 18,   # motor pos [1]
                            7,  -8,  -9, 10, -11, 12,   # motor pos [2]
                            25, -26, -27, 28, -29, 30,   # motor vel [1]
                            19, -20, -21, 22, -23, 24,   # motor vel [2]
            ]
            append_obs = [(len(base_mir_obs)+i) for i in range(10)]
            self.robot.clock_inds = append_obs[0:2]
            self.robot.mirrored_obs = np.array(base_mir_obs + append_obs, copy=True).tolist()
            self.robot.mirrored_acts = [6, -7, -8, 9, -10, 11,
                                        0.1, -1, -2, 3, -4, 5,]

            # set action space
            action_space_size = len(self.robot.actuators)
            action = np.zeros(action_space_size)
            self.action_space = np.zeros(action_space_size)

            # set observation space
            self.base_obs_len = 41
            self.observation_space = np.zeros(self.base_obs_len)

        elif self._algorithm_type == AlgorithmType.IMITATION_LEARNING:
            print("模仿学习！")

    # 这里同样要考虑两种观测空间，一种是模仿学习的观测空间，另一种是强化学习的
    def get_obs(self):
        if self._algorithm_type == AlgorithmType.REINFORCEMENT_LEARNING:
            # print("强化学习")
            # external state
            clock = [np.sin(2 * np.pi * self.task._phase / self.task._period),
                    np.cos(2 * np.pi * self.task._phase / self.task._period)]
            ext_state = np.concatenate((clock,
                                        np.asarray(self.task._goal_steps_x).flatten(),
                                        np.asarray(self.task._goal_steps_y).flatten(),
                                        np.asarray(self.task._goal_steps_z).flatten(),
                                        np.asarray(self.task._goal_steps_theta).flatten()))
            
            # internal state
            qpos = np.copy(self.interface.get_qpos())
            qvel = np.copy(self.interface.get_qvel())

            root_r, root_p = tf3.euler.quat2euler(qpos[3:7])[0:2]
            root_orient = tf3.euler.euler2quat(root_r, root_p, 0)
            root_ang_vel = qvel[3:6]

            motor_pos = self.interface.get_act_joint_positions()
            motor_vel = self.interface.get_act_joint_velocities()
            motor_pos = [motor_pos[i] for i in self.actuators]
            motor_vel = [motor_vel[i] for i in self.actuators]

            robot_state = np.concatenate([
                root_orient,
                root_ang_vel,
                motor_pos,
                motor_vel,
            ])

            state = np.concatenate([robot_state, ext_state])
            assert state.shape==(self.base_obs_len,)
            return state.flatten()
        
        elif self._algorithm_type == AlgorithmType.IMITATION_LEARNING:
            print("模仿学习！")
    #---------------------------------------------------------------------------------------------------------



    #---------------------------------------------------------------------------------------------------------
    #----------------------------------------- step步进部分 ---------------------------------------------------
    #---------------------------------------------------------------------------------------------------------
    def step(self, a):
        if self._algorithm_type == AlgorithmType.REINFORCEMENT_LEARNING:
            # print("强化学习")
            applied_action = self.robot.step(a)
            # compute reward
            self.task.step()
            rewards = self.task.calc_reward(self.robot.prev_torque, self.robot.prev_action, applied_action)
            total_reward = sum([float(i) for i in rewards.values()])

            # check if terminate
            done = self._has_done()

            obs = self.get_obs()
            return obs, total_reward, done, rewards
        elif self._algorithm_type == AlgorithmType.IMITATION_LEARNING:
            print("模仿学习！")

    # TODO：特别注意，这里的强化学习的reset和模仿学习的reset是有很大的区别的
    def reset_model(self):
        '''
        # dynamics randomization
        dofadr = [self.interface.get_jnt_qveladr_by_name(jn)
                  for jn in self.interface.get_actuated_joint_names()]
        for jnt in dofadr:
            self.model.dof_frictionloss[jnt] = np.random.uniform(0,10)    # actuated joint frictionloss
            self.model.dof_damping[jnt] = np.random.uniform(0.2,5)        # actuated joint damping
            self.model.dof_armature[jnt] *= np.random.uniform(0.90, 1.10) # actuated joint armature
        '''

        c = 0.02
        self.init_qpos = list(self.robot.init_qpos_)
        self.init_qvel = list(self.robot.init_qvel_)
        self.init_qpos = self.init_qpos + np.random.uniform(low=-c, high=c, size=self._model.nq)
        self.init_qvel = self.init_qvel + np.random.uniform(low=-c, high=c, size=self._model.nv)

        # modify init state acc to task
        root_adr = self.interface.get_jnt_qposadr_by_name('root')[0]
        self.init_qpos[root_adr+0] = np.random.uniform(-1, 1)
        self.init_qpos[root_adr+1] = np.random.uniform(-1, 1)
        self.init_qpos[root_adr+2] = 0.81 # 初始状态下机器人距离地面的高度 这里注意需要根据机器人模型进行修改
        self.init_qpos[root_adr+3:root_adr+7] = tf3.euler.euler2quat(0, np.random.uniform(-5, 5)*np.pi/180, np.random.uniform(-np.pi, np.pi))
        self.set_state(
            self.init_qpos,
            self.init_qvel
        )
        self.task.reset(iter_count = self.robot.iteration_count)
        obs = self.get_obs()

        return obs
    #---------------------------------------------------------------------------------------------------------
            
    #---------------------------------------------------------------------------------------------------------
    #----------------------------------------- xml操作(也相当于是对观测空间的操作) --------------------------------
    #---------------------------------------------------------------------------------------------------------

    def _reorient_arms(xml_handle):
        """
        Reorients the elbow to not collide with the hip.

        Args:
            xml_handle: Handle to Mujoco XML.

        Returns:
            Modified Mujoco XML handle.

        """
        # TODO:这部分需要再考虑
        # modify the arm orientation
        # left_shoulder_pitch_link = xml_handle.find("body", "left_shoulder_pitch_link")
        # left_shoulder_pitch_link.quat = [1.0, 0.25, 0.1, 0.0]
        # right_elbow_link = xml_handle.find("body", "right_elbow_link")
        # right_elbow_link.quat = [1.0, 0.0, 0.25, 0.0]
        # right_shoulder_pitch_link = xml_handle.find("body", "right_shoulder_pitch_link")
        # right_shoulder_pitch_link.quat = [1.0, -0.25, 0.1, 0.0]
        # left_elbow_link = xml_handle.find("body", "left_elbow_link")
        # left_elbow_link.quat = [1.0, 0.0, 0.25, 0.0]

        return xml_handle

    # TODO:这里的删除是为了保证手臂等部位不会影响训练工作,也就是对观测空间obs的操作
    def _get_xml_modifications(self):
        """
        Function that specifies which joints, motors and equality constraints
        should be removed from the Mujoco xml.

        Returns:
            A tuple of lists consisting of names of joints to remove, names of motors to remove,
             and names of equality constraints to remove.

        """

        joints_to_remove = []
        motors_to_remove = []
        equ_constr_to_remove = []

        if self._disable_arms:
            joints_to_remove += [
                # head
                "NECK_Y",
                "NECK_R",
                "NECK_P",
                # waist
                "WAIST_Y",
                "WAIST_P",
                "WAIST_R",
                # right arm
                # "R_SHOULDER_P",
                # "R_SHOULDER_R",
                "R_SHOULDER_Y",
                # "R_ELBOW_P",
                "R_ELBOW_Y",
                "R_WRIST_R",
                "R_WRIST_Y",
                "R_UTHUMB",
                "R_LTHUMB",
                "R_UINDEX",
                "R_LINDEX",
                "R_ULITTLE",
                "R_LLITTLE",
                # left arm
                # "L_SHOULDER_P",
                # "L_SHOULDER_R",
                "L_SHOULDER_Y",
                # "L_ELBOW_P",
                "L_ELBOW_Y",
                "L_WRIST_R",
                "L_WRIST_Y",
                "L_UTHUMB",
                "L_LTHUMB",
                "L_UINDEX",
                "L_LINDEX",
                "L_ULITTLE",
                "L_LLITTLE",
            ]
            motors_to_remove += [
                # head
                "NECK_Y_motor",
                "NECK_R_motor",
                "NECK_P_motor",
                # waist
                "WAIST_Y_motor",
                "WAIST_P_motor",
                "WAIST_R_motor",
                # right arm
                "R_SHOULDER_P_motor",
                "R_SHOULDER_R_motor",
                "R_SHOULDER_Y_motor",
                "R_ELBOW_P_motor",
                "R_ELBOW_Y_motor",
                "R_WRIST_R_motor",
                "R_WRIST_Y_motor",
                "R_UTHUMB_motor",
                # left arm
                "L_SHOULDER_P_motor",
                "L_SHOULDER_R_motor",
                "L_SHOULDER_Y_motor",
                "L_ELBOW_P_motor",
                "L_ELBOW_Y_motor",
                "L_WRIST_R_motor",
                "L_WRIST_Y_motor",
                "L_UTHUMB_motor",
            ]
        return joints_to_remove, motors_to_remove, equ_constr_to_remove

    @staticmethod
    def _get_observation_specification():
        """
        Getter for the observation space specification.

        Returns:
            A list of tuples containing the specification of each observation
            space entry.

        """

        observation_spec = [  # ------------- JOINT POS -------------
            ("q_R_HIP_R", "R_HIP_R", ObservationType.JOINT_POS),
            ("q_R_HIP_Y", "R_HIP_Y", ObservationType.JOINT_POS),
            ("q_R_HIP_P", "R_HIP_P", ObservationType.JOINT_POS),
            ("q_R_KNEE", "R_KNEE", ObservationType.JOINT_POS),
            ("q_R_ANKLE_R", "R_ANKLE_R", ObservationType.JOINT_POS),
            ("q_R_ANKLE_P", "R_ANKLE_P", ObservationType.JOINT_POS),
            ("q_L_HIP_R", "L_HIP_R", ObservationType.JOINT_POS),
            ("q_L_HIP_Y", "L_HIP_Y", ObservationType.JOINT_POS),
            ("q_L_HIP_P", "L_HIP_P", ObservationType.JOINT_POS),
            ("q_L_KNEE", "L_KNEE", ObservationType.JOINT_POS),
            ("q_L_ANKLE_R", "L_ANKLE_R", ObservationType.JOINT_POS),
            ("q_L_ANKLE_P", "L_ANKLE_P", ObservationType.JOINT_POS),
            # ("q_R_SHOULDER_R", "R_SHOULDER_R", ObservationType.JOINT_POS),
            # ("q_R_SHOULDER_Y", "R_SHOULDER_Y", ObservationType.JOINT_POS),
            # ("q_R_SHOULDER_P", "R_SHOULDER_P", ObservationType.JOINT_POS),
            # ("q_R_ELBOW_P", "R_ELBOW_P", ObservationType.JOINT_POS),
            # ("q_R_ELBOW_Y", "R_ELBOW_Y", ObservationType.JOINT_POS),
            # ("q_L_SHOULDER_R", "L_SHOULDER_R", ObservationType.JOINT_POS),
            # ("q_L_SHOULDER_Y", "L_SHOULDER_Y", ObservationType.JOINT_POS),
            # ("q_L_SHOULDER_P", "L_SHOULDER_P", ObservationType.JOINT_POS),
            # ("q_L_ELBOW_P", "L_ELBOW_P", ObservationType.JOINT_POS),
            # ("q_L_ELBOW_Y", "L_ELBOW_Y", ObservationType.JOINT_POS),
            # ------------- JOINT VEL -------------
            ("dq_R_HIP_R", "R_HIP_R", ObservationType.JOINT_VEL),
            ("dq_R_HIP_Y", "R_HIP_Y", ObservationType.JOINT_VEL),
            ("dq_R_HIP_P", "R_HIP_P", ObservationType.JOINT_VEL),
            ("dq_R_KNEE", "R_KNEE", ObservationType.JOINT_VEL),
            ("dq_R_ANKLE_R", "R_ANKLE_R", ObservationType.JOINT_VEL),
            ("dq_R_ANKLE_P", "R_ANKLE_P", ObservationType.JOINT_VEL),
            ("dq_L_HIP_R", "L_HIP_R", ObservationType.JOINT_VEL),
            ("dq_L_HIP_Y", "L_HIP_Y", ObservationType.JOINT_VEL),
            ("dq_L_HIP_P", "L_HIP_P", ObservationType.JOINT_VEL),
            ("dq_L_KNEE", "L_KNEE", ObservationType.JOINT_VEL),
            ("dq_L_ANKLE_R", "L_ANKLE_R", ObservationType.JOINT_VEL),
            ("dq_L_ANKLE_P", "L_ANKLE_P", ObservationType.JOINT_VEL),
            # ("dq_R_SHOULDER_R", "R_SHOULDER_R", ObservationType.JOINT_VEL),
            # ("dq_R_SHOULDER_Y", "R_SHOULDER_Y", ObservationType.JOINT_VEL),
            # ("dq_R_SHOULDER_P", "R_SHOULDER_P", ObservationType.JOINT_VEL),
            # ("dq_R_ELBOW_P", "R_ELBOW_P", ObservationType.JOINT_VEL),
            # ("dq_R_ELBOW_Y", "R_ELBOW_Y", ObservationType.JOINT_VEL),
            # ("dq_L_SHOULDER_R", "L_SHOULDER_R", ObservationType.JOINT_VEL),
            # ("dq_L_SHOULDER_Y", "L_SHOULDER_Y", ObservationType.JOINT_VEL),
            # ("dq_L_SHOULDER_P", "L_SHOULDER_P", ObservationType.JOINT_VEL),
            # ("dq_L_ELBOW_P", "L_ELBOW_P", ObservationType.JOINT_VEL),
            # ("dq_L_ELBOW_Y", "L_ELBOW_Y", ObservationType.JOINT_VEL),
        ]

        return observation_spec

    @staticmethod
    def _get_action_specification():
        """
        Getter for the action space specification.

        Returns:
            A list of tuples containing the specification of each action
            space entry.

        """

        action_spec = [
            # right leg
            "R_HIP_P_motor",
            "R_HIP_R_motor",
            "R_HIP_Y_motor",
            "R_KNEE_motor",
            "R_ANKLE_R_motor",
            "R_ANKLE_P_motor",
            # left leg
            "L_HIP_P_motor",
            "L_HIP_R_motor",
            "L_HIP_Y_motor",
            "L_KNEE_motor",
            "L_ANKLE_R_motor",
            "L_ANKLE_P_motor",
            # # waist
            # "WAIST_Y_motor",
            # "WAIST_P_motor",
            # "WAIST_R_motor",
            # # right arm
            # "R_SHOULDER_P_motor",
            # "R_SHOULDER_R_motor",
            # "R_SHOULDER_Y_motor",
            # "R_ELBOW_P_motor",
            # "R_ELBOW_Y_motor",
            # "R_WRIST_R_motor",
            # "R_WRIST_Y_motor",
            # "R_UTHUMB_motor",
            # # left arm
            # "L_SHOULDER_P_motor",
            # "L_SHOULDER_R_motor",
            # "L_SHOULDER_Y_motor",
            # "L_ELBOW_P_motor",
            # "L_ELBOW_Y_motor",
            # "L_WRIST_R_motor",
            # "L_WRIST_Y_motor",
            # "L_UTHUMB_motor",
        ]

        return action_spec
    #---------------------------------------------------------------------------------------------------------


    #---------------------------------------------------------------------------------------------------------
    #----------------------------------------- 吸收终止函数 --------------------------------------------------
    #---------------------------------------------------------------------------------------------------------
    def _has_done(self):
        return self.task.done()

    def _has_fallen(self, obs, return_err_msg=False):
        """
        Checks if a model has fallen.

        Args:
            obs (np.array): Current observation.
            return_err_msg (bool): If True, an error message with violations is returned.

        Returns:
            True, if the model has fallen for the current observation, False otherwise.
            Optionally an error message is returned.

        """
        return False
        # # Extracting information related to pelvic Euler angles from observational data
        # pelvis_euler = self._get_from_obs(obs, ["q_pelvis_tilt", "q_pelvis_list", "q_pelvis_rotation"])
        # # Determine whether the position of the pelvis along the y-axis meets specific conditions
        # # 判断骨盆沿y轴的位置是否满足特定条件
        # pelvis_y_condition = (obs[0] < -0.3) or (obs[0] > 0.1)
        # pelvis_tilt_condition = (pelvis_euler[0] < (-np.pi / 4.5)) or (pelvis_euler[0] > (np.pi / 12))
        # pelvis_list_condition = (pelvis_euler[1] < -np.pi / 12) or (pelvis_euler[1] > np.pi / 8)
        # pelvis_rotation_condition = (pelvis_euler[2] < (-np.pi / 8)) or (pelvis_euler[2] > (np.pi / 8))

        # # Based on the above conditions, determine whether the pelvis is in an unsafe state
        # # If any condition is true, it is considered that the pelvic condition is not met
        # pelvis_condition = (pelvis_y_condition or pelvis_tilt_condition or
        #                     pelvis_list_condition or pelvis_rotation_condition)

        # if return_err_msg:
        #     error_msg = ""
        #     if pelvis_y_condition:
        #         error_msg += "pelvis_y_condition violated.\n"
        #     elif pelvis_tilt_condition:
        #         error_msg += "pelvis_tilt_condition violated.\n"
        #     elif pelvis_list_condition:
        #         error_msg += "pelvis_list_condition violated.\n"
        #     elif pelvis_rotation_condition:
        #         error_msg += "pelvis_rotation_condition violated.\n"

        #     return pelvis_condition, error_msg
        # else:

        #     return pelvis_condition

    #---------------------------------------------------------------------------------------------------------

    def _get_ground_forces(self):
        """
        Returns the ground forces (np.array). By default, 4 ground force sensors are used.
        Environments that use more or less have to override this function.

        """

        grf = np.concatenate(
            [
                self._get_collision_force("floor", "foot_r")[:3],
                self._get_collision_force("floor", "foot_l")[:3],
            ]
        )

        return grf

    @staticmethod
    def _get_grf_size():
        """
        Returns the size of the ground force vector.

        """

        return 6


    @staticmethod
    def generate(task="walk", dataset_type="real", **kwargs):
        """
        Returns an environment corresponding to the specified task.

        Args:
        task (str): Main task to solve. Either "walk", "run" or "carry". The latter is walking while carrying
                an unknown weight, which makes the task partially observable.
        dataset_type (str): "real" or "perfect". "real" uses real motion capture data as the
                reference trajectory. This data does not perfectly match the kinematics
                and dynamics of this environment, hence it is more challenging. "perfect" uses
                a perfect dataset.

        """
        check_validity_task_mode_dataset(
            Jvrc.__name__, task, None, dataset_type, *Jvrc.valid_task_confs.get_all()
        )
        if dataset_type == "real":
            if task == "run":
                path = "datasets/humanoids/real/random_jvrc.npz"
            else:
                path = "datasets/humanoids/real/random_jvrc.npz"
        elif dataset_type == "perfect":
            if "use_foot_forces" in kwargs.keys():
                assert kwargs["use_foot_forces"] is False
            if "disable_arms" in kwargs.keys():
                assert kwargs["disable_arms"] is True
            if "disable_back_joint" in kwargs.keys():
                assert kwargs["disable_back_joint"] is False
            if "hold_weight" in kwargs.keys():
                assert kwargs["hold_weight"] is False

            if task == "run":
                path = "datasets/humanoids/perfect/unitreeh1_run/perfect_expert_dataset_det.npz"
            else:
                path = "datasets/humanoids/perfect/unitreeh1_walk/perfect_expert_dataset_det.npz"

        return BaseHumanoidRobot.generate(
            Jvrc,
            path,
            task,
            dataset_type,
            clip_trajectory_to_joint_ranges=True,
            **kwargs
        )