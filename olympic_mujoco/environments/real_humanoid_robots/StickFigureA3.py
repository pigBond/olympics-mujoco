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

class StickFigureA3(BaseHumanoidRobot):

    valid_task_confs = ValidTaskConf(tasks=["walk", "run","test"],
                                     data_types=["real", "perfect"]
                                    )

    def __init__(self, disable_arms=True, disable_back_joint=False, **kwargs):
        """
        Constructor.

        """
        # print("StickFigureA3")

        xml_path = (Path(__file__).resolve().parent.parent / "data" / "stickFigure_A3" / "a3.xml").as_posix()

        action_spec = self._get_action_specification()

        observation_spec = self._get_observation_specification()

        collision_groups = [("floor", ["floor"]),
                            ("foot_r", ["right_foot"]),
                            ("foot_l", ["left_foot"])]

        self._hidable_obs = ("positions", "velocities", "foot_forces", "weight")

        # --- Modify the xml, the action_spec, and the observation_spec if needed ---
        self._disable_arms = disable_arms
        self._disable_back_joint = disable_back_joint

        xml_handles = mjcf.from_path(xml_path)

        # 禁用手臂的逻辑
        # 分为两部分 1.使能手臂使得其不会触碰到机器人的其他身体部位
        # if disable_arms:
        #     xml_handle = self._reorient_arms(xml_handle) # reposition the arm

        # 2.将手臂等无关机器人行走的joint,motor,equ从观测空间中移除
        # joints_to_remove, motors_to_remove, equ_constr_to_remove = self._get_xml_modifications()
        # xml_handles = self._delete_from_xml_handle(xml_handles, joints_to_remove,
        #                                                     motors_to_remove, equ_constr_to_remove)

        super().__init__(xml_handles, action_spec, observation_spec, collision_groups, **kwargs)

        self._initialize_observation_space()

    #---------------------------------------------------------------------------------------------------------
    #----------------------------------------- 观测空间的处理 --------------------------------------------------
    #---------------------------------------------------------------------------------------------------------
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

            pdgains.T[0] = coeff * np.array(kp_values)
            pdgains.T[1] = coeff * np.array(kd_values)

            # list of desired actuators
            # left_hip_y,left_hip_x,left_hip_z,left_knee,left_ankle_x,left_ankle_y
            # right_hip_y,right_hip_x,right_hip_z,right_knee,right_ankle_x,right_ankle_y
            self.actuators = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

            # set up interface
            self.interface = MujocoRobotInterface(self._model, self._data, 'right_foot', 'left_foot')

            # print("***********************************************************")
            # print("joint names = ",self.interface.get_joint_names())
            # print("***********************************************************")

            # set up task
            self.task = walking_task.WalkingTask(client=self.interface,
                                               dt=control_dt,
                                               neutral_foot_orient=np.array([1, 0, 0, 0]),
                                               root_body='torso',
                                               lfoot_body='left_foot',
                                               rfoot_body='right_foot',
                                               head_body='head',
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
            done = self.task.done()

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
        self.init_qpos[root_adr+2] = 1.34 # 初始状态下机器人距离地面的高度 这里注意需要根据机器人模型进行修改
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
            # joints_to_remove += ["left_shoulder1","left_shoulder2","left_elbow",
            #                      "right_shoulder1","right_shoulder2","right_elbow"]
            # motors_to_remove += ["left_shoulder1_actuator","left_shoulder2_actuator","left_elbow_actuator",
            #                      "right_shoulder1_actuator","right_shoulder2_actuator","right_elbow_actuator"]
            pass

        return joints_to_remove, motors_to_remove, equ_constr_to_remove
    

    @staticmethod
    def _get_observation_specification():
        """
        Getter for the observation space specification.

        Returns:
            A list of tuples containing the specification of each observation
            space entry.

        """

        observation_spec = [# ------------- JOINT POS -------------
                            # ("q_pelvis_ty", "pelvis_ty", ObservationType.JOINT_POS),
                            # ("q_pelvis_tz", "pelvis_tz", ObservationType.JOINT_POS),
                            # ("q_pelvis_tx", "pelvis_tx", ObservationType.JOINT_POS),
                            ("q_right_hip_x", "right_hip_x", ObservationType.JOINT_POS),
                            ("q_right_hip_z", "right_hip_z", ObservationType.JOINT_POS),
                            ("q_right_hip_y", "right_hip_y", ObservationType.JOINT_POS),
                            ("q_right_knee", "right_knee", ObservationType.JOINT_POS),
                            ("q_right_ankle_x", "right_ankle_x", ObservationType.JOINT_POS),
                            ("q_right_ankle_y", "right_ankle_y", ObservationType.JOINT_POS),
                            ("q_left_hip_x", "left_hip_x", ObservationType.JOINT_POS),
                            ("q_left_hip_z", "left_hip_z", ObservationType.JOINT_POS),
                            ("q_left_hip_y", "left_hip_y", ObservationType.JOINT_POS),
                            ("q_left_knee", "left_knee", ObservationType.JOINT_POS),
                            ("q_left_ankle_x", "left_ankle_x", ObservationType.JOINT_POS),
                            ("q_left_ankle_y", "left_ankle_y", ObservationType.JOINT_POS),
                            # ("q_right_shoulder1", "right_shoulder1", ObservationType.JOINT_POS),
                            # ("q_right_shoulder2", "right_shoulder2", ObservationType.JOINT_POS),
                            # ("q_right_elbow", "right_elbow", ObservationType.JOINT_POS),
                            # ("q_left_shoulder1", "left_shoulder1", ObservationType.JOINT_POS),
                            # ("q_left_shoulder2", "left_shoulder2", ObservationType.JOINT_POS),
                            # ("q_left_elbow", "left_elbow", ObservationType.JOINT_POS),

                            # ------------- JOINT VEL -------------

                            # ("dq_pelvis_ty", "pelvis_ty", ObservationType.JOINT_VEL),
                            # ("dq_pelvis_tz", "pelvis_tz", ObservationType.JOINT_VEL),
                            # ("dq_pelvis_tx", "pelvis_tx", ObservationType.JOINT_VEL),
                            ("dq_right_hip_x", "right_hip_x", ObservationType.JOINT_VEL),
                            ("dq_right_hip_z", "right_hip_z", ObservationType.JOINT_VEL),
                            ("dq_right_hip_y", "right_hip_y", ObservationType.JOINT_VEL),
                            ("dq_right_knee", "right_knee", ObservationType.JOINT_VEL),
                            ("dq_right_ankle_x", "right_ankle_x", ObservationType.JOINT_VEL),
                            ("dq_right_ankle_y", "right_ankle_y", ObservationType.JOINT_VEL),
                            ("dq_left_hip_x", "left_hip_x", ObservationType.JOINT_VEL),
                            ("dq_left_hip_z", "left_hip_z", ObservationType.JOINT_VEL),
                            ("dq_left_hip_y", "left_hip_y", ObservationType.JOINT_VEL),
                            ("dq_left_knee", "left_knee", ObservationType.JOINT_VEL),
                            ("dq_left_ankle_x", "left_ankle_x", ObservationType.JOINT_VEL),
                            ("dq_left_ankle_y", "left_ankle_y", ObservationType.JOINT_VEL),
                            # ("dq_right_shoulder1", "right_shoulder1", ObservationType.JOINT_VEL),
                            # ("dq_right_shoulder2", "right_shoulder2", ObservationType.JOINT_VEL),
                            # ("dq_right_elbow", "right_elbow", ObservationType.JOINT_VEL),
                            # ("dq_left_shoulder1", "left_shoulder1", ObservationType.JOINT_VEL),
                            # ("dq_left_shoulder2", "left_shoulder2", ObservationType.JOINT_VEL),
                            # ("dq_left_elbow", "left_elbow", ObservationType.JOINT_VEL),
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

        # action_spec = ["pelvis_ty_actuator","pelvis_tz_actuator","pelvis_tx_actuator",
        #         "right_hip_x_actuator","right_hip_z_actuator","right_hip_y_actuator","right_knee_actuator",
        #         "right_ankle_x_actuator","right_ankle_y_actuator","left_hip_x_actuator","left_hip_z_actuator","left_hip_y_actuator",
        #         "left_knee_actuator","left_ankle_x_actuator","left_ankle_y_actuator",
        #         "right_shoulder1_actuator","right_shoulder2_actuator","right_elbow_actuator",
        #         "left_shoulder1_actuator","left_shoulder2_actuator","left_elbow_actuator"]
        action_spec = [
                "right_hip_x_motor","right_hip_z_motor","right_hip_y_motor","right_knee_motor",
                "right_ankle_x_motor","right_ankle_y_motor","left_hip_x_motor","left_hip_z_motor","left_hip_y_motor",
                "left_knee_motor","left_ankle_x_motor","left_ankle_y_motor"
        ]
                
        return action_spec
    #---------------------------------------------------------------------------------------------------------

    def _get_ground_forces(self):
        """
        Returns the ground forces (np.array). By default, 4 ground force sensors are used.
        Environments that use more or less have to override this function.

        """

        grf = np.concatenate([self._get_collision_force("floor", "foot_r")[:3],
                              self._get_collision_force("floor", "foot_l")[:3]])

        return grf

    @staticmethod
    def _get_grf_size():
        """
        Returns the size of the ground force vector.

        """

        return 6
    
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
        check_validity_task_mode_dataset(StickFigureA3.__name__, task, None, dataset_type,
                                         *StickFigureA3.valid_task_confs.get_all())
        if dataset_type == "real":
            if task == "run":
                path = "datasets/humanoids/real/random_stick.npz"
            else:
                path = "datasets/humanoids/real/random_stick.npz"
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

        return BaseHumanoidRobot.generate(StickFigureA3, path, task, dataset_type,
                                          clip_trajectory_to_joint_ranges=True, **kwargs)
