from pathlib import Path
from copy import deepcopy
from dm_control import mjcf

from mushroom_rl.utils.running_stats import *
from mushroom_rl.utils.mujoco import *

from olympic_mujoco.environments.base_robot.base_humanoid_robot import BaseHumanoidRobot
from olympic_mujoco.utils import check_validity_task_mode_dataset
from olympic_mujoco.environments.loco_env_base import ValidTaskConf

class StickFigureA1(BaseHumanoidRobot):

    valid_task_confs = ValidTaskConf(tasks=["walk", "run","test"],
                                     data_types=["real", "perfect"]
                                    )

    def __init__(self, disable_arms=True, disable_back_joint=False, **kwargs):
        """
        Constructor.

        """
        print("StickFigureA1")

        xml_path = (Path(__file__).resolve().parent.parent / "data" / "stickFigure_A1" / "a1.xml").as_posix()

        action_spec = self._get_action_specification()

        observation_spec = self._get_observation_specification()

        collision_groups = [("floor", ["floor"]),
                            ("foot_r", ["right_right_foot"]),
                            ("foot_l", ["left_left_foot"])]

        self._hidable_obs = ("positions", "velocities", "foot_forces", "weight")

        # --- Modify the xml, the action_spec, and the observation_spec if needed ---
        self._disable_arms = disable_arms
        self._disable_back_joint = disable_back_joint

        xml_handles = mjcf.from_path(xml_path)

        super().__init__(xml_handles, action_spec, observation_spec, collision_groups, **kwargs)

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
            # joints_to_remove += ["l_arm_shy", "l_arm_shx", "l_arm_shz", "left_elbow", "r_arm_shy",
            #                      "r_arm_shx", "r_arm_shz", "right_elbow"]
            # motors_to_remove += ["l_arm_shy_actuator", "l_arm_shx_actuator", "l_arm_shz_actuator",
            #                      "left_elbow_actuator", "r_arm_shy_actuator", "r_arm_shx_actuator",
            #                      "r_arm_shz_actuator", "right_elbow_actuator"]
            joints_to_remove += ["left_shoulder1","left_shoulder2","left_elbow",
                                 "right_shoulder1","right_shoulder2","right_elbow"]
            motors_to_remove += ["left_shoulder1_actuator","left_shoulder2_actuator","left_elbow_actuator",
                                 "right_shoulder1_actuator","right_shoulder2_actuator","right_elbow_actuator"]
        if self._disable_back_joint:
            # TODO:这里存在问题
            print()
            # joints_to_remove += ["back_bkz"]
            # motors_to_remove += ["back_bkz_actuator"]

        return joints_to_remove, motors_to_remove, equ_constr_to_remove
    

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
        check_validity_task_mode_dataset(StickFigureA1.__name__, task, None, dataset_type,
                                         *StickFigureA1.valid_task_confs.get_all())
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

        return BaseHumanoidRobot.generate(StickFigureA1, path, task, dataset_type,
                                          clip_trajectory_to_joint_ranges=True, **kwargs)

    @staticmethod
    def _get_observation_specification():
        """
        Getter for the observation space specification.

        Returns:
            A list of tuples containing the specification of each observation
            space entry.

        """

        observation_spec = [# ------------- JOINT POS -------------
                            ("q_pelvis_ty", "pelvis_ty", ObservationType.JOINT_POS),
                            ("q_pelvis_tz", "pelvis_tz", ObservationType.JOINT_POS),
                            ("q_pelvis_tx", "pelvis_tx", ObservationType.JOINT_POS),
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
                            ("q_right_shoulder1", "right_shoulder1", ObservationType.JOINT_POS),
                            ("q_right_shoulder2", "right_shoulder2", ObservationType.JOINT_POS),
                            ("q_right_elbow", "right_elbow", ObservationType.JOINT_POS),
                            ("q_left_shoulder1", "left_shoulder1", ObservationType.JOINT_POS),
                            ("q_left_shoulder2", "left_shoulder2", ObservationType.JOINT_POS),
                            ("q_left_elbow", "left_elbow", ObservationType.JOINT_POS),

                            # ------------- JOINT VEL -------------

                            ("dq_pelvis_ty", "pelvis_ty", ObservationType.JOINT_VEL),
                            ("dq_pelvis_tz", "pelvis_tz", ObservationType.JOINT_VEL),
                            ("dq_pelvis_tx", "pelvis_tx", ObservationType.JOINT_VEL),
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
                            ("dq_right_shoulder1", "right_shoulder1", ObservationType.JOINT_VEL),
                            ("dq_right_shoulder2", "right_shoulder2", ObservationType.JOINT_VEL),
                            ("dq_right_elbow", "right_elbow", ObservationType.JOINT_VEL),
                            ("dq_left_shoulder1", "left_shoulder1", ObservationType.JOINT_VEL),
                            ("dq_left_shoulder2", "left_shoulder2", ObservationType.JOINT_VEL),
                            ("dq_left_elbow", "left_elbow", ObservationType.JOINT_VEL),
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

        action_spec = ["pelvis_ty_actuator","pelvis_tz_actuator","pelvis_tx_actuator",
                "right_hip_x_actuator","right_hip_z_actuator","right_hip_y_actuator","right_knee_actuator",
                "right_ankle_x_actuator","right_ankle_y_actuator","left_hip_x_actuator","left_hip_z_actuator","left_hip_y_actuator",
                "left_knee_actuator","left_ankle_x_actuator","left_ankle_y_actuator",
                "right_shoulder1_actuator","right_shoulder2_actuator","right_elbow_actuator",
                "left_shoulder1_actuator","left_shoulder2_actuator","left_elbow_actuator"]

        return action_spec