from pathlib import Path
from copy import deepcopy
from dm_control import mjcf

from mushroom_rl.utils.running_stats import *
from mushroom_rl.utils.mujoco import *

from olympic_mujoco.environments.base_robot.base_humanoid_robot import BaseHumanoidRobot
from olympic_mujoco.utils import check_validity_task_mode_dataset
from olympic_mujoco.environments.loco_env_base import ValidTaskConf


class Jvrc(BaseHumanoidRobot):

    valid_task_confs = ValidTaskConf(
        tasks=["walk", "run", "test"], data_types=["real", "perfect"]
    )

    def __init__(self, disable_arms=True, disable_back_joint=False, **kwargs):
        """
        Constructor.

        """
        print("Jvrc init")

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

        super().__init__(
            xml_handles, action_spec, observation_spec, collision_groups, **kwargs
        )

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
            joints_to_remove += [
                # waist
                "WAIST_Y",
                "WAIST_P",
                "WAIST_R",
                # right arm
                "R_SHOULDER_P",
                "R_SHOULDER_R",
                "R_SHOULDER_Y",
                "R_ELBOW_P",
                "R_ELBOW_Y",
                "R_WRIST_R",
                "R_WRIST_Y",
                "R_UTHUMB",
                # left arm
                "L_SHOULDER_P",
                "L_SHOULDER_R",
                "L_SHOULDER_Y",
                "L_ELBOW_P",
                "L_ELBOW_Y",
                "L_WRIST_R",
                "L_WRIST_Y",
                "L_UTHUMB",
            ]
            motors_to_remove += [
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
        check_validity_task_mode_dataset(
            Jvrc.__name__, task, None, dataset_type, *Jvrc.valid_task_confs.get_all()
        )
        if dataset_type == "real":
            if task == "run":
                path = "datasets/humanoids/real/random_stick_jvrc.npz"
            else:
                path = "datasets/humanoids/real/random_stick_jvrc.npz"
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
            ("q_R_SHOULDER_R", "R_SHOULDER_R", ObservationType.JOINT_POS),
            ("q_R_SHOULDER_Y", "R_SHOULDER_Y", ObservationType.JOINT_POS),
            ("q_R_SHOULDER_P", "R_SHOULDER_P", ObservationType.JOINT_POS),
            ("q_R_ELBOW_P", "R_ELBOW_P", ObservationType.JOINT_POS),
            ("q_R_ELBOW_Y", "R_ELBOW_Y", ObservationType.JOINT_POS),
            ("q_L_SHOULDER_R", "L_SHOULDER_R", ObservationType.JOINT_POS),
            ("q_L_SHOULDER_Y", "L_SHOULDER_Y", ObservationType.JOINT_POS),
            ("q_L_SHOULDER_P", "L_SHOULDER_P", ObservationType.JOINT_POS),
            ("q_L_ELBOW_P", "L_ELBOW_P", ObservationType.JOINT_POS),
            ("q_L_ELBOW_Y", "L_ELBOW_Y", ObservationType.JOINT_POS),
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
            ("dq_R_SHOULDER_R", "R_SHOULDER_R", ObservationType.JOINT_VEL),
            ("dq_R_SHOULDER_Y", "R_SHOULDER_Y", ObservationType.JOINT_VEL),
            ("dq_R_SHOULDER_P", "R_SHOULDER_P", ObservationType.JOINT_VEL),
            ("dq_R_ELBOW_P", "R_ELBOW_P", ObservationType.JOINT_VEL),
            ("dq_R_ELBOW_Y", "R_ELBOW_Y", ObservationType.JOINT_VEL),
            ("dq_L_SHOULDER_R", "L_SHOULDER_R", ObservationType.JOINT_VEL),
            ("dq_L_SHOULDER_Y", "L_SHOULDER_Y", ObservationType.JOINT_VEL),
            ("dq_L_SHOULDER_P", "L_SHOULDER_P", ObservationType.JOINT_VEL),
            ("dq_L_ELBOW_P", "L_ELBOW_P", ObservationType.JOINT_VEL),
            ("dq_L_ELBOW_Y", "L_ELBOW_Y", ObservationType.JOINT_VEL),
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

        return action_spec
