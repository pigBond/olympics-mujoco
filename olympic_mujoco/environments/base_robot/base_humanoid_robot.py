import os.path
import warnings
from pathlib import Path
from copy import deepcopy

from mushroom_rl.utils.running_stats import *

import olympic_mujoco
from olympic_mujoco.environments import LocoEnvBase
from olympic_mujoco.enums.enums import AlgorithmType


class BaseHumanoidRobot(LocoEnvBase):
    """
    Base Class for the Mujoco simulation of real robots

    """

    def create_dataset(self, ignore_keys=None):
        """
        Creates a dataset from the specified trajectories.

        Args:
            ignore_keys (list): List of keys to ignore in the dataset. Default is ["q_pelvis_tx", "q_pelvis_tz"].

        Returns:
            Dictionary containing states, next_states and absorbing flags. 
            For the states the shape is
            (N_traj x N_samples_per_traj, dim_state), 
            while the absorbing flag has the shape is
            (N_traj x N_samples_per_traj).

        """

        if ignore_keys is None:
            ignore_keys = ["q_pelvis_tx", "q_pelvis_tz"]

        dataset = super().create_dataset(ignore_keys)

        return dataset
    
    def get_mask(self, obs_to_hide):
        """
        This function returns a boolean mask to hide observations from a fully observable state.
            To hide certain information when processing robot status.

        Args:
            obs_to_hide (tuple): A tuple of strings with names of objects to hide.
            Hidable objects are "positions", "velocities", "foot_forces", and "env_type".

        Returns:
            Mask in form of a np.array of booleans. True means that that the obs should be
            included, and False means that it should be discarded.

        """

        # If obs_to_hide is passed as a string, convert it into a tuple with one element.
        if type(obs_to_hide) == str:
            obs_to_hide = (obs_to_hide,)
        # Check if all elements in obs_to_hide are valid observations that can be hidden.
        assert all(x in self._hidable_obs for x in obs_to_hide), "Some of the observations you want to hide are not" \
                                                                 "supported. Valid observations to hide are %s." \
                                                                 % (self._hidable_obs,)
        # Get the dimensions for position and velocity observations.
        pos_dim, vel_dim = self._len_qpos_qvel()
        # Get the size of the ground reaction force (GRF) if applicable.
        force_dim = self._get_grf_size() # Fixed to 12

        mask = []
        if "positions" not in obs_to_hide:
            mask += [np.ones(pos_dim, dtype=np.bool)]
        else:
            mask += [np.zeros(pos_dim, dtype=np.bool)]

        if "velocities" not in obs_to_hide:
            mask += [np.ones(vel_dim, dtype=np.bool)]
        else:
            mask += [np.zeros(vel_dim, dtype=np.bool)]

        if self._use_foot_forces:
            if "foot_forces" not in obs_to_hide:
                mask += [np.ones(force_dim, dtype=np.bool)]
            else:
                mask += [np.zeros(force_dim, dtype=np.bool)]
        else:
            # Ensure that "foot_forces" is not in obs_to_hide since it's not being used.
            assert "foot_forces" not in obs_to_hide, "Creating a mask to hide foot forces without activating " \
                                                     "the latter is not allowed."


        return np.concatenate(mask).ravel()

    def _get_observation_space(self):
        """
        Returns a tuple of the lows and highs (np.array) of the observation space.
            The self._valid_weights array contains the valid weights, 
            [self._valid_weights[0]] and [self._valid_weights[-1]] are used to get the minimum and maximum valid weights, respectively. 
            These are then concatenated to the existing low and high arrays to define the new observation space for the robot.

        """

        low, high = super(BaseHumanoidRobot, self)._get_observation_space()

        return low, high
    
    def _create_observation(self, obs):
        """
        Creates a full vector of observations.

        Args:
            obs (np.array): Observation vector to be modified or extended;

        Returns:
            New observation vector (np.array).

        """

        obs = super(BaseHumanoidRobot, self)._create_observation(obs)

        return obs
    
    
    @staticmethod
    def generate(env, path, task="walk", dataset_type="real", debug=False,
                 clip_trajectory_to_joint_ranges=False, **kwargs):
        """
        Returns an environment corresponding to the specified task.
        This static method is used to create a Markov Decision Process (MDP) for a humanoid robot environment with a specific task and dataset type.

        Args:
            env (class): Humanoid class, real robots
            path (str): Path to the dataset.
            task (str): Main task to solve. Either "walk" or "carry". The latter is walking while carrying
                an unknown weight, which makes the task partially observable.
            dataset_type (str): "real" or "perfect". "real" uses real motion capture data as the
                reference trajectory. This data does not perfectly match the kinematics
                and dynamics of this environment, hence it is more challenging. "perfect" uses
                a perfect dataset.
            debug (bool): If True, the smaller test datasets are used for debugging purposes.
            clip_trajectory_to_joint_ranges (bool): If True, trajectory is clipped to joint ranges.

        Returns:
            An MDP of the Robot.

        """

        # Generate the MDP based on the specified task
        # TODO：这里实际上是触发real_robot层的__init__函数
        if task == "walk":
            reward_params = dict(target_velocity=1.25)
            mdp = env(reward_type="target_velocity", reward_params=reward_params, **kwargs)
        elif task == "run":
            reward_params = dict(target_velocity=2.5)
            mdp = env(reward_type="target_velocity", reward_params=reward_params, **kwargs)


        # Load the trajectory
        # Calculate the environment and desired control frequencies
        env_freq = 1 / mdp._timestep  # hz
        desired_contr_freq = 1 / mdp.dt  # hz
        n_substeps = env_freq // desired_contr_freq

        if dataset_type == "real":
            traj_data_freq = 500  # hz
            # Check if there is a dataset, if not found, use the use_mini_dataset flag
            use_mini_dataset = not os.path.exists(Path(olympic_mujoco.__file__).resolve().parent / path)
            if debug or use_mini_dataset:
                if use_mini_dataset:
                    warnings.warn("Datasets not found, falling back to test datasets. Please download and install "
                                  "the datasets to use this environment for imitation learning!")
                # Modify the path to use a small dataset
                path = path.split("/")
                path.insert(3, "mini_datasets")
                path = "/".join(path)
            '''
                设置轨迹参数
                traj_path 轨迹路径
                traj_dt 轨迹数据的时间间隔
                control_dt 控制数据的时间间隔
                clip_trajectory_to_joint_ranges 是否将轨迹剪辑到关节范围
            '''
            traj_params = dict(traj_path=Path(olympic_mujoco.__file__).resolve().parent / path,
                               traj_dt=(1 / traj_data_freq),
                               control_dt=(1 / desired_contr_freq),
                               clip_trajectory_to_joint_ranges=clip_trajectory_to_joint_ranges)

        elif dataset_type == "perfect":
            traj_data_freq = 100  # hz
            traj_files = mdp.load_dataset_and_get_traj_files(path, traj_data_freq)
            traj_params = dict(traj_files=traj_files,
                               traj_dt=(1 / traj_data_freq),
                               control_dt=(1 / desired_contr_freq),
                               clip_trajectory_to_joint_ranges=clip_trajectory_to_joint_ranges)

        elif dataset_type == "preference":
            traj_data_freq = 100  # hz
            infos = []
            # Get all files under the specified path
            all_paths = next(os.walk(Path(olympic_mujoco.__file__).resolve().parent / path), (None, None, []))[2]
            for i, p in enumerate(all_paths):
                # 加载每个数据集并获取轨迹文件
                # Load each dataset and obtain trajectory files
                traj_files = mdp.load_dataset_and_get_traj_files(path + p, traj_data_freq)
                # 对于第一个文件，直接使用轨迹文件
                # For the first file, use the trajectory file directly
                if i == 0:
                    all_traj_files = traj_files
                else:
                    # 对于其他文件，将它们的轨迹数据合并到all_traj_files中
                    # For other files, merge their trajectory data into all_traj_files
                    for key in traj_files.keys():
                        # For segmentation points, special handling is required to avoid duplication
                        # 分割点
                        if key == "split_points":
                            all_traj_files[key] = np.concatenate([all_traj_files[key],
                                                                  traj_files[key][1:] + all_traj_files[key][-1]])
                        else:
                            all_traj_files[key] = np.concatenate([all_traj_files[key], traj_files[key]])
                # Extracting information from file names
                info = p.split(".")[0]
                info = info.split("_")[-2]
                n_traj = len(traj_files["split_points"]) - 1
                infos += [info] * n_traj

            traj_params = dict(traj_files=all_traj_files,
                               traj_dt=(1 / traj_data_freq),
                               traj_info = infos,
                               control_dt=(1 / desired_contr_freq),
                               clip_trajectory_to_joint_ranges=clip_trajectory_to_joint_ranges)


        mdp.load_trajectory(traj_params, warn=False)

        return mdp
    
    #---------------------------------------------------------------------------------------------------------
    #----------------------------------------- 奖励函数 ------------------------------------------------------
    #---------------------------------------------------------------------------------------------------------


    #---------------------------------------------------------------------------------------------------------

    #---------------------------------------------------------------------------------------------------------
    #----------------------------------------- 吸收终止函数 --------------------------------------------------
    #---------------------------------------------------------------------------------------------------------

    def _has_fallen(self, obs, return_err_msg=False):
        raise NotImplementedError
    def _has_done(self):
        raise NotImplementedError

    def is_absorbing(self, obs):
        """
        Checks if an observation is an absorbing state or not.

        Args:
            obs (np.array): Current observation;

        Returns:
            True, if the observation is an absorbing state; otherwise False;

        """
        if self._algorithm_type == AlgorithmType.REINFORCEMENT_LEARNING:
            return self._has_done()
        elif self._algorithm_type == AlgorithmType.IMITATION_LEARNING:
            return self._has_fallen(obs) if self._use_absorbing_states else False

    #---------------------------------------------------------------------------------------------------------
