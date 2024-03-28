import warnings
from copy import deepcopy

import numpy as np
from scipy import interpolate


class Trajectory:
    """
    通用类
    用于处理轨迹数据。它从一个numpy二进制文件(.npy)构建一般轨迹，
    并自动将轨迹插值到期望的控制频率。该类用于生成数据集，以及从数据集中采样以初始化模拟。
    所有的轨迹都要求具有相等的长度。
    """

    def __init__(
        self,
        keys,
        low,
        high,
        joint_pos_idx,
        interpolate_map,
        interpolate_remap,
        traj_path=None,
        traj_files=None,
        interpolate_map_params=None,
        interpolate_remap_params=None,
        traj_dt=0.002,
        control_dt=0.01,
        ignore_keys=None,
        clip_trajectory_to_joint_ranges=False,
        traj_info=None,
        warn=True,
    ):
        """
        Constructor.

        Args:
        keys (list): 要从轨迹中提取的数据的键的列表。
        low (np.array): 轨迹值的下界。
        high (np.array): 轨迹值的上界。
        joint_pos_idx (np.array): 包含轨迹中所有关节位置索引的数组。
        interpolate_map (func): 用于将轨迹映射到可以进行插值的空间的函数。
        interpolate_remap (func): 用于在插值后将变换后的轨迹映射回原始空间的函数。
        traj_path (str, optional): 轨迹数据的路径,应该是一个numpy压缩文件(.npz)。默认为None。
        traj_files (dict, optional): 包含所有轨迹文件的字典。如果traj_path被指定,这个参数应该为None。默认为None。
        interpolate_map_params (_type_, optional): 进行插值操作所需的参数集。默认为None。
        interpolate_remap_params (_type_, optional): 进行插值后的逆映射操作所需的参数集。默认为None。
        traj_dt (float, optional): 轨迹文件的时间步长。默认为0.002。
        control_dt (float, optional): 模型控制频率,用于插值轨迹。默认为0.01。
        ignore_keys (list, optional): 要在数据集中忽略的键的列表。默认为None。
        clip_trajectory_to_joint_ranges (bool, optional): 如果为True,则将轨迹中的关节位置裁剪到轨迹的low和high值之间。默认为False。
        traj_info (list, optional): 每个轨迹的自定义标签的列表。默认为None。
        warn (bool, optional): 如果为True,当轨迹范围被违反时将发出警告。默认为True。
        """
        # 断言确保 traj_path 和 traj_files 不会同时被指定
        assert (traj_path is not None) != (traj_files is not None), (
            "Please specify either traj_path or " "traj_files, but not both."
        )
        # load data
        if traj_path is not None:
            self._trajectory_files = np.load(traj_path, allow_pickle=True)
        else:
            self._trajectory_files = traj_files

        # 转换为可变字典
        self._trajectory_files = {k: d for k, d in self._trajectory_files.items()}
        # 检查轨迹是否在范围内
        self.check_if_trajectory_is_in_range(
            low, high, keys, joint_pos_idx, warn, clip_trajectory_to_joint_ranges
        )

        # add all goals to keys (goals have to start with 'goal' if not in keys)
        keys += [
            key
            for key in self._trajectory_files.keys()
            if key.startswith("goal") and key not in keys
        ]
        self.keys = keys

        # 删除不需要的键
        if ignore_keys is not None:
            for ik in ignore_keys:
                keys.remove(ik)
        # 分割点标记下一个轨迹的开始。
        # 最后一个分割点指向轨迹中最后一个元素的索引 -> len(traj)
        if "split_points" in self._trajectory_files.keys():
            self.split_points = self._trajectory_files["split_points"]
        else:
            self.split_points = np.array(
                [0, len(list(self._trajectory_files.values())[0])]
            )
        # 从文件中提取轨迹。返回一个np.array列表。列表的长度是观察数的数量。
        # 每个np.array的形状为（n_trajectories, n_samples, (dim_observation)）。
        # 如果dim_observation为1，则数组的形状仅为（n_trajectories, n_samples）。
        self.trajectories = self._extract_trajectory_from_files()

        # 如果提供了轨迹信息，检查其数量是否与轨迹数量匹配
        if traj_info is not None:
            assert (
                len(traj_info) == self.number_of_trajectories
            ), "The number of trajectory infos/labels need to be equal to the number of trajectories."
        # 保存轨迹信息
        self._traj_info = traj_info
        # 设置时间步长参数
        self.traj_dt = traj_dt
        self.control_dt = control_dt

        # 如果轨迹的时间步长与控制频率不同，则插值轨迹
        # interpolate_map：映射函数，用于将原始轨迹映射到可以进行插值的空间
        # interpolate_map_params：映射函数所需的参数集，用于辅助映射过程的执行
        # interpolate_remap：逆映射函数，用于将插值后的轨迹数据映射回原始空间
        # interpolate_remap_params：逆映射函数所需的参数集，用于辅助逆映射过程的执行
        if self.traj_dt != control_dt:
            self._interpolate_trajectories(
                map_funct=interpolate_map,
                map_params=interpolate_map_params,
                re_map_funct=interpolate_remap,
                re_map_params=interpolate_remap_params,
            )

        # 初始化子轨迹步数和轨迹编号
        self.subtraj_step_no = 0
        self.traj_no = 0

        # 获取当前轨迹的子轨迹
        self.subtraj = self._get_subtraj(self.traj_no)

    def create_dataset(
        self, ignore_keys=None, state_callback=None, state_callback_params=None
    ):
        """
        创建一个供模仿学习算法使用的数据集。
        Args:
            ignore_keys (list): 在数据集中需要忽略的键列表。
            state_callback (func): 应在每个状态下调用的函数。
            state_callback_params (dict): 用于执行状态转换所需的参数字典。
        Returns:
            包含状态、下一个状态、吸收标志和最后标志的字典。
            状态的形状为 (N_traj x N_samples_per_traj-1, dim_state)，
            而标志的形状为 (N_traj x N_samples_per_traj-1)。
            如果指定了traj_info，它也将包含这些信息。
        """
        # 获取平展后的轨迹数据
        flat_traj = self.flattened_trajectories()
        # 创建一个字典，并提取除了ignore_keys中指定的元素之外的所有元素
        all_data = dict(zip(self.keys, deepcopy(list(flat_traj))))
        if ignore_keys is not None:
            # 如果指定了需要忽略的键，则从字典中删除这些键
            for ikey in ignore_keys:
                del all_data[ikey]
        # 获取字典中的所有轨迹数据
        traj = list(all_data.values())
        # 创建一个状态数组，形状为 (n_states, dim_obs)
        states = np.concatenate(traj, axis=1)
        if state_callback is not None:
            transformed_states = []
            for state in states:
                # 使用状态回调函数和提供的参数转换每个状态
                transformed_states.append(
                    state_callback(state, **state_callback_params)
                )
            states = np.array(transformed_states)
        # 将状态转换为包含状态和下一个状态的字典
        new_states = states[:-1]  # 当前状态
        new_next_states = states[1:]  # 下一状态
        # 创建一个吸收标志数组，默认假设轨迹中没有吸收状态
        absorbing = np.zeros(len(states[:-1]))
        # 创建一个标志数组，标识每个轨迹的最后一个样本
        last = np.zeros(len(states))
        # 在每个子轨迹的最后一个样本位置设置1
        last[self.split_points[1:] - 1] = np.ones(len(self.split_points) - 1)
        if self._traj_info is not None:
            # 如果有轨迹信息，则将其重塑为一个与状态数组相同长度的数组
            info = np.array(
                [[l] * self.trajectory_length for l in self._traj_info]
            ).reshape(-1)
            # 返回包含状态、下一个状态、吸收标志、最后标志和轨迹信息的字典
            return dict(
                states=new_states,
                next_states=new_next_states,
                absorbing=absorbing,
                last=last,
                info=info,
            )
        else:
            # 如果没有轨迹信息，只返回包含状态、下一个状态、吸收标志和最后标志的字典
            return dict(
                states=new_states,
                next_states=new_next_states,
                absorbing=absorbing,
                last=last,
            )

    def _extract_trajectory_from_files(self):
        """
        从轨迹文件中通过筛选相关键提取轨迹。
        然后使用分割点将轨迹分为多个轨迹。

        Returns:
            一个包含 np.array 的列表。列表的长度是观察次数。
            每个np.array的形状为 (n_trajectories, n_samples, (dim_observation))。
            如果 dim_observation 为 1,则数组的形状仅为 (n_trajectories, n_samples)。
        """
        # load data of relevant keys
        trajectories = [self._trajectory_files[key] for key in self.keys]
        # 检查所有观察是否长度相等
        len_obs = np.array([len(obs) for obs in trajectories])
        # 确保所有观察长度一致，否则抛出异常
        assert np.all(len_obs == len_obs[0]), (
            "Some observations have different lengths than others. "
            "Trajectory is corrupted. "
        )

        # 使用分割点将轨迹分割成多个轨迹
        for i in range(len(trajectories)):
            # 对当前轨迹按分割点进行切分
            trajectories[i] = np.split(trajectories[i], self.split_points[1:-1])
            # 检查所有轨迹是否长度相等
            len_trajectories = np.array([len(traj) for traj in trajectories[i]])
            # 确保所有轨迹长度相同，否则抛出异常
            assert np.all(len_trajectories == len_trajectories[0]), (
                "Only trajectories of equal length " "are currently supported."
            )

            trajectories[i] = np.array(trajectories[i])
        # 返回轨迹列表
        return trajectories

    def _interpolate_trajectories(
        self, map_funct, re_map_funct, map_params, re_map_params
    ):
        """
        对所有轨迹进行三次样条法插值。

        Args:
        map_funct (func): 用于将轨迹映射到某个允许插值的空间的函数。
        re_map_funct (func): 在插值后，将转换后的轨迹映射回原始空间的函数。
        map_params: 进行插值的相应环境所需的参数集合。
        re_map_params: 进行插值的相应环境所需的参数集合。
        """
        # 确保映射函数和逆映射函数要么都有值，要么都为None
        assert (map_funct is None) == (re_map_funct is None)

        new_trajs = list()  # 创建一个新的列表，用于存储插值后的轨迹
        # 对每个轨迹进行插值
        for i in range(self.number_of_trajectories):
            traj = [obs[i] for obs in self.trajectories]  # 从原始轨迹中提取第i个轨迹
            x = np.arange(self.trajectory_length)  # 创建一个数组，表示原始轨迹的索引
            new_traj_sampling_factor = (
                self.traj_dt / self.control_dt
            )  # 计算新的轨迹采样因子
            x_new = np.linspace(
                0,
                self.trajectory_length - 1,
                round(self.trajectory_length * new_traj_sampling_factor),
                endpoint=True,
            )  # 创建新的插值点
            # 预处理轨迹
            traj = (
                map_funct(traj) if map_params is None else map_funct(traj, **map_params)
            )
            # 使用三次样条插值方法对轨迹进行插值
            new_traj = interpolate.interp1d(x, traj, kind="cubic", axis=1)(x_new)
            # 后处理轨迹
            new_traj = (
                re_map_funct(new_traj)
                if re_map_params is None
                else re_map_funct(new_traj, **re_map_params)
            )
            new_trajs.append(new_traj)  # 将插值后的轨迹添加到列表中
        # 将插值后的轨迹转换回原始形状
        trajectories = []
        for i in range(self.number_obs_trajectory):
            trajectories.append([])  # 为每个观察轨迹创建一个新的子列表
            for traj in new_trajs:
                trajectories[i].append(traj[i])  # 将插值后的轨迹按原始形状重新组织
            trajectories[i] = np.array(trajectories[i])  # 将列表转换为NumPy数组
        self.trajectories = trajectories  # 更新类的轨迹属性
        # 对分割点进行插值
        self.split_points = [0]  # 重置分割点列表，首先添加起始点0
        for k in range(self.number_of_trajectories):
            self.split_points.append(
                self.split_points[-1] + len(self.trajectories[0][k])
            )
            # 根据插值后的轨迹长度更新分割点
        self.split_points = np.array(self.split_points)  # 将分割点列表转换为NumPy数组

    def reset_trajectory(self, substep_no=None, traj_no=None):
        """
        重置轨迹到一个特定的轨迹以及该轨迹内的一个子步骤。如果其中之一为None,
        则它们会被随机设置。

        Args:
            substep_no (int, None): 轨迹的起始点。
            如果为None,则从轨迹的随机点开始。
            traj_no (int, None): 要开始的轨迹的编号。
            如果为None,则从随机轨迹开始。
        Returns:
            从轨迹中选择的样本（或随机采样的样本）。
        """
        # 如果没有提供轨迹编号，则从轨迹总数范围内随机选择一个
        if traj_no is None:
            self.traj_no = np.random.randint(0, self.number_of_trajectories)
        else:
            # 如果提供了轨迹编号，确保它在有效范围内
            assert 0 <= traj_no <= self.number_of_trajectories
            self.traj_no = traj_no
        # 如果没有提供子步骤编号，则从轨迹长度范围内随机选择一个
        if substep_no is None:
            self.subtraj_step_no = np.random.randint(0, self.trajectory_length)
        else:
            # 如果提供了子步骤编号，确保它在有效范围内
            assert 0 <= substep_no <= self.trajectory_length
            self.subtraj_step_no = substep_no
        # 选择一个子轨迹
        self.subtraj = self._get_subtraj(self.traj_no)
        # 将子轨迹的x和y值重置到中间位置
        self.subtraj[0] -= self.subtraj[0][self.subtraj_step_no]
        self.subtraj[1] -= self.subtraj[1][self.subtraj_step_no]
        # 从子轨迹中获取当前步骤的样本
        sample = [obs[self.subtraj_step_no] for obs in self.subtraj]
        return sample  # 返回样本

    def check_if_trajectory_is_in_range(
        self, low, high, keys, j_idx, warn, clip_trajectory_to_joint_ranges
    ):
        # 如果需要警告或裁剪轨迹到关节范围
        if warn or clip_trajectory_to_joint_ranges:
            # 获取关节位置（q_pos）的索引
            j_idx = j_idx[2:]  # 排除x和y坐标
            # 检查这些关节位置是否在指定范围内
            for i, item in enumerate(self._trajectory_files.items()):
                k, d = item  # 解构元组，获取键（关节名称）和值（关节轨迹数据）
                if i in j_idx:  # 如果当前索引在关节索引列表中
                    high_i = high[i - 2]  # 获取对应关节的最大范围
                    low_i = low[i - 2]  # 获取对应关节的最小范围
                    # 如果需要警告
                    if warn:
                        clip_message = (
                            "Clipping the trajectory into range!"
                            if clip_trajectory_to_joint_ranges
                            else ""
                        )
                        # 如果轨迹数据超过了最大范围，发出警告
                        if np.max(d) > high_i:
                            warnings.warn(
                                "Trajectory violates joint range in %s. Maximum in trajectory is %f "
                                "and maximum range is %f. %s"
                                % (keys[i], np.max(d), high_i, clip_message),
                                RuntimeWarning,
                            )
                        # 如果轨迹数据低于最小范围，发出警告
                        elif np.min(d) < low_i:
                            warnings.warn(
                                "Trajectory violates joint range in %s. Minimum in trajectory is %f "
                                "and minimum range is %f. %s"
                                % (keys[i], np.min(d), low_i, clip_message),
                                RuntimeWarning,
                            )
                    # 如果需要裁剪轨迹到最小和最大范围
                    if clip_trajectory_to_joint_ranges:
                        # 使用np.clip函数裁剪关节轨迹数据到指定范围
                        self._trajectory_files[k] = np.clip(
                            self._trajectory_files[k], low_i, high_i
                        )

    """
        get_current_sample: 返回当前轨迹中的样本。
        get_next_sample: 返回轨迹中的下一个样本,或到达末尾时返回None。
        get_from_sample: 从样本中提取指定键的数据。
        get_idx: 返回指定键的索引。
        flattened_trajectories: 返回所有轨迹展平后的数据。
        _get_subtraj: 返回指定索引处的轨迹副本。
        _get_ith_sample_from_subtraj: 返回当前子轨迹中指定索引的样本副本。
        number_obs_trajectory: 返回轨迹中的观察值数量。
        trajectory_length: 返回单个轨迹的长度。
        number_of_trajectories: 返回轨迹的总数量。
    """
    
    def get_current_sample(self):
        """
        Returns the current sample in the trajectory.

        """

        return self._get_ith_sample_from_subtraj(self.subtraj_step_no)

    def get_next_sample(self):
        """
        Returns the next sample in the trajectory.

        """

        self.subtraj_step_no += 1
        if self.subtraj_step_no == self.trajectory_length:
            sample = None
        else:
            sample = self._get_ith_sample_from_subtraj(self.subtraj_step_no)

        return sample

    def get_from_sample(self, sample, key):
        """
        Returns the part of the sample whose key is specified. In contrast to the
        function _get_from_obs from the base environment, this function also allows to
        access information that is in the trajectory, but not in the simulation such
        as goal definitions.

        Note: This function is not suited for getting an observation from environment samples!

        Args:
            sample (list or np.array): Current sample to extract an observation from.
            key (string): Name of the observation to extract from sample

        Returns:
            np.array consisting of the observation specified by the key.

        """
        assert len(sample) == len(self.keys)

        idx = self.get_idx(key)

        return sample[idx]

    def get_idx(self, key):
        """
        Returns the index of the key.

        Note: This function is not suited for getting the index for an observation of the environment!

        Args:
            key (string): Name of the observation to extract from sample

        Returns:
            int containing the desired index.

        """

        return self.keys.index(key)

    def flattened_trajectories(self):
        """
        Returns the trajectories flattened in the N_traj dimension. Also expands dim if obs has dimension 1.

        """
        trajectories = []
        for obs in self.trajectories:
            if len(obs.shape) == 2:
                trajectories.append(obs.reshape((-1, 1)))
            elif len(obs.shape) == 3:
                trajectories.append(obs.reshape((-1, obs.shape[2])))
            else:
                raise ValueError("Unsupported shape of observation %s." % obs.shape)

        return trajectories

    def _get_subtraj(self, i):
        """
        Returns a copy of the i-th trajectory included in trajectories.

        """

        return [obs[i].copy() for obs in self.trajectories]

    def _get_ith_sample_from_subtraj(self, i):
        """
        Returns a copy of the i-th sample included in the current subtraj.

        """

        return [np.array(obs[i].copy()).flatten() for obs in self.subtraj]

    @property
    def number_obs_trajectory(self):
        """
        Returns the number of observations in the trajectory.

        """
        return len(self.trajectories)

    @property
    def trajectory_length(self):
        """
        Returns the length of a trajectory. Note that all trajectories have to be equal in length.

        """
        return self.trajectories[0].shape[1]

    @property
    def number_of_trajectories(self):
        """
        Returns the number of trajectories.

        """
        return self.trajectories[0].shape[0]
