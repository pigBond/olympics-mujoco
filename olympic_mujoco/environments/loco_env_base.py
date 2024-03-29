import os

import warnings
from pathlib import Path
from copy import deepcopy
from tempfile import mkdtemp
from itertools import product

from dm_control import mjcf

from mushroom_rl.core import Environment
from mushroom_rl.environments import MultiMuJoCo
from mushroom_rl.utils import spaces
from mushroom_rl.utils.running_stats import *
from mushroom_rl.utils.mujoco import *
from mushroom_rl.utils.record import VideoRecorder

import mujoco
from olympic_mujoco.utils import Trajectory
from olympic_mujoco.utils import NoReward, CustomReward, TargetVelocityReward, PosReward


class LocoEnvBase(MultiMuJoCo):
    """
    Base class for all kinds of locomotion environments.

    """

    def __init__(
        self,
        xml_handles,
        action_spec,
        observation_spec,
        collision_groups=None,
        gamma=0.99,
        horizon=1000,
        n_substeps=10,
        reward_type=None,
        reward_params=None,
        traj_params=None,
        random_start=True,
        init_step_no=None,
        timestep=0.001,
        use_foot_forces=False,
        default_camera_mode="follow",
        use_absorbing_states=True,
        domain_randomization_config=None,
        parallel_dom_rand=True,
        N_worker_per_xml_dom_rand=4,
        **viewer_params
    ):
        """
        Constructor.

        Args:
        xml_handles : MuJoCo XML 文件句柄。
        action_spec (list): 指定应由代理控制的活动关节名称的列表。
                            当需要使用所有执行器时,可以留空。
        observation_spec (list): 包含应作为观察提供给代理的数据名称及其类型(ObservationType)的列表。
                                 它们与一个键组合在一起,该键用于访问数据。
                                 列表中的条目由：(键, 名称, 类型) 组成。
                                 An entry in the list is given by: (key, name, type).
                                 名称可用于稍后检索特定观察。
        collision_groups (list, None): 包含在模拟期间通过 `check_collision` 方法检查碰撞的几何组列表。
                                       条目形式为：(键, geom_names),
                                       The entries are given as: (key, geom_names)
                                       其中键是稍后在 "check_collision" 方法中引用的字符串,
                                       geom_names 是 XML 规范中的几何名称列表。
        gamma (float): 环境的折扣因子。
        horizon (int): 环境的最大时间步长。
        n_substeps (int): MuJoCo 模拟器使用的子步骤数。
                         代理给出的一个动作将在代理收到下一个观察并相应地采取行动之前应用 n_substeps。
        reward_type (string): 要使用的奖励函数类型。
        reward_params (dict): 对应所选奖励函数的参数字典。
        traj_params (dict): 用于构建轨迹的参数字典。
        random_start (bool): 如果为 True,则在每个时间步的开始处从轨迹中随机选择一个样本,
                             并根据该样本初始化模拟。
                             这需要传递 traj_params
        init_step_no (int): 如果设置,则从轨迹中取相应的样本来初始化模拟。
        timestep (float): MuJoCo 模拟器使用的时步长。
                          如果为 None,则使用 XML 中指定的默认时步长。
        use_foot_forces (bool): 如果为 True,计算足部力并将其添加到观察空间。
        default_camera_mode (str): 定义默认相机模式的字符串。
                                  可用模式有 "static"、"follow" 和 "top_static"。
        use_absorbing_states (bool): 如果为 True,则为每个环境定义吸收状态。
                                     这意味着情节可以提前终止。
        domain_randomization_config (str): 域/动力学随机化配置文件的路径。
        parallel_dom_rand (bool): 如果为 True 并且传递了 domain_randomization_config 文件,
                                  域随机化将以并行方式运行以加快模拟运行时间。
        N_worker_per_xml_dom_rand (int): 用于并行域随机化的每个 XML 文件的工人数量。
                                        如果并行设置为 True,则此数字必须大于 1。
        viewer_params: 其他视图参数。
        """
        print("locoEnv")

        if type(xml_handles) != list:
            xml_handles = [xml_handles]

        if collision_groups is None:
            collision_groups = list()

        if use_foot_forces:
            n_intermediate_steps = n_substeps
            n_substeps = 1
        else:
            n_intermediate_steps = 1

        if "geom_group_visualization_on_startup" not in viewer_params.keys():
            viewer_params["geom_group_visualization_on_startup"] = [
                0,
                2,
            ]  # enable robot geom [0] and floor visual [2]

        # if domain_randomization_config is not None:
        #     self._domain_rand = DomainRandomizationHandler(
        #         xml_handles,
        #         domain_randomization_config,
        #         parallel_dom_rand,
        #         N_worker_per_xml_dom_rand,
        #     )
        # else:
        #     self._domain_rand = None
        self._domain_rand = None

        super().__init__(
            xml_handles,
            action_spec,
            observation_spec,
            gamma=gamma,
            horizon=horizon,
            n_substeps=n_substeps,
            n_intermediate_steps=n_intermediate_steps,
            timestep=timestep,
            collision_groups=collision_groups,
            default_camera_mode=default_camera_mode,
            **viewer_params
        )

        # 指定奖励函数
        self._reward_function = self._get_reward_function(reward_type, reward_params)
        # 选择是否使用足部力在观察空间中
        self._use_foot_forces = use_foot_forces
        # 设置观察空间为使用_get_observation_space方法返回的Box空间
        self.info.observation_space = spaces.Box(*self._get_observation_space())
        # 动作空间应该在-1到1之间,因此需要对其进行归一化
        # 复制当前的动作空间的最小值和最大值
        low, high = (
            self.info.action_space.low.copy(),
            self.info.action_space.high.copy(),
        )
        # 计算归一化动作的平均值和差值
        self.norm_act_mean = (high + low) / 2.0  # 动作的平均值
        self.norm_act_delta = (high - low) / 2.0  # 动作的差值
        # 重新设置动作空间的最小值和最大值为-1和1
        self.info.action_space.low[:] = -1.0
        self.info.action_space.high[:] = 1.0
        # 为平均地面反作用力设置一个运行平均值窗口
        self.mean_grf = self._setup_ground_force_statistics()

        self._dataset = None

        if traj_params:
            self.trajectories = None
            # 加载给定的轨迹参数
            self.load_trajectory(traj_params)
        else:
            self.trajectories = None
            # 设置是否从随机位置开始
            self._random_start = random_start
            # 设置初始步骤编号
            self._init_step_no = init_step_no
            # 设置是否使用吸收状态（在某些强化学习任务中,用于指示一个episode的结束）
            self._use_absorbing_states = use_absorbing_states

    def load_trajectory(self, traj_params, warn=True):
        """
        加载轨迹。如果已经加载了轨迹,这个函数将覆盖之前的轨迹。

        Args:
        traj_params (dict): 加载轨迹所需的参数字典。
        warn (bool): 如果为True,当轨迹范围被违反时,将发出警告。

        该方法首先检查是否存在已加载的轨迹,如果存在,则发出警告并覆盖之前的轨迹。
        接着使用提供的参数和内部方法创建一个新的Trajectory对象。
        """
        # 检查是否已经有轨迹被加载
        if self.trajectories is not None:
            # 如果有,发出警告,提示旧的轨迹将被新的轨迹覆盖
            warnings.warn("新轨迹已加载,将覆盖旧轨迹。", RuntimeWarning)

        # 创建新的Trajectory对象,使用以下参数进行初始化：
        # get_all_observation_keys(): 获取所有观察键的方法
        # low: 观察空间的最小值
        # high: 观察空间的最大值
        # joint_pos_idx: 关节位置索引,用于映射和重映射
        # interpolate_map: 插值映射方法
        # interpolate_remap: 插值重映射方法
        # _get_interpolate_map_params(): 获取插值映射参数的方法
        # _get_interpolate_remap_params(): 获取插值重映射参数的方法
        # warn: 是否在轨迹范围被违反时发出警告
        # **traj_params: 从外部传入的其他轨迹参数
        self.trajectories = Trajectory(
            keys=self.get_all_observation_keys(),
            low=self.info.observation_space.low,
            high=self.info.observation_space.high,
            joint_pos_idx=self.obs_helper.joint_pos_idx,
            interpolate_map=self._interpolate_map,
            interpolate_remap=self._interpolate_remap,
            interpolate_map_params=self._get_interpolate_map_params(),
            interpolate_remap_params=self._get_interpolate_remap_params(),
            warn=warn,
            **traj_params
        )

    def play_trajectory(
        self,
        n_episodes=None,
        n_steps_per_episode=None,
        render=True,
        record=False,
        recorder_params=None,
    ):
        """
        播放加载的轨迹演示,通过在每个步骤强制模型位置到轨迹中的位置。

        Args:
            n_episodes (int): 要重播的剧集数量。
            n_steps_per_episode (int): 每集重播的步骤数量。
            render (bool): 如果为True,将渲染轨迹。
            record (bool): 如果为True,将记录渲染的轨迹。
            recorder_params (dict): 包含录像机参数的字典。
        """
        # 确保已经加载了轨迹数据
        assert self.trajectories is not None

        # 如果需要记录,则必须渲染
        if record:
            assert render
            # 根据模型的时间步长计算每秒帧数
            fps = 1 / self.dt
            # 根据提供的参数创建视频记录器
            recorder = (
                VideoRecorder(fps=fps, **recorder_params)
                if recorder_params is not None
                else VideoRecorder(fps=fps)
            )
        else:
            recorder = None

        # 重置模拟环境
        self.reset()
        # 获取当前样本
        sample = self.trajectories.get_current_sample()
        # 设置模拟状态
        self.set_sim_state(sample)

        # 如果需要渲染
        if render:
            # 渲染当前帧,如果记录则保存
            frame = self.render(record)
        else:
            frame = None

        # 如果需要记录
        if record:
            # 将渲染的帧传递给记录器
            recorder(frame)

        # 获取最大的整数值
        highest_int = np.iinfo(np.int32).max
        # 如果每集的步骤数量未指定,则设置为最大整数值
        if n_steps_per_episode is None:
            n_steps_per_episode = highest_int
        # 如果剧集数量未指定,则设置为最大整数值
        if n_episodes is None:
            n_episodes = highest_int

        # 遍历指定的剧集数量
        for i in range(n_episodes):
            # 在每集内遍历指定的步骤数量
            for j in range(n_steps_per_episode):
                # 设置模拟状态
                self.set_sim_state(sample)
                # 在步进模拟之前执行的操作
                self._simulation_pre_step()
                # 执行MuJoCo的前向动力学模拟
                mujoco.mj_forward(self._model, self._data)
                # 在步进模拟之后执行的操作
                self._simulation_post_step()
                # 获取下一个样本
                sample = self.trajectories.get_next_sample()
                # 如果样本为None,表示到达轨迹末尾,重置环境
                if sample is None:
                    self.reset()
                    sample = self.trajectories.get_current_sample()
                # 创建观察值
                obs = self._create_observation(np.concatenate(sample))
                # 检查是否跌倒
                if self._has_fallen(obs):
                    print("Has fallen!")
                # 如果需要渲染
                if render:
                    # 渲染当前帧,如果记录则保存
                    frame = self.render(record)
                else:
                    frame = None

                if record:
                    # 将渲染的帧传递给记录器
                    recorder(frame)

        # 完成演示后重置环境
        self.reset()
        # 停止模拟
        self.stop()

        if record:
            recorder.stop()

    def play_trajectory_from_velocity(
        self,
        n_episodes=None,
        n_steps_per_episode=None,
        render=True,
        record=False,
        recorder_params=None,
    ):
        """
        通过在每个步骤中根据轨迹中的关节速度计算出的模型位置来播放加载的轨迹演示。
        因此,在第一步中从轨迹设置关节位置。之后,使用数值积分来根据轨迹中的关节速度计算下一个关节位置。

        Args:
            n_episodes (int): 要重播的剧集数量。
            n_steps_per_episode (int): 每集重播的步骤数量。
            render (bool): 如果为True,将渲染轨迹。
            record (bool): 如果为True,将记录重播。
            recorder_params (dict): 包含录像机参数的字典。
        """
        # 确保已经加载了轨迹数据
        assert self.trajectories is not None

        # 如果需要记录,则必须渲染
        if record:
            assert render
            # 根据模型的时间步长计算每秒帧数
            fps = 1 / self.dt
            # 根据提供的参数创建视频记录器
            recorder = (
                VideoRecorder(fps=fps, **recorder_params)
                if recorder_params is not None
                else VideoRecorder(fps=fps)
            )
        else:
            recorder = None

        # 重置模拟环境
        self.reset()
        # 获取当前样本
        sample = self.trajectories.get_current_sample()
        # 设置模拟状态
        self.set_sim_state(sample)

        if render:
            # 渲染当前帧,如果记录则保存
            frame = self.render(record)
        else:
            frame = None

        if record:
            # 将渲染的帧传递给记录器
            recorder(frame)

        highest_int = np.iinfo(np.int32).max
        # 如果每集的步骤数量未指定,则设置为最大整数值
        if n_steps_per_episode is None:
            n_steps_per_episode = highest_int
        # 如果剧集数量未指定,则设置为最大整数值
        if n_episodes is None:
            n_episodes = highest_int

        # 获取关节位置和速度的长度
        len_qpos, len_qvel = self._len_qpos_qvel()
        # 从样本中提取当前的关节位置
        curr_qpos = sample[0:len_qpos]

        # 遍历指定的剧集数量
        for i in range(n_episodes):
            # 在每集内遍历指定的步骤数量
            for j in range(n_steps_per_episode):
                # 从样本中提取当前的关节速度
                qvel = sample[len_qpos : len_qpos + len_qvel]
                # 使用关节速度和当前时间步长计算下一个关节位置
                qpos = [qp + self.dt * qv for qp, qv in zip(curr_qpos, qvel)]
                # 更新样本中的关节位置
                sample[: len(qpos)] = qpos
                # 设置模拟状态
                self.set_sim_state(sample)
                # 在步进模拟之前执行的操作
                self._simulation_pre_step()
                # 执行MuJoCo的前向动力学模拟
                mujoco.mj_forward(self._model, self._data)
                # 在步进模拟之后执行的操作
                self._simulation_post_step()
                # 获取当前的关节位置
                curr_qpos = self._get_joint_pos()
                # 获取下一个样本
                sample = self.trajectories.get_next_sample()
                # 如果样本为None,表示到达轨迹末尾,重置环境
                if sample is None:
                    self.reset()
                    sample = self.trajectories.get_current_sample()
                    curr_qpos = sample[0:len_qpos]
                # 创建观察值
                obs = self._create_observation(np.concatenate(sample))

                if self._has_fallen(obs):
                    print("Has fallen!")

                if render:
                    # 渲染当前帧,如果记录则保存
                    frame = self.render(record)
                else:
                    frame = None

                if record:
                    # 将渲染的帧传递给记录器
                    recorder(frame)

        # 完成演示后重置环境
        self.reset()
        # 获取当前的关节位置
        curr_qpos = self._get_joint_pos()
        # 停止模拟
        self.stop()
        # 如果需要记录

    def reward(self, state, action, next_state, absorbing):
        """
        Calls the reward function of the environment.

        """

        return self._reward_function(state, action, next_state, absorbing)

    def reset(self, obs=None):
        """
        重置模拟环境的状态。

        Args:
            obs (可选[np.array]): 用于重置环境的观测值。

        """
        # 使用MuJoCo库函数重置模型数据和数据结构
        mujoco.mj_resetData(self._model, self._data)
        self.mean_grf.reset()

        # 如果设置了域随机化（domain randomization）
        if self._domain_rand is not None:
            self._models[self._current_model_idx] = (
                self._domain_rand.get_randomized_model(self._current_model_idx)
            )
            self._datas[self._current_model_idx] = mujoco.MjData(
                self._models[self._current_model_idx]
            )

        # 如果开启了随机环境重置
        if self._random_env_reset:
            # 随机选择一个模型索引
            self._current_model_idx = np.random.randint(0, len(self._models))
        else:
            # 顺序选择下一个模型索引（如果到达末尾，则回到第一个）
            self._current_model_idx = (
                self._current_model_idx + 1
                if self._current_model_idx < len(self._models) - 1
                else 0
            )
        # 更新当前模型和数据结构
        self._model = self._models[self._current_model_idx]
        self._data = self._datas[self._current_model_idx]
        # 更新观测值助手（用于构建观测值）
        self.obs_helper = self.obs_helpers[self._current_model_idx]

        self.setup(obs)

        if self._viewer is not None and self.more_than_one_env:
            self._viewer.load_new_model(self._model)

        self._obs = self._create_observation(self.obs_helper._build_obs(self._data))
        return self._modify_observation(self._obs)

    def setup(self, obs):
        """
        Function to setup the initial state of the simulation. Initialization can be done either
        randomly, from a certain initial, or from the default initial state of the model.
        用于设置模拟的初始状态的功能。初始化可以通过随机生成、从特定初始状态或从模型的默认初始状态进行。

        Args:
            obs (np.array): Observation to initialize the environment from;

        """

        # 重置奖励函数的状态
        self._reward_function.reset_state()

        # 如果提供了观测值obs，则从obs初始化模拟
        if obs is not None:
            self._init_sim_from_obs(obs)
        else:
            # 如果没有轨迹数据，不能随机开始
            if not self.trajectories and self._random_start:
                raise ValueError("Random start not possible without trajectory data.")
            # 如果没有轨迹数据，不能设置初始步骤
            elif not self.trajectories and self._init_step_no is not None:
                raise ValueError(
                    "Setting an initial step is not possible without trajectory data."
                )
            # 不能同时设置随机开始和初始步骤
            elif self._init_step_no is not None and self._random_start:
                raise ValueError(
                    "Either use a random start or set an initial step, not both."
                )

            # 如果存在轨迹数据
            if self.trajectories is not None:
                if self._random_start:
                    sample = self.trajectories.reset_trajectory()
                elif self._init_step_no:
                    # 获取轨迹长度和轨迹数量
                    traj_len = self.trajectories.trajectory_length
                    n_traj = self.trajectories.number_of_trajectories
                    # 确保初始化步骤编号在有效范围内
                    assert self._init_step_no <= traj_len * n_traj
                    # 计算子步骤编号和轨迹编号
                    substep_no = int(self._init_step_no % traj_len)
                    traj_no = int(self._init_step_no / traj_len)
                    sample = self.trajectories.reset_trajectory(substep_no, traj_no)
                else:
                    # sample random trajectory and use the first sample
                    # 随机选择一个轨迹并使用第一个样本
                    sample = self.trajectories.reset_trajectory(substep_no=0)

                self.set_sim_state(sample)

    def is_absorbing(self, obs):
        """
        Checks if an observation is an absorbing state or not.

        Args:
            obs (np.array): Current observation;

        Returns:
            True, if the observation is an absorbing state; otherwise False;

        """

        return self._has_fallen(obs) if self._use_absorbing_states else False

    def set_sim_state(self, sample):
        """
        根据观察值设置模拟的状态。

        Args:
            sample (list 或 np.array): 用于设置模拟状态的样本。
        """
        # 获取观察值的规范
        obs_spec = self.obs_helper.observation_spec
        # 确保样本的长度与观察规范长度相同
        assert len(sample) == len(obs_spec)

        # 遍历观察规范和样本的每一个条目
        for key_name_ot, value in zip(obs_spec, sample):
            # 解构关键信息（key, name, observation type）
            key, name, ot = key_name_ot

            # 根据观察类型设置模拟数据
            if ot == ObservationType.JOINT_POS:
                self._data.joint(name).qpos = value
            elif ot == ObservationType.JOINT_VEL:
                self._data.joint(name).qvel = value
            elif ot == ObservationType.SITE_ROT:
                self._data.site(name).xmat = value
            # 这里可以扩展其他类型的观察值设置

    def _init_sim_from_obs(self, obs):
        """
        从一个观测值初始化模拟。

        Args:
            obs (np.array): 要将模拟状态设置为的观测值。  The observation to set the simulation state to.
        """
        # 确保观测值的维度是一维的
        assert len(obs.shape) == 1
        # 在观测值前面添加 x 和 y 位置信息（这里假设观测值不包含这些信息）
        # 使用全0填充，可能是为了对齐期望的状态格式
        obs = np.concatenate([[0.0, 0.0], obs])
        # 获取观测值的规范（可能是观测值的维度或者各个维度的意义）
        obs_spec = self.obs_helper.observation_spec
        # 确保观测值的长度至少与观测值规范一样长
        assert len(obs) >= len(obs_spec)
        # 移除观测值中不符合观测值规范的部分（如果观测值长度超过了规范长度）
        obs = obs[: len(obs_spec)]
        # 设置模拟状态为提供的观测值
        self.set_sim_state(obs)

    def _get_observation_space(self):
        """
        Returns a tuple of the lows and highs (np.array) of the observation space.

        """
        sim_low, sim_high = (
            self.info.observation_space.low[2:],
            self.info.observation_space.high[2:],
        )

        if self._use_foot_forces:
            grf_low, grf_high = (
                -np.ones((self._get_grf_size(),)) * np.inf,
                np.ones((self._get_grf_size(),)) * np.inf,
            )
            return (
                np.concatenate([sim_low, grf_low]),
                np.concatenate([sim_high, grf_high]),
            )
        else:
            return sim_low, sim_high

    def _create_observation(self, obs):
        """
        Creates a full vector of observations.

        Args:
            obs (np.array): Observation vector to be modified or extended;

        Returns:
            New observation vector (np.array);

        """

        if self._use_foot_forces:
            # 将观测向量的第3个元素（索引为2）之后的所有元素与 mean_grf.mean/1000 进行拼接
            # mean_grf.mean 是平均地面反作用力（Ground Reaction Force, GRF）
            # 除以1000是为了归一化或调整单位
            obs = np.concatenate(
                [
                    obs[2:],
                    self.mean_grf.mean / 1000.0,
                ]
            ).flatten()  # .flatten() 方法用于将任何形状的数组转换为1维数组；
        else:
            # 仅拼接观测向量的第3个元素之后的所有元素
            obs = np.concatenate(
                [
                    obs[2:],
                ]
            ).flatten()

        return obs

    def _setup_ground_force_statistics(self):
        """
        Returns a running average method for the mean ground forces.  By default, 4 ground force sensors are used.
        Environments that use more or less have to override this function.

        """

        mean_grf = RunningAveragedWindow(
            shape=(self._get_grf_size(),), window_size=self._n_intermediate_steps
        )

        return mean_grf

    def _get_reward_function(self, reward_type, reward_params):
        """
        Constructs a reward function.

        Args:
            reward_type (string): Name of the reward.
            reward_params (dict): Parameters of the reward function.

        Returns:
            Reward function.

        """

        if reward_type == "custom":
            reward_func = CustomReward(**reward_params)
        elif reward_type == "target_velocity":
            x_vel_idx = self.get_obs_idx("dq_pelvis_tx")
            assert len(x_vel_idx) == 1
            x_vel_idx = x_vel_idx[0]
            reward_func = TargetVelocityReward(x_vel_idx=x_vel_idx, **reward_params)
        elif reward_type == "x_pos":
            x_idx = self.get_obs_idx("q_pelvis_tx")
            assert len(x_idx) == 1
            x_idx = x_idx[0]
            reward_func = PosReward(pos_idx=x_idx)
        elif reward_type is None:
            reward_func = NoReward()
        else:
            raise NotImplementedError(
                "The specified reward has not been implemented: %s" % reward_type
            )

        return reward_func

    # TODO: 这里是否还要抽象出一个类 但是这里已经是调用obs_helper类了
    def _get_joint_pos(self):
        """
        Returns a vector (np.array) containing the current joint position of the model in the simulation.

        """
        return self.obs_helper.get_joint_pos_from_obs(
            self.obs_helper._build_obs(self._data)
        )

    def get_obs_idx(self, key):
        """
        Returns a list of indices corresponding to the respective key.

        """
        idx = self.obs_helper.obs_idx_map[key]

        # shift by 2 to account for deleted x and y
        idx = [i - 2 for i in idx]

        return idx

    def _has_fallen(self, obs, return_err_msg=False):
        """
        检查模型是否跌倒。这个方法需要为每个环境单独实现,因为不同环境的跌倒条件可能不同。

        Args:
            obs (np.array): 当前观察值。
            return_err_msg (bool): 如果为True,返回包含违规情况的错误信息。

        Returns:
            如果当前观察值表明模型已经跌倒,则返回True,否则返回False。
        """
        # 抛出一个未实现异常。这意味着该方法需要在子类中被覆盖实现。
        # 如果直接调用这个方法,将会导致程序错误,因为基类不知道如何检查跌倒。
        raise NotImplementedError

    # TODO： 奇葩方法  但是有用,不知道怎么实现的
    @staticmethod
    def list_registered_loco_mujoco():
        """
        List registered loco_mujoco environments.

        Returns:
             The list of the registered loco_mujoco environments.

        """
        return list(LocoEnvBase._registered_envs.keys())

    def _len_qpos_qvel(self):
        """
        Returns the lengths of the joint position vector and the joint velocity vector, including x and y.
        返回关节位置向量和关节速度向量的长度,包括x和y。
        """
        # 获取所有的观测键
        keys = self.get_all_observation_keys()
        # 计算以 "q_" 开头的键的数量,这些键代表关节位置
        len_qpos = len([key for key in keys if key.startswith("q_")])
        len_qvel = len([key for key in keys if key.startswith("dq_")])

        return len_qpos, len_qvel

    @staticmethod
    def _interpolate_map(traj, **interpolate_map_params):
        """
        A mapping that is supposed to transform a trajectory into a space where interpolation is
        allowed. E.g., maps a rotation matrix to a set of angles. If this function is not
        overwritten, it just converts the list of np.arrays to a np.array.
        一个映射,其目的是将轨迹转换到一个可以进行插值的空间。
        例如,将旋转矩阵映射到一组角度。
        如果这个方法没有被重写,它仅仅是将 np.array 列表转换为一个 np.array。

        Args:
            traj (list): List of np.arrays containing each observations. Each np.array
                has the shape (n_trajectories, n_samples, (dim_observation)). If dim_observation
                is one the shape of the array is just (n_trajectories, n_samples).
            interpolate_map_params: Set of parameters needed by the individual environments.

        Returns:
            A np.array with shape (n_observations, n_trajectories, n_samples). dim_observation
            has to be one.

        """
        # 直接将传入的列表转换为一个 np.array 并返回
        # 这意味着在默认情况下,这个方法不执行任何实际的映射转换
        return np.array(traj)

    @staticmethod
    def _interpolate_remap(traj, **interpolate_remap_params):
        """
        The corresponding backwards transformation to _interpolation_map. If this function is
        not overwritten, it just converts the np.array to a list of np.arrays.

        Args:
            traj (np.array): Trajectory as np.array with shape (n_observations, n_trajectories, n_samples).
            dim_observation is one.
            interpolate_remap_params: Set of parameters needed by the individual environments.

        Returns:
            List of np.arrays containing each observations. Each np.array has the shape
            (n_trajectories, n_samples, (dim_observation)). If dim_observation
            is one the shape of the array is just (n_trajectories, n_samples).

        """
        # 通过列表推导式,将传入的 np.array 按照第一个维度（n_observations）拆分成列表中的独立 np.array
        # 在这个默认实现中,它实际上没有执行任何逆向变换,只是将数组拆分成了列表
        return [obs for obs in traj]

    def _get_interpolate_map_params(self):
        """
        Returns all parameters needed to do the interpolation mapping for the respective environment.

        """

        pass

    def _get_interpolate_remap_params(self):
        """
        Returns all parameters needed to do the interpolation remapping for the respective environment.

        """

        pass

    @classmethod
    def register(cls):
        """
        Register an environment in the environment list and in the loco_mujoco env list.

        """
        env_name = cls.__name__
        print("register  env_name = ", env_name)

        if env_name not in Environment._registered_envs:
            Environment._registered_envs[env_name] = cls

        if env_name not in LocoEnvBase._registered_envs:
            LocoEnvBase._registered_envs[env_name] = cls

    @classmethod
    def get_all_task_names(cls):
        """
        Returns a list of all available tasks in LocoMujoco.
        cls是一个通常在Python类方法中使用的约定变量名,代表“类本身”。
        """

        task_names = []
        for e in cls.list_registered_loco_mujoco():
            env = cls._registered_envs[e]
            confs = env.valid_task_confs.get_all_combinations()
            for conf in confs:
                task_name = list(conf.values())
                task_name.insert(
                    0,
                    env.__name__,
                )
                task_name = ".".join(task_name)
                task_names.append(task_name)

        return task_names

    _registered_envs = dict()


class ValidTaskConf:
    """Simple class that holds all valid configurations of an environments."""

    def __init__(self, tasks=None, modes=None, data_types=None, non_combinable=None):
        """

        Args:
            tasks (list): List of valid tasks.
            modes (list): List of valid modes.
            data_types (list): List of valid data_types.
            non_combinable (list): List of tuples ("task", "mode", "dataset_type"),
                which are NOT allowed to be combined. If one of them is None, it is neglected.

        """

        self.tasks = tasks
        self.modes = modes
        self.data_types = data_types
        self.non_combinable = non_combinable
        if non_combinable is not None:
            for nc in non_combinable:
                assert len(nc) == 3

    def get_all(self):
        return (
            deepcopy(self.tasks),
            deepcopy(self.modes),
            deepcopy(self.data_types),
            deepcopy(self.non_combinable),
        )

    def get_all_combinations(self):
        """
        Returns all possible combinations of configurations.

        """

        confs = []

        if self.tasks is not None:
            tasks = self.tasks
        else:
            tasks = [None]
        if self.modes is not None:
            modes = self.modes
        else:
            modes = [None]
        if self.data_types is not None:
            data_types = self.data_types
        else:
            data_types = [None]

        for t, m, dt in product(tasks, modes, data_types):
            conf = dict()
            if t is not None:
                conf["task"] = t
            if m is not None:
                conf["mode"] = m
            if dt is not None:
                conf["data_type"] = dt

            # check for non-combinable
            if self.non_combinable is not None:
                for nc in self.non_combinable:
                    bad_t, bad_m, bad_dt = nc
                    if not (
                        (t == bad_t or bad_t is None)
                        and (m == bad_m or bad_m is None)
                        and (dt == bad_dt or bad_dt is None)
                    ):
                        confs.append(conf)
            else:
                confs.append(conf)

        return confs
