import os

import warnings
from pathlib import Path
from copy import deepcopy
from tempfile import mkdtemp
from itertools import product
import transforms3d as tf3

from dm_control import mjcf

from mushroom_rl.core import Environment
from mushroom_rl.environments import MultiMuJoCo
from mushroom_rl.utils import spaces
from mushroom_rl.utils.running_stats import *
from mushroom_rl.utils.mujoco import *
from mushroom_rl.utils.record import VideoRecorder

import mujoco
import olympic_mujoco
from olympic_mujoco.utils import Trajectory
from olympic_mujoco.utils import NoReward, CustomReward, TargetVelocityReward, PosReward

from olympic_mujoco.interfaces.mujoco_robot_interface import MujocoRobotInterface

from olympic_mujoco.enums.enums import AlgorithmType

import mujoco_viewer
import mujoco.viewer


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
        train_start=False,
        algorithm_type: AlgorithmType=AlgorithmType.REINFORCEMENT_LEARNING,
        sim_dt=0.0025,
        control_dt=0.025,
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

        # TODO：区分算法的类型
        # TODO：这里传递参数的方式存在问题,make()中应该想办法包含这个类型
        # self._algorithm_type=algorithm_type
        self._algorithm_type=AlgorithmType.IMITATION_LEARNING

        self.viewer=None

        # TODO: 特别注意handles 和 handle 的区别
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
            # 设置是否使用吸收状态(在某些强化学习任务中,用于指示一个episode的结束)
            self._use_absorbing_states = use_absorbing_states

        # TODO:强化学习相关
        self.frame_skip = (control_dt/sim_dt)
        self._model.opt.timestep = sim_dt

        self.init_qpos = self._data.qpos.ravel().copy()
        self.init_qvel = self._data.qvel.ravel().copy()

    # TODO:这部分是强化学习训练相关的代码
    # ***********************************************************************************************************
    def test(self):
        self.obs_helper.get_joint_pos_from_obs(
            self.obs_helper._build_obs(self._data)
        )


    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError
    
    def test_reset(self):
        # print("这里调用的是test_reset")
        mujoco.mj_resetData(self._model, self._data)
        ob = self.reset_model()
        return ob
    
    # TODO:这里应该专门有一个模块

    def render(self):
        if self.viewer is None:
            # TODO:后续有时间这里应修改成自己的viewer库
            self.viewer = mujoco_viewer.MujocoViewer(self._model, self._data)
            self.viewer_setup()
        self.viewer.render()

    # ***********************************************************************************************************

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self._model.stat.extent * 1.5
        self.viewer.cam.lookat[2] = 1.5
        self.viewer.cam.lookat[0] = 2.0
        self.viewer.cam.elevation = -20
        self.viewer.vopt.geomgroup[0] = 1
        self.viewer._render_every_frame = True

    def viewer_is_paused(self):
        return self.viewer._paused

    def uploadGPU(self, hfieldid=None, meshid=None, texid=None):
        # hfield
        if hfieldid is not None:
            mujoco.mjr_uploadHField(self._model, self.viewer.ctx, hfieldid)
        # mesh
        if meshid is not None:
            mujoco.mjr_uploadMesh(self._model, self.viewer.ctx, meshid)
        # texture
        if texid is not None:
            mujoco.mjr_uploadTexture(self._model, self.viewer.ctx, texid)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    #---------------------------------------------------------------------------------------------------------
    #----------------------------------------- 轨迹的加载与播放 ----------------------------------------------
    #---------------------------------------------------------------------------------------------------------

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
            frame = self.render()
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
            frame = self.render()
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


                # print("///////////////////////////////////////////////////////////////////")
                # print("self._get_joint_pos() = ",self._get_joint_pos())
                # print("///////////////////////////////////////////////////////////////////")

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
                    frame = self.render()
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
    
    #---------------------------------------------------------------------------------------------------------


    #---------------------------------------------------------------------------------------------------------
    #----------------------------------------- 环境状态的设置 ----------------------------------------------
    #---------------------------------------------------------------------------------------------------------

    def reset(self, obs=None):
        """
        重置模拟环境的状态。

        Args:
            obs (可选[np.array]): 用于重置环境的观测值。

        """

        if self._algorithm_type == AlgorithmType.REINFORCEMENT_LEARNING:
            obs = self.test_reset()
            return obs
        elif self._algorithm_type == AlgorithmType.IMITATION_LEARNING:

            # 使用MuJoCo库函数重置模型数据和数据结构
            mujoco.mj_resetData(self._model, self._data)
            self.mean_grf.reset()

            # 更新当前模型和数据结构
            self._model = self._models[self._current_model_idx]
            self._data = self._datas[self._current_model_idx]
            # 更新观测值助手(用于构建观测值)
            self.obs_helper = self.obs_helpers[self._current_model_idx]

            self.setup(obs)
            
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
        # self._reward_function.reset_state()

        # 如果提供了观测值obs,则从obs初始化模拟
        if obs is not None:
            self._init_sim_from_obs(obs)
        else:
            # 如果没有轨迹数据,不能随机开始
            if not self.trajectories and self._random_start:
                raise ValueError("Random start not possible without trajectory data.")
            # 如果没有轨迹数据,不能设置初始步骤
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

    def set_sim_state(self, sample):
        """
        根据观察值设置模拟的状态。

        Args:
            sample (list 或 np.array): 用于设置模拟状态的样本。
        """
        # 获取观察值的规范
        obs_spec = self.obs_helper.observation_spec

        # print("***********************************************************************************")
        # print("obs_spec = ",obs_spec)
        # print("***********************************************************************************")


        # 确保样本的长度与观察规范长度相同
        assert len(sample) == len(obs_spec)

        # 遍历观察规范和样本的每一个条目
        for key_name_ot, value in zip(obs_spec, sample):
            # 解构关键信息(key, name, observation type)
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
        # 在观测值前面添加 x 和 y 位置信息(这里假设观测值不包含这些信息)
        # 使用全0填充,可能是为了对齐期望的状态格式
        obs = np.concatenate([[0.0, 0.0], obs])
        # 获取观测值的规范(可能是观测值的维度或者各个维度的意义)
        obs_spec = self.obs_helper.observation_spec
        # 确保观测值的长度至少与观测值规范一样长
        assert len(obs) >= len(obs_spec)
        # 移除观测值中不符合观测值规范的部分(如果观测值长度超过了规范长度)
        obs = obs[: len(obs_spec)]
        # 设置模拟状态为提供的观测值
        self.set_sim_state(obs)

    #---------------------------------------------------------------------------------------------------------

    def is_absorbing(self, obs):
        """
        Checks if an observation is an absorbing state or not.

        Args:
            obs (np.array): Current observation;

        Returns:
            True, if the observation is an absorbing state; otherwise False;

        """

        return self._has_fallen(obs) if self._use_absorbing_states else False
    

    #---------------------------------------------------------------------------------------------------------
    #----------------------------------------- 对观测空间操作 ----------------------------------------------
    #---------------------------------------------------------------------------------------------------------

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
            # 将观测向量的第3个元素(索引为2)之后的所有元素与 mean_grf.mean/1000 进行拼接
            # mean_grf.mean 是平均地面反作用力(Ground Reaction Force, GRF)
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
    
    #---------------------------------------------------------------------------------------------------------


    #---------------------------------------------------------------------------------------------------------
    #----------------------------------------- 奖励函数操作 ----------------------------------------------
    #---------------------------------------------------------------------------------------------------------

    def reward(self, state, action, next_state, absorbing):
        """
        Calls the reward function of the environment.

        """
        return None
        # return self._reward_function(state, action, next_state, absorbing)

    # def _get_reward_function(self, reward_type, reward_params):
    #     """
    #     Constructs a reward function.

    #     Args:
    #         reward_type (string): Name of the reward.
    #         reward_params (dict): Parameters of the reward function.

    #     Returns:
    #         Reward function.

    #     """

    #     if reward_type == "custom":
    #         reward_func = CustomReward(**reward_params)
    #     elif reward_type == "target_velocity":
    #         # TODO:这里要 dq_pelvis_tx 干什么
    #         x_vel_idx = self.get_obs_idx("dq_pelvis_tx")
    #         print("--------------------------------------------")
    #         print("x_vel_idx = ",x_vel_idx)
    #         print("--------------------------------------------")
    #         assert len(x_vel_idx) == 1
    #         x_vel_idx = x_vel_idx[0]
    #         reward_func = TargetVelocityReward(x_vel_idx=x_vel_idx, **reward_params)
    #     elif reward_type == "x_pos":
    #         x_idx = self.get_obs_idx("q_pelvis_tx")
    #         print("--------------------------------------------")
    #         print("x_vel_idx = ",x_vel_idx)
    #         print("--------------------------------------------")
    #         assert len(x_idx) == 1
    #         x_idx = x_idx[0]
    #         reward_func = PosReward(pos_idx=x_idx)
    #     elif reward_type is None:
    #         reward_func = NoReward()
    #     else:
    #         raise NotImplementedError(
    #             "The specified reward has not been implemented: %s" % reward_type
    #         )

    #     return reward_func
    
    def _get_reward_function(self, reward_type, reward_params):
        """
        构建一个奖励函数。
        
        参数:
            reward_type (string): 奖励的名称。
            reward_params (dict): 奖励函数的参数。
        
        返回:
            reward_func: 奖励函数对象。
        """
        # 根据奖励类型创建不同的奖励函数对象
        if reward_type == "custom":
            # 使用自定义奖励,并通过 reward_params 传递参数
            reward_func = CustomReward(**reward_params)
        elif reward_type == "target_velocity":
            # 获取与 pelvis_tx（通常是沿x轴的线性速度）相关的观察索引
            x_vel_idx = self.get_obs_idx("dq_pelvis_tx")
            # 确保观察索引的数量为1
            assert len(x_vel_idx) == 1
            # 获取具体的索引值
            x_vel_idx = x_vel_idx[0]
            # 使用目标速度奖励,并传递相关参数和索引
            reward_func = TargetVelocityReward(x_vel_idx=x_vel_idx, **reward_params)
        elif reward_type == "x_pos":
            # 获取与 pelvis_tx（通常是沿x轴的位置）相关的观察索引
            x_idx = self.get_obs_idx("q_pelvis_tx")
            # 确保观察索引的数量为1
            assert len(x_idx) == 1
            # 获取具体的索引值
            x_idx = x_idx[0]
            # 使用位置奖励,并传递索引
            reward_func = PosReward(pos_idx=x_idx)
        elif reward_type is None:
            # 如果没有指定奖励类型,使用无奖励函数
            reward_func = NoReward()
        else:
            # 如果提供的奖励类型没有被实现,抛出一个异常
            raise NotImplementedError("The specified reward has not been implemented: %s" % reward_type)
        
        # 返回构建的奖励函数对象
        return reward_func



    #---------------------------------------------------------------------------------------------------------


    #---------------------------------------------------------------------------------------------------------
    #----------------------------------------- xml操作(也相当于是对观测空间的操作) --------------------------------
    #---------------------------------------------------------------------------------------------------------

    @staticmethod
    def _delete_from_xml_handle(xml_handle, joints_to_remove, motors_to_remove, equ_constraints):
        """
        Deletes certain joints, motors and equality constraints from a Mujoco XML handle.

        Args:
            xml_handle: Handle to Mujoco XML.
            joints_to_remove (list): List of joint names to remove.
            motors_to_remove (list): List of motor names to remove.
            equ_constraints (list): List of equality constraint names to remove.

        Returns:
            Modified Mujoco XML handle.

        """

        for j in joints_to_remove:
            j_handle = xml_handle.find("joint", j)
            j_handle.remove()
        for m in motors_to_remove:
            m_handle = xml_handle.find("actuator", m)
            m_handle.remove()
        for e in equ_constraints:
            e_handle = xml_handle.find("equality", e)
            e_handle.remove()

        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # for joint in xml_handle.find_all('joint'):
        #     # 打印每个关节的名称
        #     print(joint.name)
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        
        return xml_handle
    
    def get_kinematic_obs_mask(self):
        """
        Returns a mask (np.array) for the observation specified in observation_spec (or part of it).

        """

        return np.arange(len(self.obs_helper.observation_spec) - 2)
    
    def get_kinematic_obs_mask(self):
        """
        返回一个掩码(np.array类型),该掩码针对observation_spec中指定的观察值(或其一部分)。

        这个函数通常在一个处理动力学状态或者观察数据的类中使用。
        掩码在这里是一个数组,它指示哪些部分的观察数据应该被考虑或排除。
        在很多算法中,不是所有的观察数据都是相关的,掩码允许我们选择那些重要的数据。


        返回:
        np.array: 包含布尔值的数组,指示哪些观察值应该被包含。

        """
        # 使用numpy的arange函数生成一个序列数组,这个序列从0开始到观察规格长度减去2的位置
        # 这里的观察规格(observation_spec)是一个列表或数组,它定义了观察值的结构或类型
        # np.arange函数在这里生成一个索引数组,用于选择观察值的一部分
        return np.arange(len(self.obs_helper.observation_spec) - 2)

    @staticmethod
    def _save_xml_handle(xml_handle, tmp_dir_name, file_name="tmp_model.xml"):
        """
        将 MuJoCo XML 句柄保存到 tmp_dir_name 位置的文件中。
        如果 tmp_dir_name 是 None,则在 /tmp 下创建一个临时目录。
        
        参数:
        xml_handle: MuJoCo XML 句柄。
        tmp_dir_name (str): 临时目录的路径。如果为 None,则在 /tmp 下创建临时目录。
        
        返回:
        保存路径的字符串。
        """
        # 检查是否提供了 tmp_dir_name,如果提供了,确保它存在
        if tmp_dir_name is not None:
            assert os.path.exists(tmp_dir_name), "指定的目录(\"%s\")不存在。" % tmp_dir_name
            # 在提供的目录中创建一个临时目录
            dir = mkdtemp(dir=tmp_dir_name)
        else:
            # 如果没有提供目录名称,则在系统默认的临时目录 /tmp 下创建临时目录
            dir = mkdtemp()
        
        # 创建完整的文件路径
        file_path = os.path.join(dir, file_name)
        
        # 将 MuJoCo XML 数据和相关的资源（如纹理、形状等）导出到指定目录
        mjcf.export_with_assets(xml_handle, dir, file_name)
        
        # 返回文件保存的完整路径
        return file_path

    #---------------------------------------------------------------------------------------------------------


    #---------------------------------------------------------------------------------------------------------
    #--------------------------------------------- 数据集相关操作 ----------------------------------------------
    #---------------------------------------------------------------------------------------------------------

    # TODO：这里应是与模仿学习相关的代码
    def create_dataset(self, ignore_keys=None):
        """
        从指定的轨迹创建一个数据集。

        参数:
            ignore_keys (list): 在数据集中要忽略的键的列表。

        返回:
            一个字典,包含状态(states)、下一个状态(next_states)和吸收标志(absorbing flags)。
            状态的形状为 (N_traj x N_samples_per_traj, dim_state),而吸收标志的形状为
            (N_traj x N_samples_per_traj)。对于完美和偏好数据集,还会提供动作(actions)。

            N_traj: 轨迹的数量
            N_samples_per_traj: 每个轨迹中的样本数量
            dim_state: 状态的维度
        """
        # 如果数据集属性为空,则进行创建
        if self._dataset is None:
            # 检查是否提供了轨迹
            if self.trajectories is not None:
                # 使用提供的忽略键列表创建数据集
                dataset = self.trajectories.create_dataset(ignore_keys=ignore_keys)
                
                # 检查数据集中的所有状态是否满足 _has_fallen 方法
                for state in dataset["states"]:
                    has_fallen, msg = self._has_fallen(state, return_err_msg=True)
                    # 如果有状态表示已经跌倒,抛出异常
                    if has_fallen:
                        err_msg = "Some of the states in the created dataset are terminal states. " \
                                  "This should not happen.\n\nViolations:\n"
                        err_msg += msg
                        raise ValueError(err_msg)
                
                # 深拷贝创建的数据集,以避免后续操作对原始数据的影响
                self._dataset = deepcopy(dataset)
                return dataset
            else:
                # 如果没有提供轨迹,抛出异常
                raise ValueError("No trajectory was passed to the environment. "
                                 "To create a dataset pass a trajectory first.")
        else:
            # 如果数据集已经被创建过,直接返回一个深拷贝,避免对原始数据集的修改
            return deepcopy(self._dataset)



    def load_dataset_and_get_traj_files(self, dataset_path, freq=None):
        """
        给定一个数据集,计算一个包含动力学的字典。如果提供了freq,那么x和z位置将基于速度计算。

        参数:
            dataset_path (str): 数据集的路径。
            freq (float): 数据在观察中的频率。

        返回:
            一个字典,包含在observation_spec中指定的键以及数据集中的相应值。
        """

        # 加载数据集
        # 使用Path对象来确保路径正确,并将路径解析为绝对路径
        dataset = np.load(str(Path(olympic_mujoco.__file__).resolve().parent / dataset_path))
        
        # 创建数据集的深拷贝以避免修改原始数据
        self._dataset = deepcopy({k: d for k, d in dataset.items()})
        
        # 获取数据集中的状态数据
        states = dataset["states"]
        last = dataset["last"]
        
        # 确保状态数据至少是二维的,以便进行索引操作
        states = np.atleast_2d(states)
        
        # 获取与观察规格相关的键
        rel_keys = [obs_spec[0] for obs_spec in self.obs_helper.observation_spec]
        
        # 获取状态数据的数量
        num_data = len(states)
        
        # 初始化一个字典来存储轨迹数据
        trajectories = dict()
        
        # 遍历相关的键
        for i, key in enumerate(rel_keys):
            # 对于位置数据（通常是前两个键,即x和y位置）
            if i < 2:
                if freq is None:
                    # 如果没有提供频率,则用零填充x和y位置数据
                    data = np.zeros(num_data)
                else:
                    # 如果提供了频率,根据速度计算位置
                    dt = 1 / float(freq)
                    # 确保状态数据的长度足够进行计算
                    assert len(states) > 2
                    # 获取对应速度的索引
                    vel_idx = rel_keys.index("d" + key) - 2
                    # 初始化位置数据列表,第一个元素为0.0
                    data = [0.0]
                    # 遍历状态数据,计算位置
                    for j, o in enumerate(states[:-1, vel_idx], 1):
                        # 如果当前样本是吸收状态（即轨迹结束）,将位置设置为0.0
                        if last is not None and last[j - 1] == 1:
                            data.append(0.0)
                        else:
                            # 否则,根据速度和时间间隔计算位置
                            data.append(data[-1] + dt * o)
                    # 将位置数据转换为数组
                    data = np.array(data)
            else:
                # 对于其他数据（非位置）,直接从状态数据中获取
                data = states[:, i - 2]
            
            # 将计算得到的数据存入轨迹字典
            trajectories[key] = data
        
        # 如果状态数据长度大于2,添加分割点（即轨迹结束的位置）
        if len(states) > 2:
            # 找到所有吸收状态的位置,并将其转换为分割点
            trajectories["split_points"] = np.concatenate([[0], np.squeeze(np.argwhere(last == 1) + 1)])
        
        # 返回填充了数据的轨迹字典
        return trajectories


    #---------------------------------------------------------------------------------------------------------


    def _preprocess_action(self, action):
        """
        这个函数预处理所有动作。在这个环境中,所有预期的动作应该在 -1 和 1 之间。
        因此,我们需要将动作反归一化,以发送正确的动作到仿真中。
        
        注意：如果动作不在 [-1, 1] 范围内,反归一化后的版本将在 Mujoco 中被裁剪。
        
        参数:
            action (np.array): 将要发送到环境的动作；

        返回:
            反归一化后的动作 (np.array),该动作将被发送到环境；
        """
        # 复制动作数组以避免修改原始数据
        # 使用归一化动作的范围和均值来反归一化动作
        # self.norm_act_delta 是归一化时使用的范围,self.norm_act_mean 是动作的均值
        unnormalized_action = (action.copy() * self.norm_act_delta) + self.norm_act_mean
        
        # 返回反归一化后的动作
        return unnormalized_action


    def _simulation_post_step(self):
        """
        如果需要,更新地面反作用力统计信息。
        """
        # 如果类属性 _use_foot_forces 被设置为 True,则执行以下代码
        if self._use_foot_forces:
            # 获取当前步的地面反作用力（GRF,Ground Reaction Forces）
            grf = self._get_ground_forces()
            
            # 使用获取到的地面反作用力更新统计信息
            # self.mean_grf 是一个用于统计地面反作用力的对象
            # update_stats 方法是用于更新统计数据的自定义方法
            self.mean_grf.update_stats(grf)


    def _get_ground_forces(self):
        """
        返回地面反作用力（np.array类型）。默认情况下,使用4个地面力传感器。
        使用更多或更少传感器的环境必须重写这个函数。
        
        返回值：
        grf (np.array): 包含所有地面反作用力的数组。
        """
        # 获取右脚与地面碰撞的力,并取前三轴的分量（通常为x, y, z轴）
        # _get_collision_force 是一个私有方法,用于获取两个物体之间的碰撞力
        # "floor" 表示地面,"foot_r" 表示右脚
        # 同理,"front_foot_r" 表示右前脚,"foot_l" 表示左脚,"front_foot_l" 表示左前脚
        grf_right_foot = self._get_collision_force("floor", "foot_r")[:3]
        grf_right_front_foot = self._get_collision_force("floor", "front_foot_r")[:3]
        grf_left_foot = self._get_collision_force("floor", "foot_l")[:3]
        grf_left_front_foot = self._get_collision_force("floor", "front_foot_l")[:3]
        
        # 将所有获取到的力沿着第一个轴（通常是列方向）连接起来,形成一个包含所有力的单一数组
        grf = np.concatenate([grf_right_foot,
                            grf_right_front_foot,
                            grf_left_foot,
                            grf_left_front_foot])
        
        # 返回包含了所有地面反作用力的数组
        return grf

    def _get_idx(self, keys):
        """
        返回指定键的索引。
        
        参数:
            keys (list or str): 键的列表或单个键,用于从观察空间获取索引。
        
        返回:
            np.array: 包含指定键索引的数组。
        """
        # 检查 keys 是否为列表类型,如果不是,则假设它是一个字符串
        if type(keys) != list:
            # 确保提供的 keys 实际上是一个字符串
            assert type(keys) == str
            # 将单个字符串键包装成一个列表
            keys = [keys]
        
        # 初始化一个空列表,用于存储所有键对应的索引
        entries = []
        # 遍历所有提供的键
        for key in keys:
            # 使用 obs_helper 中的 obs_idx_map 字典获取每个键的索引
            # obs_idx_map 是一个将键映射到它们在观察空间中索引的字典
            entries.append(self.obs_helper.obs_idx_map[key])
        
        # 使用 np.concatenate 将所有索引连接成一个数组
        # 由于这些索引可能从 2 开始（取决于环境的实现）,因此从结果中减去 2
        # 这可能是因为在某些环境中,观察向量前两个元素是固定的,例如,时间步和奖励
        return np.concatenate(entries) - 2

    @staticmethod
    def _get_grf_size():
        """
        Returns the size of the ground force vector.

        """

        return 12

   #************************************************************************

    # TODO:这个函数不应该在这里
    def set_state(self, qpos, qvel):
        # 设置模型的状态,包括位置和速度
        assert qpos.shape == (self._model.nq,) and qvel.shape == (self._model.nv,)
        self._data.qpos[:] = qpos
        self._data.qvel[:] = qvel
        mujoco.mj_forward(self._model, self._data)


    def _setup_ground_force_statistics(self):
        """
        Returns a running average method for the mean ground forces.  By default, 4 ground force sensors are used.
        Environments that use more or less have to override this function.

        """

        mean_grf = RunningAveragedWindow(
            shape=(self._get_grf_size(),), window_size=self._n_intermediate_steps
        )

        return mean_grf

    

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
        # print("keys = ", keys)
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
        # 通过列表推导式,将传入的 np.array 按照第一个维度(n_observations)拆分成列表中的独立 np.array
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
        # print("register  env_name = ", env_name)

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
    


    def _get_from_obs(self, obs, keys):
        """
        Returns a part of the observation based on the specified keys.

        Args:
            obs (np.array): Observation array.
            keys (list or str): List of keys or just one key which are
                used to extract entries from the observation.

        Returns:
            np.array including the parts of the original observation whose
            keys were specified.

        """
        if self._algorithm_type == AlgorithmType.IMITATION_LEARNING:
            # obs has removed x and y positions, add dummy entries
            obs = np.concatenate([[0.0, 0.0], obs])
            if type(keys) != list:
                assert type(keys) == str
                keys = list(keys)

            entries = []
            for key in keys:
                entries.append(self.obs_helper.get_from_obs(obs, key))

            return np.concatenate(entries)


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
