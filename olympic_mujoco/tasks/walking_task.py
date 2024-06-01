import numpy as np
import random
import transforms3d as tf3
from olympic_mujoco.tasks import rewards
from enum import Enum, auto

class WalkModes(Enum):
    STANDING = auto()
    FORWARD = auto()
    BACKWARD = auto()
    LATERAL = auto()

class WalkingTask(object):
    """Bipedal locomotion by stepping on targets."""

    def __init__(self,
                 client=None,
                 dt=0.025,
                 neutral_foot_orient=[],
                 root_body='pelvis',
                 lfoot_body='lfoot',
                 rfoot_body='rfoot',
                 head_body='head',
    ):

        self._client = client  # 保存客户端实例
        self._control_dt = dt  # 控制时间步长

        self._mass = self._client.get_robot_mass()  # 获取机器人的质量
        self._goal_speed_ref = 0  # 目标速度参考值
        self._goal_height_ref = []  # 目标高度参考值
        self._swing_duration = []  # 摆动阶段持续时间
        self._stance_duration = []  # 站立阶段持续时间
        self._total_duration = []  # 总持续时间

        self._head_body_name = head_body  # 保存头部身体部分名称
        self._root_body_name = root_body  # 保存根身体部分名称
        self._lfoot_body_name = lfoot_body  # 保存左脚身体部分名称
        self._rfoot_body_name = rfoot_body  # 保存右脚身体部分名称

        # 读取先前生成的脚步计划
        with open('footstep_plans.txt', 'r') as fn:
            lines = [l.strip() for l in fn.readlines()]  # 读取文件，并去除每行首尾空白字符
        self.plans = []  # 初始化计划列表
        sequence = []  # 初始化序列列表
        for line in lines:
            if line == '---':  # 遇到分隔符
                if len(sequence):  # 如果序列不为空
                    self.plans.append(sequence)  # 将当前序列添加到计划列表
                sequence = []  # 重置序列
                continue
            else:
                sequence.append(np.array([float(l) for l in line.split(',')]))  # 将行数据转换为浮点数数组并添加到序列
    
    # 计算步进奖励
    def step_reward(self):
        target_pos = self.sequence[self.t1][0:3]  # 目标位置
        # 计算左右脚到目标位置的最小距离
        foot_dist_to_target = min([np.linalg.norm(ft - target_pos) for ft in [self.l_foot_pos, self.r_foot_pos]])
        hit_reward = 0  # 初始化击中奖励
        if self.target_reached:  # 如果目标被达到
            hit_reward = np.exp(-foot_dist_to_target / 0.25)  # 根据距离计算击中奖励
        # 计算目标中间位置
        target_mp = (self.sequence[self.t1][0:2] + self.sequence[self.t2][0:2]) / 2
        # 获取根节点的x、y位置
        root_xy_pos = self._client.get_object_xpos_by_name(self._root_body_name, 'OBJ_BODY')[0:2]
        # 计算根节点到目标中间位置的距离
        root_dist_to_target = np.linalg.norm(root_xy_pos - target_mp)
        # 根据距离计算进度奖励
        progress_reward = np.exp(-root_dist_to_target / 2)
        # 返回击中奖励和进度奖励的加权和
        return (0.8 * hit_reward + 0.2 * progress_reward)
    
    def calc_reward(self, prev_torque, prev_action, action):
        # 使用tf3库将序列中的第三个元素（代表偏航角）转换为四元数姿态
        orient = tf3.euler.euler2quat(0, 0, self.sequence[self.t1][3])
        r_frc = self.right_clock[0]
        l_frc = self.left_clock[0]
        r_vel = self.right_clock[1]
        l_vel = self.left_clock[1]
        # 如果当前模式是站立模式
        if self.mode == WalkModes.STANDING:
            # 设置右脚的力为1（是一个标志值，表示站立状态）
            r_frc = (lambda _: 1)
            # 设置左脚的力为1（同样，表示站立状态）
            l_frc = (lambda _: 1)
            # 设置右脚的速度为-1（表示在站立状态下不希望有速度）
            r_vel = (lambda _: -1)
            # 设置左脚的速度为-1（同样，表示在站立状态下不希望有速度）
            l_vel = (lambda _: -1)
        # 获取头部的x、y位置
        head_pos = self._client.get_object_xpos_by_name(self._head_body_name, 'OBJ_BODY')[0:2]
        # 获取根节点的x、y位置
        root_pos = self._client.get_object_xpos_by_name(self._root_body_name, 'OBJ_BODY')[0:2]
        # 计算奖励字典，包含多个部分：
        reward = dict(
            # 脚部力的得分，占总体奖励的15%
            foot_frc_score=0.150 * rewards._calc_foot_frc_clock_reward(self, l_frc, r_frc),
            # 脚部速度的得分，占总体奖励的15%
            foot_vel_score=0.150 * rewards._calc_foot_vel_clock_reward(self, l_vel, r_vel),
            # 姿态成本，占总体奖励的5%
            orient_cost=0.050 * rewards._calc_body_orient_reward(self, self._root_body_name, quat_ref=orient),
            # 高度误差成本，占总体奖励的5%
            height_error=0.050 * rewards._calc_height_reward(self),
            # 步进奖励，占总体奖励的45%
            step_reward=0.450 * self.step_reward(),
            # 上半身奖励，占总体奖励的5%，基于头部和根节点之间的距离计算
            upper_body_reward=0.050 * np.exp(-10*np.square(np.linalg.norm(head_pos-root_pos)))
        )
        return reward


    def transform_sequence(self, sequence):
        # 转换的目的是将序列中的脚步位置和角度调整为相对于机器人当前状态（尤其是根节点的偏航角）的坐标。
        # 获取当前左右脚的位置
        lfoot_pos = self._client.get_lfoot_body_pos()
        rfoot_pos = self._client.get_rfoot_body_pos()
        # 获取根节点的偏航角（绕Z轴旋转的角度）
        root_yaw = tf3.euler.quat2euler(self._client.get_object_xquat_by_name(self._root_body_name, 'OBJ_BODY'))[2]
        # 计算两只脚位置的中间点
        mid_pt = (lfoot_pos + rfoot_pos) / 2
        # 初始化一个列表，用于存储转换后的序列步骤
        sequence_rel = []
        for x, y, z, theta in sequence:
            # 根据当前根节点的偏航角转换每一步的x和y坐标
            x_ = mid_pt[0] + x * np.cos(root_yaw) - y * np.sin(root_yaw)
            y_ = mid_pt[1] + x * np.sin(root_yaw) + y * np.cos(root_yaw)
            # 将步的偏航角调整为相对于根节点当前偏航角的新角度
            theta_ = root_yaw + theta
            # 将转换后的坐标和角度组合成一个步骤
            step = np.array([x_, y_, z, theta_])
            # 将转换后的步骤添加到新的序列列表中
            sequence_rel.append(step)
        # 返回转换后的序列
        return sequence_rel

    def generate_step_sequence(self, **kwargs):

        step_size, step_gap, step_height, num_steps, curved, lateral = kwargs.values()
        
        # 横向步序列
        if lateral:
            sequence = []
            y = 0  # 初始化y坐标
            c = np.random.choice([-1, 1])  # 随机选择方向，左或右
            # 生成步序列
            for i in range(1, num_steps):
                # 如果步数是奇数
                if i % 2:
                    y += step_size  # 向当前方向移动一个步长
                else:
                    y -= (2/3)*step_size  # 向相反方向移动2/3步长
                step = np.array([0, c*y, 0, 0])  # 创建步骤，x坐标为0，z坐标为0，不旋转
                sequence.append(step)  # 添加步骤到序列中
            return sequence
        
        # 如果不是横向步序列，则生成直线路径
        sequence = []
        # 如果当前相位是周期的一半
        if self._phase == (0.5 * self._period):
            # 随机生成第一步的y坐标，向左或向右
            first_step = np.array([0, -1*np.random.uniform(0.095, 0.105), 0, 0])
            y = -step_gap  # 设置y坐标的初始偏移量
        else:
            # 随机生成第一步的y坐标，向左或向右
            first_step = np.array([0, 1*np.random.uniform(0.095, 0.105), 0, 0])
            y = step_gap  # 设置y坐标的初始偏移量
        sequence.append(first_step)  # 将第一步添加到序列中
        
        x, z = 0, 0  # 初始化x和z坐标
        c = np.random.randint(2, 4)  # 随机选择一个整数，用于控制前几步的高度
        # 生成剩余的步序列
        for i in range(1, num_steps):
            x += step_size  # 增加x坐标，向前进
            y *= -1  # y坐标交替方向
            # 如果步数大于c，增加z坐标，即步高
            if i > c:
                z += step_height
            step = np.array([x, y, z, 0])  # 创建步骤，旋转角度为0
            sequence.append(step)  # 添加步骤到序列中
        
        return sequence

    def update_goal_steps(self):
        # 用于更新机器人的目标步骤。这些目标步骤是相对于机器人当前根节点的位置和方向来计算的。
        # 方法首先将目标步骤的坐标和角度初始化为零，然后根据序列中的目标时间点计算每个步骤的绝对位置和旋转。
        # 接着，将这些绝对位置转换为相对于当前根节点的相对位置，并将结果存储在类的成员变量中。
        # 这个方法在机器人行走或执行其他移动任务时用于更新目标步骤，以确保机器人朝着正确的方向移动。

        # 初始化目标步骤的x, y, z, theta坐标为两个全0的数组
        # 这里的"2"代表两个目标步骤
        self._goal_steps_x[:] = np.zeros(2)
        self._goal_steps_y[:] = np.zeros(2)
        self._goal_steps_z[:] = np.zeros(2)
        self._goal_steps_theta[:] = np.zeros(2)
        
        # 获取根节点的位置和四元数（用于表示旋转）
        root_pos = self._client.get_object_xpos_by_name(self._root_body_name, 'OBJ_BODY')
        root_quat = self._client.get_object_xquat_by_name(self._root_body_name, 'OBJ_BODY')
        
        # 遍历目标时间点self.t1和self.t2
        for idx, t in enumerate([self.t1, self.t2]):
            # 构建参考坐标系，使用根节点的位置、四元数和单位向量
            ref_frame = tf3.affines.compose(root_pos, tf3.quaternions.quat2mat(root_quat), np.ones(3))
            
            # 获取绝对目标位置和旋转（从序列中）
            abs_goal_pos = self.sequence[t][0:3]  # 位置（x, y, z）
            abs_goal_rot = tf3.euler.euler2mat(0, 0, self.sequence[t][3])  # 旋转（只考虑偏航角）
            
            # 构建绝对目标坐标系的变换矩阵
            absolute_target = tf3.affines.compose(abs_goal_pos, abs_goal_rot, np.ones(3))
            
            # 将绝对目标位置转换为相对于参考坐标系的位置
            relative_target = np.linalg.inv(ref_frame).dot(absolute_target)
            
            # 如果当前模式不是站立模式，则更新目标步骤的坐标
            if self.mode != WalkModes.STANDING:
                # 更新目标步骤的x, y, z坐标
                self._goal_steps_x[idx] = relative_target[0, 3]
                self._goal_steps_y[idx] = relative_target[1, 3]
                self._goal_steps_z[idx] = relative_target[2, 3]
                # 更新目标步骤的偏航角（theta）
                self._goal_steps_theta[idx] = tf3.euler.mat2euler(relative_target[:3, :3])[2]
        
        return


    def update_target_steps(self):
        # 用于更新机器人在序列中的目标时间点。这个方法的主要作用是移动目标时间点，以便机器人知道接下来要走向序列中的哪个位置。
        # 确保序列长度大于0，否则抛出断言错误
        assert len(self.sequence) > 0
        
        # 更新第一个目标时间点t1为上一个目标时间点t2的值
        self.t1 = self.t2
        
        # 更新第二个目标时间点t2，递增1
        self.t2 += 1
        
        # 如果新的目标时间点t2等于序列长度，说明已达到序列的末尾
        if self.t2 == len(self.sequence):
            # 将t2设置回序列的最后一个元素索引，以保持在序列范围内
            self.t2 = len(self.sequence) - 1
        
        return

    def step(self):
        # 增加步相（用于步态周期的计数器）
        self._phase += 1
        # 如果步相大于或等于周期长度，重置步相为0
        if self._phase >= self._period:
            self._phase = 0
        
        # 获取左脚和右脚的旋转四元数
        self.l_foot_quat = self._client.get_object_xquat_by_name('lf_force', 'OBJ_SITE')
        self.r_foot_quat = self._client.get_object_xquat_by_name('rf_force', 'OBJ_SITE')
        
        # 获取左脚和右脚的位置
        self.l_foot_pos = self._client.get_object_xpos_by_name('lf_force', 'OBJ_SITE')
        self.r_foot_pos = self._client.get_object_xpos_by_name('rf_force', 'OBJ_SITE')
        
        # 获取左脚和右脚的速度
        self.l_foot_vel = self._client.get_lfoot_body_vel()[0]
        self.r_foot_vel = self._client.get_rfoot_body_vel()[0]
        
        # 获取左脚和右脚的地面反作用力
        self.l_foot_frc = self._client.get_lfoot_grf()
        self.r_foot_frc = self._client.get_rfoot_grf()
        
        # 检查是否到达目标
        target_pos = self.sequence[self.t1][0:3]  # 获取当前目标位置
        # 计算两只脚到目标的距离
        foot_dist_to_target = min([np.linalg.norm(ft - target_pos) for ft in [self.l_foot_pos, self.r_foot_pos]])
        # 检查左脚和右脚是否在目标范围内
        lfoot_in_target = (np.linalg.norm(self.l_foot_pos - target_pos) < self.target_radius)
        rfoot_in_target = (np.linalg.norm(self.r_foot_pos - target_pos) < self.target_radius)
        # 如果任意一只脚在目标范围内，设置目标到达标志为True
        if lfoot_in_target or rfoot_in_target:
            self.target_reached = True
            self.target_reached_frames += 1  # 目标到达帧数递增
        else:
            self.target_reached = False
            self.target_reached_frames = 0  # 重置目标到达帧数
        
        # 如果到达目标并且在延迟帧数内，更新目标步骤
        if self.target_reached and (self.target_reached_frames >= self.delay_frames):
            self.update_target_steps()
            self.target_reached = False  # 重置目标到达标志
            self.target_reached_frames = 0  # 重置目标到达帧数
        
        # 更新目标
        self.update_goal_steps()
        
        return

    def substep(self):
        pass

    def done(self):
        # 检查是否存在不良碰撞
        contact_flag = self._client.check_bad_collisions()
        
        # 获取根节点的位置（_root_body_name是根节点的名称）
        qpos = self._client.get_object_xpos_by_name(self._root_body_name, 'OBJ_BODY')
        
        # 获取左右脚位置中z轴坐标最小的值，这通常表示脚与地面的接触点
        foot_pos = min([c[2] for c in (self.l_foot_pos, self.r_foot_pos)])
        
        # 计算根节点相对于脚的位置的高度差
        root_rel_height = qpos[2] - foot_pos
        
        # 定义终止条件字典，其中包含了两个条件：
        # 1. 根节点的相对高度小于0.6（用来检测机器人是否跌倒或处于不稳定状态）
        # 2. 存在不良碰撞（contact_flag为True）
        terminate_conditions = {"qpos[2]_ll":(root_rel_height < 0.6),
                                "contact_flag":contact_flag,
                                }
        done = True in terminate_conditions.values()
    
        return done

    def reset(self, iter_count=0):
        # 设置训练迭代次数
        self.iteration_count = iter_count
        
        # 初始化目标步骤的坐标和角度
        self._goal_steps_x = [0, 0]
        self._goal_steps_y = [0, 0]
        self._goal_steps_z = [0, 0]
        self._goal_steps_theta = [0, 0]
        
        # 设置目标到达的半径阈值
        self.target_radius = 0.20
        
        # 计算延迟帧数，即到达目标后需要等待的帧数
        self.delay_frames = int(np.floor(self._swing_duration/self._control_dt))
        
        # 初始化是否到达目标的标志和到达目标后的帧数计数器
        self.target_reached = False
        self.target_reached_frames = 0
        
        # 初始化时间步计数器
        self.t1 = 0
        self.t2 = 0
        
        # 创建奖励函数，用于根据步态周期进行奖励计算
        self.right_clock, self.left_clock = rewards.create_phase_reward(self._swing_duration,
                                                                        self._stance_duration,
                                                                        0.1,
                                                                        "grounded",
                                                                        1/self._control_dt)
        
        # 计算一个完整周期内的控制步数（一个完整周期包括左摆动和右摆动）
        self._period = np.floor(2*self._total_duration*(1/self._control_dt))
        
        # 在初始化时随机化相位
        self._phase = int(np.random.choice([0, self._period/2]))
        
        # 生成步序列
        # 随机选择一种行走模式
        # self.mode = np.random.choice(
        #     [WalkModes.STANDING, WalkModes.BACKWARD, WalkModes.LATERAL, WalkModes.FORWARD],
        #     p=[0.15, 0.25, 0.3, 0.3])
        self.mode = np.random.choice(
            [WalkModes.STANDING, WalkModes.BACKWARD, WalkModes.LATERAL, WalkModes.FORWARD],
            p=[0.2, 0, 0, 0.8])
        
        # 初始化参数字典，用于生成步态序列
        d = {'step_size':0.3, 'step_gap':0.15, 'step_height':0, 'num_steps':20, 'curved':False, 'lateral':False}
        
        # print("mode = ",self.mode)

        # 根据模式生成步态序列
        if self.mode == WalkModes.STANDING:
            d['num_steps'] = 1
        elif self.mode == WalkModes.BACKWARD:
            d['step_size'] = -0.1
        elif self.mode == WalkModes.LATERAL:
            d['step_size'] = 0.4
            d['lateral'] = True
        elif self.mode == WalkModes.FORWARD:
            # 根据迭代次数调整步高
            h = np.clip((self.iteration_count-3000)/8000, 0, 1)*0.1
            d['step_height']=np.random.choice([-h, h])
        else:
            # 如果模式无效，抛出异常
            raise Exception("Invalid WalkModes")
        
        # 生成步态序列
        sequence = self.generate_step_sequence(**d)
        # 转换步态序列
        self.sequence = self.transform_sequence(sequence)
        # 更新目标步骤
        self.update_target_steps()
        
        # 如果模式是向前走，将地形的position设置为远离机器人的位置
        if self.mode == WalkModes.FORWARD:
            self._client.model.geom('floor').pos[:] = np.array([0, 0, -100])
