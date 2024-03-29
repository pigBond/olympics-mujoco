import numpy as np
import transforms3d as tf3
import mujoco
"""
nq(): 获取模型的配置状态(qpos)的维度。
nu(): 获取模型的执行器数量。
nv(): 获取模型的广义坐标数量。
sim_dt(): 获取模拟的时间步长。
get_robot_mass(): 获取机器人的总质量。
get_qpos(): 获取模型的配置状态(qpos)。
get_qvel(): 获取模型的广义速度(qvel)。
get_qacc(): 获取模型的广义加速度(qacc)。
get_cvel(): 获取模型的中心速度(cvel)。
get_jnt_id_by_name(name): 获取指定名称的关节的 ID。
get_jnt_qposadr_by_name(name): 获取指定名称的关节的配置状态(qpos)地址。
get_jnt_qveladr_by_name(name): 获取指定名称的关节的速度(qvel)地址。
get_body_ext_force(): 获取作用于身体的扩展力。
get_motor_speed_limits(): 获取执行器的速度限制。
get_act_joint_speed_limits(): 获取关节的速度限制。
get_gear_ratios(): 获取传动比。
get_motor_names(): 获取执行器的名称列表。
get_actuated_joint_inds(): 获取与执行器连接的关节索引列表。
get_actuated_joint_names(): 获取与执行器连接的关节名称列表。
get_motor_qposadr(): 获取所有驱动关节的配置状态地址列表。
get_motor_positions(): 获取执行器的位置。
get_motor_velocities(): 获取执行器的速度。
get_act_joint_torques(): 获取关节空间中的执行器力。
get_act_joint_positions(): 获取关节级别的执行器位置。
get_act_joint_velocities(): 获取关节级别的执行器速度。
get_act_joint_range(): 获取所有驱动关节的极限。
get_actuator_ctrl_range(): 获取模型 XML 中定义的执行器控制范围。
get_actuator_user_data(): 获取附加到每个执行器的用户数据(如果有的话)。
get_root_body_pos(): 获取根身体的位置。
get_root_body_vel(): 获取根身体的速度。
get_sensordata(sensor_name): 获取指定名称的传感器的数据。
get_rfoot_body_pos(): 获取右脚的位置。
get_lfoot_body_pos(): 获取左脚的位置。
get_rfoot_floor_contacts(): 获取右脚与地面接触的信息。
get_lfoot_floor_contacts(): 获取左脚与地面接触的信息。
get_rfoot_grf(): 获取右脚的总地面反作用力。
get_lfoot_grf(): 获取左脚的总地面反作用力。
get_body_vel(body_name, frame=0): 获取指定身体在身体中心框架中的平移和旋转速度。
get_rfoot_body_vel(frame=0): 获取右脚的速度。
get_lfoot_body_vel(frame=0): 获取左脚的速度。
get_object_xpos_by_name(obj_name, object_type): 获取指定名称的物体的位置。
get_object_xquat_by_name(obj_name, object_type): 获取指定名称的物体的四元数。
get_robot_com(): 获取机器人的质心位置。
get_robot_linmom(): 获取机器人的线动量。
get_robot_angmom(): 获取机器人的角动量。
check_rfoot_floor_collision(): 检查右脚与地面是否发生碰撞。
check_lfoot_floor_collision(): 检查左脚与地面是否发生碰撞。
check_bad_collisions(): 检查除脚部与地面之外的碰撞。
check_self_collisions(): 检查除任何物体与地面之外的自我碰撞。
get_pd_target(): 获取 PD 控制的目标值。
set_pd_gains(kp, kv): 设置 PD 控制的增益。
step_pd(p, v): 执行 PD 控制的步进。
set_motor_torque(torque): 应用扭矩到执行器。
step(): 模拟步进。
"""

class MujocoRobotInterface(object):
    def __init__(self, model, data, rfoot_body_name=None, lfoot_body_name=None):
        self.model = model
        self.data = data

        self.rfoot_body_name = rfoot_body_name
        self.lfoot_body_name = lfoot_body_name
        self.floor_body_name = 'world'

    def test(self):
        return "12138"

    def nq(self):
        return self.model.nq

    def nu(self):
        return self.model.nu

    def nv(self):
        return self.model.nv

    def sim_dt(self):
        return self.model.opt.timestep

    def get_robot_mass(self):
        return mujoco.mj_getTotalmass(self.model)

    def get_qpos(self):
        return self.data.qpos.copy()

    def get_qvel(self):
        return self.data.qvel.copy()

    def get_qacc(self):
        return self.data.qacc.copy()

    def get_cvel(self):
        return self.data.cvel.copy()

    def get_jnt_id_by_name(self, name):
        return self.model.joint(name)

    def get_jnt_qposadr_by_name(self, name):
        return self.model.joint(name).qposadr

    def get_jnt_qveladr_by_name(self, name):
        return self.model.joint(name).dofadr

    def get_body_ext_force(self):
        return self.data.cfrc_ext.copy()

    def get_motor_speed_limits(self):
        """
        Returns speed limits of the *actuator* in radians per sec.
        This assumes the actuator 'user' element defines speed limits
        at the actuator level in revolutions per minute.
        """
        rpm_limits = self.model.actuator_user[:,0] # RPM
        return ((rpm_limits)*(2*np.pi/60)).tolist() # radians per sec

    def get_act_joint_speed_limits(self):
        """
        Returns speed limits of the *joint* in radians per sec.
        This assumes the actuator 'user' element defines speed limits
        at the actuator level in revolutions per minute.
        """
        gear_ratios = self.model.actuator_gear[:,0]
        mot_lims = self.get_motor_speed_limits()
        return [float(i/j) for i,j in zip(mot_lims, gear_ratios)]

    def get_gear_ratios(self):
        """
        Returns transmission ratios.
        """
        return self.model.actuator_gear[:,0]

    def get_motor_names(self):
        actuator_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)]
        return actuator_names

    def get_actuated_joint_inds(self):
        """
        Returns list of joint indices to which actuators are attached.
        """
        joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)]
        actuator_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)]
        return [idx for idx, jnt in enumerate(joint_names) if jnt+'_motor' in actuator_names]

    def get_actuated_joint_names(self):
        """
        Returns list of joint names to which actuators are attached.
        """
        joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)]
        actuator_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)]
        return [jnt for idx, jnt in enumerate(joint_names) if jnt+'_motor' in actuator_names]

    def get_motor_qposadr(self):
        """
        Returns the list of qpos indices of all actuated joints.
        """
        indices = self.get_actuated_joint_inds()
        return [self.model.jnt_qposadr[i] for i in indices]

    def get_motor_positions(self):
        """
        Returns position of actuators.
        """
        return self.data.actuator_length.tolist()

    def get_motor_velocities(self):
        """
        Returns velocities of actuators.
        """
        return self.data.actuator_velocity.tolist()

    def get_act_joint_torques(self):
        """
        Returns actuator force in joint space.
        """
        gear_ratios = self.model.actuator_gear[:,0]
        motor_torques = self.data.actuator_force.tolist()
        return [float(i*j) for i,j in zip(motor_torques, gear_ratios)]

    def get_act_joint_positions(self):
        """
        Returns position of actuators at joint level.
        """
        gear_ratios = self.model.actuator_gear[:,0]
        motor_positions = self.get_motor_positions()
        return [float(i/j) for i,j in zip(motor_positions, gear_ratios)]

    def get_act_joint_velocities(self):
        """
        Returns velocities of actuators at joint level.
        """
        gear_ratios = self.model.actuator_gear[:,0]
        motor_velocities = self.get_motor_velocities()
        return [float(i/j) for i,j in zip(motor_velocities, gear_ratios)]

    def get_act_joint_range(self):
        """
        Returns the lower and upper limits of all actuated joints.
        """
        indices = self.get_actuated_joint_inds()
        low, high = self.model.jnt_range[indices, :].T
        return low, high

    def get_actuator_ctrl_range(self):
        """
        Returns the acutator ctrlrange defined in model xml.
        """
        low, high = self.model.actuator_ctrlrange.copy().T
        return low, high

    def get_actuator_user_data(self):
        """
        Returns the user data (if any) attached to each actuator.
        """
        return self.model.actuator_user.copy()

    def get_root_body_pos(self):
        return self.data.xpos[1].copy()

    def get_root_body_vel(self):
        qveladr = self.get_jnt_qveladr_by_name("root")
        return self.data.qvel[qveladr:qveladr+6].copy()

    def get_sensordata(self, sensor_name):
        sensor_id = self.model.sensor(sensor_name)
        sensor_adr = self.model.sensor_adr[sensor_id]
        data_dim = self.model.sensor_dim[sensor_id]
        return self.data.sensordata[sensor_adr:sensor_adr+data_dim]

    def get_rfoot_body_pos(self):
        return self.data.body(self.rfoot_body_name).xpos.copy()

    def get_lfoot_body_pos(self):
        return self.data.body(self.lfoot_body_name).xpos.copy()

    def get_rfoot_floor_contacts(self):
        """
        Returns list of right foot and floor contacts.
        """
        contacts = [self.data.contact[i] for i in range(self.data.ncon)]
        rcontacts = []
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.floor_body_name)
        rfoot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.rfoot_body_name)
        for i,c in enumerate(contacts):
            geom1_is_floor = (self.model.geom_bodyid[c.geom1]==floor_id)
            geom2_is_rfoot = (self.model.geom_bodyid[c.geom2]==rfoot_id)
            if (geom1_is_floor and geom2_is_rfoot):
                rcontacts.append((i,c))
        return rcontacts

    def get_lfoot_floor_contacts(self):
        """
        Returns list of left foot and floor contacts.
        """
        contacts = [self.data.contact[i] for i in range(self.data.ncon)]
        lcontacts = []
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.floor_body_name)
        lfoot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.lfoot_body_name)
        for i,c in enumerate(contacts):
            geom1_is_floor = (self.model.geom_bodyid[c.geom1]==floor_id)
            geom2_is_lfoot = (self.model.geom_bodyid[c.geom2]==lfoot_id)
            if (geom1_is_floor and geom2_is_lfoot):
                lcontacts.append((i,c))
        return lcontacts

    def get_rfoot_grf(self):
        """
        Returns total Ground Reaction Force on right foot.
        """
        right_contacts = self.get_rfoot_floor_contacts()
        rfoot_grf = 0
        for i, con in right_contacts:
            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_array)
            rfoot_grf += np.linalg.norm(c_array)
        return rfoot_grf

    def get_lfoot_grf(self):
        """
        Returns total Ground Reaction Force on left foot.
        """
        left_contacts = self.get_lfoot_floor_contacts()
        lfoot_grf = 0
        for i, con in left_contacts:
            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_array)
            lfoot_grf += (np.linalg.norm(c_array))
        return lfoot_grf

    def get_body_vel(self, body_name, frame=0):
        """
        Returns translational and rotational velocity of a body in body-centered frame, world/local orientation.
        """
        body_vel = np.zeros(6)
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY,
                                 body_id, body_vel, frame)
        return [body_vel[3:6], body_vel[0:3]]

    def get_rfoot_body_vel(self, frame=0):
        """
        Returns translational and rotational velocity of right foot.
        """
        rfoot_vel = np.zeros(6)
        rfoot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.rfoot_body_name)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY,
                                 rfoot_id, rfoot_vel, frame)
        return [rfoot_vel[3:6], rfoot_vel[0:3]]

    def get_lfoot_body_vel(self, frame=0):
        """
        Returns translational and rotational velocity of left foot.
        """
        lfoot_vel = np.zeros(6)
        lfoot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.lfoot_body_name)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY,
                                 lfoot_id, lfoot_vel, frame)
        return [lfoot_vel[3:6], lfoot_vel[0:3]]

    def get_object_xpos_by_name(self, obj_name, object_type):
        if object_type=="OBJ_BODY":
            return self.data.body(obj_name).xpos
        elif object_type=="OBJ_GEOM":
            return self.data.geom(obj_name).xpos
        elif object_type=="OBJ_SITE":
            return self.data.site(obj_name).xpos
        else:
            raise Exception("object type should either be OBJ_BODY/OBJ_GEOM/OBJ_SITE.")

    def get_object_xquat_by_name(self, obj_name, object_type):
        if object_type=="OBJ_BODY":
            return self.data.body(obj_name).xquat
        if object_type=="OBJ_SITE":
            xmat = self.data.site(obj_name).xmat
            return tf3.quaternions.mat2quat(xmat)
        else:
            raise Exception("object type should be OBJ_BODY/OBJ_SITE.")

    def get_robot_com(self):
        """
        Returns the center of mass of subtree originating at root body
        i.e. the CoM of the entire robot body in world coordinates.
        """
        sensor_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(self.model.nsensor)]
        if 'subtreecom' not in sensor_names:
            raise Exception("subtree_com sensor not attached.")
        return self.data.subtree_com[1].copy()

    def get_robot_linmom(self):
        """
        Returns linear momentum of robot in world coordinates.
        """
        sensor_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(self.model.nsensor)]
        if 'subtreelinvel' not in sensor_names:
            raise Exception("subtree_linvel sensor not attached.")
        linvel = self.data.subtree_linvel[1].copy()
        total_mass = self.get_robot_mass()
        return linvel*total_mass

    def get_robot_angmom(self):
        """
        Return angular momentum of robot's CoM about the world origin.
        """
        sensor_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(self.model.nsensor)]
        if 'subtreeangmom' not in sensor_names:
            raise Exception("subtree_angmom sensor not attached.")
        com_pos = self.get_robot_com()
        lin_mom = self.get_robot_linmom()
        return self.data.subtree_angmom[1] + np.cross(com_pos, lin_mom)


    def check_rfoot_floor_collision(self):
        """
        Returns True if there is a collision between right foot and floor.
        """
        return (len(self.get_rfoot_floor_contacts())>0)

    def check_lfoot_floor_collision(self):
        """
        Returns True if there is a collision between left foot and floor.
        """
        return (len(self.get_lfoot_floor_contacts())>0)

    def check_bad_collisions(self):
        """
        Returns True if there are collisions other than feet-floor.
        """
        num_rcons = len(self.get_rfoot_floor_contacts())
        num_lcons = len(self.get_lfoot_floor_contacts())
        return (num_rcons + num_lcons) != self.data.ncon

    def check_self_collisions(self):
        """
        Returns True if there are collisions other than any-geom-floor.
        """
        contacts = [self.data.contact[i] for i in range(self.data.ncon)]
        floor_contacts = []
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.floor_body_name)
        for i,c in enumerate(contacts):
            geom1_is_floor = (self.model.geom_bodyid[c.geom1]==floor_id)
            geom2_is_floor = (self.model.geom_bodyid[c.geom2]==floor_id)
            if (geom1_is_floor or geom2_is_floor):
                floor_contacts.append((i,c))
        return len(floor_contacts) != self.data.ncon

    def get_pd_target(self):
        return [self.current_pos_target, self.current_vel_target]

    def set_pd_gains(self, kp, kv):
        assert kp.size==self.model.nu
        assert kv.size==self.model.nu
        self.kp = kp.copy()
        self.kv = kv.copy()
        return

    def step_pd(self, p, v):
        self.current_pos_target = p.copy()
        self.current_vel_target = v.copy()
        target_angles = self.current_pos_target
        target_speeds = self.current_vel_target

        assert type(target_angles)==np.ndarray
        assert type(target_speeds)==np.ndarray

        curr_angles = self.get_act_joint_positions()
        curr_speeds = self.get_act_joint_velocities()

        perror = target_angles - curr_angles
        verror = target_speeds - curr_speeds

        assert self.kp.size==perror.size
        assert self.kv.size==verror.size
        assert perror.size==verror.size
        return self.kp * perror + self.kv * verror

    def set_motor_torque(self, torque):
        """
        Apply torques to motors.
        """
        if isinstance(torque, np.ndarray):
            assert torque.shape==(self.nu(), )
            ctrl = torque.tolist()
        elif isinstance(torque, list):
            assert len(torque)==self.nu()
            ctrl = np.copy(torque)
        else:
            raise Exception("motor torque should be list of ndarray.")
        try:
            self.data.ctrl[:] = ctrl
        except Exception as e:
            print("Could not apply motor torque.")
            print(e)
        return

    def step(self):
        """
        Increment simulation by one step.
        """
        mujoco.mj_step(self.model, self.data)
