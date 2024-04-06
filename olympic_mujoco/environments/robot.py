import numpy as np

class JVRC:
    def __init__(self, pdgains, dt, active, client):

        self.client = client
        self.control_dt = dt

        # list of desired actuators 希望激活的执行器列表
        self.actuators = active

        # set PD gains
        self.kp = pdgains[0]
        self.kd = pdgains[1]

        # print("self.kp = ",self.kp)
        # print("self.kd = ",self.kd)
        print("self.kp.shape = ",self.kp.shape)
        print("self.kd.shape = ",self.kd.shape)
        print("self.client.nu() = ",self.client.nu())


        """
        <actuator>
            <motor name="R_HIP_P_motor" class="/" joint="R_HIP_P" />
            <motor name="R_HIP_R_motor" class="/" joint="R_HIP_R" />
            <motor name="R_HIP_Y_motor" class="/" joint="R_HIP_Y" />
            <motor name="R_KNEE_motor" class="/" joint="R_KNEE" />
            <motor name="R_ANKLE_R_motor" class="/" joint="R_ANKLE_R" />
            <motor name="R_ANKLE_P_motor" class="/" joint="R_ANKLE_P" />
            <motor name="L_HIP_P_motor" class="/" joint="L_HIP_P" />
            <motor name="L_HIP_R_motor" class="/" joint="L_HIP_R" />
            <motor name="L_HIP_Y_motor" class="/" joint="L_HIP_Y" />
            <motor name="L_KNEE_motor" class="/" joint="L_KNEE" />
            <motor name="L_ANKLE_R_motor" class="/" joint="L_ANKLE_R" />
            <motor name="L_ANKLE_P_motor" class="/" joint="L_ANKLE_P" />
        </actuator>

        这里的 self.kp.shape = self.kd.shape = 12 对应这里的原版jvrc模型的motor数量,但是我现在修改了模型,有34个motor
        
        """

        assert self.kp.shape==self.kd.shape==(self.client.nu(),)
        self.client.set_pd_gains(self.kp, self.kd)
        
        # define init qpos and qvel
        self.init_qpos_ = [0] * self.client.nq()
        self.init_qvel_ = [0] * self.client.nv()

        self.prev_action = None
        self.prev_torque = None
        self.iteration_count = np.inf

        # frame skip parameter
        if (np.around(self.control_dt%self.client.sim_dt(), 6)):
            raise Exception("Control dt should be an integer multiple of Simulation dt.")
        self.frame_skip = int(self.control_dt/self.client.sim_dt())

        # define nominal pose
        base_position = [0, 0, 0.81]
        base_orientation = [1, 0, 0, 0]
        half_sitting_pose = [-30,  0, 0, 50, 0, -24,
                             -30,  0, 0, 50, 0, -24,
                             -3, -9.74, -30,
                             -3,  9.74, -30,
        ] # degrees

        # number of all joints
        self.num_joints = len(half_sitting_pose)
        
        # define init qpos and qvel
        nominal_pose = [q*np.pi/180.0 for q in half_sitting_pose]
        robot_pose = base_position + base_orientation + nominal_pose

        print("len(half_sitting_pose) = ",len(half_sitting_pose))
        print("robot_pose = ",robot_pose)
        print("len(robot_pose) = ",len(robot_pose))
        print("self.client.nq() = ",self.client.nq())

        assert len(robot_pose)==self.client.nq()
        self.init_qpos_[-len(robot_pose):] = base_position + base_orientation + nominal_pose

        # define actuated joint nominal pose
        motor_qposadr = self.client.get_motor_qposadr()
        self.motor_offset = [self.init_qpos_[i] for i in motor_qposadr]

    def step(self, action):
        # print("这里是robot.py的step函数")
        filtered_action = np.zeros(len(self.motor_offset))
        for idx, act_id in enumerate(self.actuators):
            filtered_action[act_id] = action[idx]

        # add fixed motor offset
        filtered_action += self.motor_offset

        if self.prev_action is None:
            self.prev_action = filtered_action
        if self.prev_torque is None:
            self.prev_torque = np.asarray(self.client.get_act_joint_torques())

        self.client.set_pd_gains(self.kp, self.kd)
        self.do_simulation(filtered_action, self.frame_skip)

        self.prev_action = filtered_action
        self.prev_torque = np.asarray(self.client.get_act_joint_torques())
        return filtered_action

    def do_simulation(self, target, n_frames):
        ratio = self.client.get_gear_ratios()
        for _ in range(n_frames):
            tau = self.client.step_pd(target, np.zeros(self.client.nu()))
            tau = [(i/j) for i,j in zip(tau, ratio)]
            self.client.set_motor_torque(tau)
            self.client.step()
