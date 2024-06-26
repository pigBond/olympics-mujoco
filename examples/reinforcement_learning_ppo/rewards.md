1. `_calc_fwd_vel_reward`
   - 参数：无
   - 范围：[0, 1]。由于使用了指数函数，当误差为0时，奖励为1；随着误差的增加，奖励指数级减小，但不会变为负数。
   - 计算方式：该函数计算智能体的前进速度奖励。它首先获取根节点的速度`root_vel`，然后计算这个速度与目标速度参考`self._goal_speed_ref`之间的欧几里得距离`error`。奖励是`np.exp(-error)`，这意味着如果智能体的前进速度接近目标速度，奖励将接近1；如果速度差异大，奖励将减小。
   
2. `_calc_action_reward`
   - 参数：`prev_action`（前一个动作）
   - 范围：[0, 1]。同样使用指数函数，当惩罚项为0时，奖励为1；随着惩罚项的增加，奖励减小，但不会变为负数。
   - 计算方式：该函数计算动作奖励。它比较当前动作`action`和前一个动作`prev_action`之间的差异，并计算一个惩罚项`penalty`，该惩罚项是动作差异的绝对值之和除以动作长度。奖励是`np.exp(-penalty)`，这意味着如果动作变化小，奖励将接近1；如果动作变化大，奖励将减小。
   
3. `_calc_torque_reward`
   - 参数：`prev_torque`（前一个扭矩）
   - 范围：[0, 1]。使用指数函数，当惩罚项为0时，奖励为1；随着惩罚项的增加，奖励减小，但不会变为负数。
   - 计算方式：该函数计算关节扭矩奖励。它比较当前扭矩`torque`和前一个扭矩`prev_torque`之间的差异，并计算一个惩罚项`penalty`，该惩罚项是扭矩差异的绝对值之和除以扭矩长度。奖励是`np.exp(-penalty)`，这意味着如果扭矩变化小，奖励将接近1；如果扭矩变化大，奖励将减小。
   
4. `_calc_height_reward`
   - 参数：无
   - 范围：[0, 1]。使用指数函数，当误差为0时，奖励为1；随着误差的增加，奖励减小，但不会变为负数。
   - 计算方式：该函数计算高度奖励。它首先检查脚是否与地面接触，然后计算当前高度`current_height`与接触点高度`contact_point`的差值`relative_height`。然后计算这个差值与目标高度参考`self._goal_height_ref`的绝对误差`error`。如果误差小于死区大小`deadzone_size`，则误差设为0。奖励是`np.exp(-40*np.square(error))`，这意味着如果高度误差小，奖励将接近1；如果高度误差大，奖励将减小。
   
5. `_calc_heading_reward`
   - 参数：无
   - 范围：[0, 1]。使用指数函数，当误差为0时，奖励为1；随着误差的增加，奖励减小，但不会变为负数。
   - 计算方式：该函数计算朝向奖励。它首先获取当前朝向`cur_heading`，并将其归一化。然后计算这个朝向与目标朝向（假设为[1, 0, 0]）之间的欧几里得距离`error`。奖励是`np.exp(-error)`，这意味着如果朝向接近目标朝向，奖励将接近1；如果朝向差异大，奖励将减小。
   
6. `_calc_root_accel_reward`
   - 参数：无
   - 范围：[0, 1]。使用指数函数，当误差为0时，奖励为1；随着误差的增加，奖励减小，但不会变为负数。
   - 计算方式：该函数计算根节点加速度奖励。它计算根节点的线性加速度`qacc`和线性速度`qvel`的绝对值之和的0.25倍作为误差`error`。奖励是`np.exp(-error)`，这意味着如果加速度和速度小，奖励将接近1；如果加速度和速度大，奖励将减小。
   
7. `_calc_feet_separation_reward`
   - 参数：无
   - 范围：[0, 1]。使用指数函数，当误差为0时，奖励为1；随着误差的增加，奖励减小，但不会变为负数。
   - 计算方式：该函数计算脚部分离奖励。它计算左右脚的Y轴分离距离`foot_dist`，并计算这个距离与目标分离距离0.35的差的平方`error`。如果脚部分离距离在0.40和0.30之间，则误差设为0。奖励是`np.exp(-error)`，这意味着如果脚部分离距离接近目标值，奖励将接近1；如果脚部分离距离偏离目标值，奖励将减小。
   
8. `_calc_foot_frc_clock_reward`
   - 参数：`left_frc_fn`和`right_frc_fn`（左右脚力的时钟函数）
   - 范围：[-1, 1]。使用正切函数，其值可以在-1到1之间变化。
   - 计算方式：该函数计算脚力时钟奖励。它首先计算左右脚力的归一化值`normed_left_frc`和`normed_right_frc`，然后根据时钟函数`left_frc_clock`和`right_frc_clock`计算左右脚力的得分`left_frc_score`和`right_frc_score`。奖励是左右脚力得分的平均值。
   
9. `_calc_foot_vel_clock_reward`
   - 参数：`left_vel_fn`和`right_vel_fn`（左右脚速度的时钟函数）
   - 范围：[-1, 1]。使用正切函数，其值可以在-1到1之间变化。
   - 计算方式：该函数计算脚速度时钟奖励。它首先计算左右脚速度的归一化值`normed_left_vel`和`normed_right_vel`，然后根据时钟函数`left_vel_clock`和`right_vel_clock`计算左右脚速度的得分`left_vel_score`和`right_vel_score`。奖励是左右脚速度得分的平均值。
   
10. `_calc_foot_pos_clock_reward`
    - 参数：无
    - 范围：[-2, 2]。使用正切函数，但由于是两个正切函数的和，其值可以在-2到2之间变化。
    - 计算方式：该函数计算脚位置时钟奖励。它首先计算左右脚高度的归一化值`normed_left_pos`和`normed_right_pos`，然后根据时钟函数`left_pos_clock`和`right_pos_clock`计算左右脚高度的得分`left_pos_score`和`right_pos_score`。奖励是左右脚高度得分的和。
    
11. `_calc_body_orient_reward`
    - 参数：`body_name`（身体部位名称），`quat_ref`（目标四元数朝向）
    - 范围：[-2, 2]。使用正切函数，但由于是两个正切函数的和，其值可以在-2到2之间变化。
    - 计算方式：该函数计算身体朝向奖励。它首先获取身体部位的四元数朝向`body_quat`，然后计算这个朝向与目标朝向`quat_ref`之间的内积的平方。误差是10乘以(1 - 内积平方)。奖励是`np.exp(-error)`，这意味着如果身体朝向接近目标朝向，奖励将接近1；如果朝向差异大，奖励将减小。
    
12. `_calc_joint_vel_reward`
    - 参数：`enabled`（启用的关节列表），`cutoff`（速度截断值）
    - 范围：[0, 1]。使用指数函数，当误差为0时，奖励为1；随着误差的增加，奖励减小，但不会变为负数。
    - 计算方式：该函数计算关节速度奖励。它首先获取启用的关节的速度`motor_speeds`和速度限制`motor_limits`，然后计算速度的平方和，但只包括那些速度绝对值大于截断值`cutoff`乘以速度限制的关节。误差是这个平方和的5e-6倍。奖励是`np.exp(-error)`，这意味着如果关节速度小，奖励将接近1；如果关节速度大，奖励将减小。
    
13. `_calc_joint_acc_reward`
    - 参数：无
    - 范围：[0, 正无穷)。由于是关节加速度的平方和乘以一个权重，其值可以是任意非负数。
    - 计算方式：该函数计算关节加速度奖励。它计算所有关节加速度的平方和`joint_acc_cost`，然后乘以权重`self.wp.joint_acc_weight`。奖励是这个加速度成本。
    
14. `_calc_ang_vel_reward`
    - 参数：无
    - 范围：[0, 正无穷)。由于是角速度的平方范数乘以一个权重，其值可以是任意非负数。
    - 计算方式：该函数计算角速度奖励。它计算角速度的平方范数`ang_vel_cost`，然后乘以权重`self.wp.ang_vel_weight`。奖励是这个角速度成本。
    
15. `_calc_impact_reward`
    - 参数：无
    - 范围：[0, 正无穷)。由于是冲击力的平方和乘以一个权重，其值可以是任意非负数。
    - 计算方式：该函数计算冲击奖励。它首先计算接触点的数量`ncon`，如果没有任何接触点，奖励为0。否则，它计算身体外部力的平方和`quad_impact_cost`，然后除以接触点数量。奖励是这个冲击成本乘以权重`self.wp.impact_weight`。
    
16. `_calc_zmp_reward`
    - 参数：无
    - 范围：[0, 正无穷)。由于是ZMP误差的平方乘以一个权重，其值可以是任意非负数。
    - 计算方式：该函数计算零力矩点（ZMP）奖励。它首先估计当前的ZMP`current_zmp`，然后计算这个ZMP与目标ZMP`desired_zmp`之间的欧几里得距离的平方`zmp_cost`。奖励是这个ZMP成本乘以权重`self.wp.zmp_weight`。
    
17. `_calc_foot_contact_reward`
    - 参数：无
    - 范围：[0, 正无穷)。由于是脚与地面接触点的距离之和乘以一个权重，其值可以是任意非负数。
    - 计算方式：该函数计算脚接触奖励。它首先获取左右脚的接触点，然后计算这些接触点与基点的距离`c_dist_r`和`c_dist_l`。奖励是这些距离中大于阈值`radius_thresh`的距离之和乘以权重