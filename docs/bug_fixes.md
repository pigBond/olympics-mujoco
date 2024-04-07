# Bug修复记录

本文档用于记录项目中遇到并解决的各种bug。

## Bug列表

1. **问题ID**: #001

   - **问题描述**:  播放轨迹数据时，卡在第一个动作画面，index 200 is out of bounds for axis 0 with size 200

   - **原因**:  

     ```python
     # 如果样本为None,表示到达轨迹末尾,重置环境            
     if sample is None:
     	sample = self.trajectories.get_current_sample()
     	curr_qpos = sample[0:len_qpos]
     ```

   - **报错信息**：

     ```
     Traceback (most recent call last):
       File "examples/test.py", line 31, in <module>
         experiment()
       File "examples/test.py", line 24, in experiment
         mdp.play_trajectory_from_velocity(n_episodes=3, n_steps_per_episode=500)
       File "/home/wzx/new-Github-Workspaces/olympics-mujoco/olympic_mujoco/environments/loco_env_base.py", line 411, in play_trajectory_from_velocity
         sample = self.trajectories.get_current_sample()
       File "/home/wzx/new-Github-Workspaces/olympics-mujoco/olympic_mujoco/utils/trajectory.py", line 387, in get_current_sample
         return self._get_ith_sample_from_subtraj(self.subtraj_step_no)
       File "/home/wzx/new-Github-Workspaces/olympics-mujoco/olympic_mujoco/utils/trajectory.py", line 472, in _get_ith_sample_from_subtraj
         return [np.array(obs[i].copy()).flatten() for obs in self.subtraj]
       File "/home/wzx/new-Github-Workspaces/olympics-mujoco/olympic_mujoco/utils/trajectory.py", line 472, in <listcomp>
         return [np.array(obs[i].copy()).flatten() for obs in self.subtraj]
     IndexError: index 200 is out of bounds for axis 0 with size 200
     ```

   - **解决方案**: 

     ```python
     # 如果样本为None,表示到达轨迹末尾,重置环境            
     if sample is None:
     	self.reset()
     	sample = self.trajectories.get_current_sample()
     	curr_qpos = sample[0:len_qpos]
     ```

     使用`self.reset() `进行重置,关键在于`sample = self.trajectories.reset_trajectory()`

     `reset_trajectory()`函数会重置轨迹到一个特定的轨迹以及该轨迹内的一个子步骤，会保证轨迹编号在有效范围内

   - **解决日期**: 2024-03-29

   - **负责人**: wzx

2.**问题ID**: #002

- **问题描述**:  使用jvrc的简易化模型时，渲染不完全，只显示机器人下半身，并且丢失机器人的皮肤

- **原因**: 猜测可能是因为使用mushroom rl的渲染，或者是因为使用xml_handle导致无法访问mesh或stl的文件
      
     ```python
    def play_trajectory_from_velocity(...)
      ...
      if render:
        # 这里使用mushroom rl内嵌的viewer组件,导致无法正常完整渲染xml模型
        # 注意本函数使用了两次self.render()
        frame = self.render(record)
        # self.test_render()
      else:
        frame = None
    ```
- **报错信息**：

- **解决方案**: 
    ```python
    # 重写render方法 使用mujoco_viewer库 后续有时间这里应修改成自己的viewer库
    def render(self):
      if self.viewer is None:
        self.viewer = mujoco_viewer.MujocoViewer(self._model, self._data)
        self.viewer_setup()
      self.viewer.render()
    ```
- **解决日期**: 2024-04-07

- **负责人**: wzx



3.**问题ID**: #003

- **问题描述**: 数据库在高峰时段响应缓慢。
- **原因**: 没有对热门数据进行索引。
- **报错信息**：
- **解决方案**: 为热门查询添加索引。
- **解决日期**: 2024-03-xx
- **负责人**: wzx



## 附录

- **问题ID**格式说明：`#`后面跟三位数字，依次递增。
- **负责人**：wzx

---

**更新记录**:
- 2024-03-29: 初始创建文档
- 2024-03-29: 添加问题ID #001的记录
- 2024-03-29: 解决问题ID #001
- 2024-04-06: 添加问题ID #002的记录
- 2024-04-07: 解决问题ID #002