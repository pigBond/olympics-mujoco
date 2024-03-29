# olympic-mujoco



1. **`LocoEnvBase`类**：
   - 这个类应该包含与MuJoCo环境交互的通用方法，如环境的初始化、渲染、步骤函数、重置环境等。
   - 可以定义一些抽象方法或接口，这些方法在子类中必须被实现，以确保所有的机器人操作类都具备某些核心功能。
2. **`BaseHumanoidRobot`类**：
   - 这个类继承自`LocoEnvBase`，应该包含所有 humanoid robot 的通用特性，例如：行走、跑步、跳跃等基础动作方法。
   - 可以在这个层面构思一些创建新数据集的方法，例如记录运动轨迹、速度、加速度等数据的方法。
3. **`UnitreeH1`类**：
   - 这个类继承自`BaseHumanoidRobot`，应该包含特定于`UnitreeH1`机器人的实现细节，比如它的机械结构、传感器数据读取、特有的动作等。
   - 对于一些特定的方法，比如与`UnitreeH1`硬件相关的控制接口，应该在这里实现。







`ObservationHelper` 类的功能：

- 管理和操作观测数据。
- 定义观测空间，包括身体、关节和站点的位置、旋转和速度。
- 提供添加、删除或检索特定观测的方法。
- 构建完整的观测数组。
- 允许根据观测修改 MuJoCo 数据，以影响模拟。

`MujocoRobotInterface` 类的功能：

- 提供了对 MuJoCo 模型和数据结构的直接访问。
- 提供了获取机器人质量、关节位置、速度、加速度等基本信息的方法。
- 提供了获取和设置关节速度限制、齿轮比率、执行器名称等参数的方法。
- 提供了获取脚部与地面接触信息的方法。
- 提供了获取和设置执行器扭矩的方法。
- 提供了单步模拟的方法。

功能对比：

- `ObservationHelper` 更专注于构建和操作观测数据，以适应强化学习中的观测空间需求。
- `MujocoRobotInterface` 提供了更广泛的机器人模拟和交互功能，包括获取传感器数据、设置控制参数、模拟步进等。





```python
class MujocoRobotInterface(object):

    def __init__(self, model, data, rfoot_body_name=None, lfoot_body_name=None):
```

由于这里使用`rfoot_body_name`和`lfoot_body_name`作为参数，所以**该类应在UnitreeH1类中实例化使用**。









