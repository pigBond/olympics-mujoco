# Olympics
from .UnitreeH1 import UnitreeH1
from .StickFigureA1 import StickFigureA1
import olympic_mujoco

# register environments in mushroom
UnitreeH1.register()
StickFigureA1.register()

from gymnasium import register
# register gymnasium wrapper environment
register("OlympicMujoco",
entry_point="olympic_mujoco.environments.gymnasium:GymnasiumWrapper"
)
