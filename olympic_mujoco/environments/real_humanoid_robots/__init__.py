# Olympics
from .UnitreeH1 import UnitreeH1
from .atlas import Atlas
from .talos import Talos
from .StickFigureA1 import StickFigureA1
from .StickFigureA3 import StickFigureA3
from .Jvrc import Jvrc
import olympic_mujoco

# register environments in mushroom
UnitreeH1.register()
Atlas.register()
Talos.register()
StickFigureA1.register()
StickFigureA3.register()
Jvrc.register()

from gymnasium import register
# register gymnasium wrapper environment
register("OlympicMujoco",
entry_point="olympic_mujoco.environments.gymnasium:GymnasiumWrapper"
)
