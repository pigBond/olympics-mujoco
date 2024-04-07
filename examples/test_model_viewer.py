import numpy as np
import os
import sys
sys.path.append(os.getcwd())

from olympic_mujoco.environments.loco_env_base import LocoEnvBase


env = LocoEnvBase.make("Jvrc.run.real")
