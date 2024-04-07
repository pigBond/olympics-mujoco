import mujoco
import mujoco.viewer

from mushroom_rl.utils.running_stats import *
from mushroom_rl.utils.mujoco import *
from dm_control import mjcf


def load_model(xml_file):
    """
        Takes an xml_file and compiles and loads the model.

        Args:
            xml_file (str/xml handle): A string with a path to the xml or an Mujoco xml handle.

        Returns:
            Mujoco model.

    """
    if type(xml_file) == mjcf.element.RootElement:
        # load from xml handle
        model = mujoco.MjModel.from_xml_string(xml=xml_file.to_xml_string(),
                                                   assets=xml_file.get_assets())
    elif type(xml_file) == str:
        # load from path
        model = mujoco.MjModel.from_xml_path(xml_file)
    else:
        raise ValueError(f"Unsupported type for xml_file {type(xml_file)}.")

    return model



xml_path="/home/wzx/new-Github-Workspaces/olympics-mujoco/olympic_mujoco/environments/data/jvrc_step/jvrc1.xml"
xml_handles = mjcf.from_path(xml_path)
_model=load_model(xml_handles)
_data = mujoco.MjData(_model)

mujoco.viewer.launch(_model, _data)