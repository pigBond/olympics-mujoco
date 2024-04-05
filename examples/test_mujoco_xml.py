import mujoco
import mujoco.viewer

# TODO:这里使用mujoco自带的viewer可以进行正常的皮肤加载
# locomujoco使用的是mushroom rl的viewer加载,所以可能是这个原因导致mesh无法正确加载


fullpath="/home/wzx/new-Github-Workspaces/olympics-mujoco/olympic_mujoco/environments/data/jvrc_mj_description/xml/jvrc1.xml"
model = mujoco.MjModel.from_xml_path(fullpath)
data = mujoco.MjData(model)

mujoco.viewer.launch(model, data)