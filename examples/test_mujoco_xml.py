import mujoco
import mujoco.viewer
import mujoco_viewer

# TODO:这里使用mujoco自带的viewer可以进行正常的皮肤加载
# locomujoco使用的是mushroom rl的viewer加载,所以可能是这个原因导致mesh无法正确加载


# fullpath="/home/wzx/new-Github-Workspaces/olympics-mujoco/olympic_mujoco/environments/data/jvrc_mj_description/xml/jvrc1.xml"
# fullpath="/home/wzx/new-Github-Workspaces/olympics-mujoco/olympic_mujoco/environments/data/jvrc_step/jvrc1.xml"
fullpath="/home/wzx/new-Github-Workspaces/olympics-mujoco/olympic_mujoco/environments/data/stickFigure_A1/a1.xml"
model = mujoco.MjModel.from_xml_path(fullpath)
data = mujoco.MjData(model)

# mujoco.viewer.launch(model, data)

viewer=mujoco_viewer.MujocoViewer(model, data)


# viewer.cam.trackbodyid = 1
# viewer.cam.distance = model.stat.extent * 1.5
# viewer.cam.lookat[2] = 1.5
# viewer.cam.lookat[0] = 2.0
# viewer.cam.elevation = -20
# viewer.vopt.geomgroup[0] = 1
# viewer._render_every_frame = True

while True:
    viewer.render()