"""Tiny I1 MuJoCo scene with shared overhead camera."""
from pathlib import Path
import mujoco
import numpy as np

XML = """
<mujoco model="i1_clean">
  <compiler angle="radian"/>
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <worldbody>
    <light pos="0 -1 2" diffuse="0.9 0.9 0.9"/>
    <camera name="overview" pos="0.45 -1.35 1.25" xyaxes="1 0 0 0 0.65 0.76"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba=".82 .84 .88 1"/>
    <geom name="table" type="box" pos="0.38 0 0.32" size=".55 .45 .025" rgba=".62 .5 .4 1"/>
    <body name="arm_base" pos="0 0 0.39">
      <joint name="shoulder" type="hinge" axis="0 -1 0" range="-2.5 2.5" limited="true" damping="2"/>
      <geom type="capsule" fromto="0 0 0 .35 0 0" size=".025" rgba=".25 .45 .7 1" mass=".8"/>
      <body pos=".35 0 0">
        <joint name="elbow" type="hinge" axis="0 -1 0" range="-2.6 2.6" limited="true" damping="1"/>
        <geom type="capsule" fromto="0 0 0 .30 0 0" size=".021" rgba=".2 .3 .65 1" mass=".5"/>
        <site name="ee" pos=".30 0 0" size=".025" rgba="0 1 0 1"/>
      </body>
    </body>
    <body name="human_hand" mocap="true" pos=".45 0 .48">
      <geom type="sphere" size=".035" rgba=".9 .55 .35 1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_position" joint="shoulder" kp="55"/>
    <position name="elbow_position" joint="elbow" kp="40"/>
  </actuator>
</mujoco>
"""

class I1Sim:
    def __init__(self):
        self.model=mujoco.MjModel.from_xml_string(XML)
        self.data=mujoco.MjData(self.model)
        self.data.qpos[:2]=np.array([0.3,-0.6])
        self.data.ctrl[:]=self.data.qpos[:2]
        self.hand_id=self.model.body("human_hand").mocapid[0]
        mujoco.mj_forward(self.model,self.data)

    def human_xyz(self,t):
        # Deterministic one-shot hand presentation, same in both expert runs.
        s=np.clip(t/2.0,0,1)
        s=s*s*(3-2*s)
        return np.array([0.42+0.06*s, 0.0, 0.48+0.05*s])

    def tick(self,q_command):
        self.data.mocap_pos[self.hand_id]=self.human_xyz(self.data.time)
        self.data.ctrl[:]=q_command
        mujoco.mj_step(self.model,self.data)

    def frame(self,renderer):
        renderer.update_scene(self.data,camera="overview")
        return renderer.render().copy()
