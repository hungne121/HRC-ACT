"""Seeded I1 MuJoCo scene: 3 cups, 3 coasters, a 3-joint arm and moving human hand.

The coasters are kinematic objects while carried and after placement, with contact
geometry; cups are free rigid bodies. Human is a kinematic proxy, not biomechanical
simulation. An active MuJoCo weld holds a cup during transport.
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
import numpy as np
import mujoco

TABLE_Z = 0.405
CUP_H = 0.075
CUP_R = 0.025
BASE = np.array([0.40, -0.58, 0.53], dtype=float)
L1, L2 = 0.50, 0.46
SOURCE = np.array([[0.22, -0.26, TABLE_Z + CUP_H/2],
                   [0.40, -0.26, TABLE_Z + CUP_H/2],
                   [0.58, -0.26, TABLE_Z + CUP_H/2]])
AREAS = np.array([0.20, 0.40, 0.60])
COASTER_R = 0.048

@dataclass(frozen=True)
class HumanClip:
    seed: int
    targets: np.ndarray
    order: np.ndarray
    clip_id: str

    @classmethod
    def sample(cls, seed: int) -> "HumanClip":
        rng = np.random.default_rng(seed)
        targets = np.column_stack([AREAS + rng.uniform(-0.045,0.045,3),
                                   rng.uniform(0.16,0.25,3),
                                   np.full(3,TABLE_Z+0.007)])
        order = rng.permutation(3)
        digest = hashlib.sha256(
            json.dumps({"seed":seed,"goals":targets.tolist(),
                        "order":order.tolist()},sort_keys=True).encode()
        ).hexdigest()[:12]
        return cls(seed,targets,order,digest)

    def phase(self,t:float, turn:int):
        # Each turn is exactly 5.0 s; deterministic across expert conditions.
        return float(np.clip(t-5.0*turn,0,5))

    def pose(self,t:float):
        turn=min(int(t//5.0),2)
        p=self.phase(t,turn)
        goal=self.targets[int(self.order[turn])]
        home=np.array([0.77,0.35,TABLE_Z+0.22])
        arrival=np.array([goal[0],goal[1],TABLE_Z+0.11])
        down=np.array([goal[0],goal[1],TABLE_Z+0.045])
        def mix(a,b,u):
            s=np.clip(u,0,1)
            s=s*s*(3-2*s)
            return a*(1-s)+b*s
        if p<0.8: pos=home.copy()
        elif p<2.5: pos=mix(home,arrival,(p-0.8)/1.7)
        elif p<3.0: pos=mix(arrival,down,(p-2.5)/0.5)
        elif p<3.35: pos=down.copy()
        elif p<4.25: pos=mix(down,home,(p-3.35)/0.90)
        else: pos=home.copy()
        return turn,p,pos

    def coaster_poses(self,t:float):
        poses=np.tile(np.array([0.84,0.37,TABLE_Z+0.012]),(3,1))
        poses[:,0] += np.arange(3)*0.06
        turn,p,hand=self.pose(t)
        for j,coaster in enumerate(self.order):
            if j<turn or (j==turn and p>=3.0):
                poses[int(coaster)]=self.targets[int(coaster)]
            elif j==turn:
                poses[int(coaster)]=hand+np.array([0,0,-0.04])
        return poses

    def ready(self,t:float,coaster_index:int):
        matches=np.flatnonzero(self.order==coaster_index)
        if len(matches)!=1: return False
        return t >= float(matches[0])*5.0+4.0

def scene_xml():
    cups=[]
    coasters=[]
    welds=[]
    for i,pos in enumerate(SOURCE):
        cups.append(f"""
        <body name="cup{i}" pos="{pos[0]} {pos[1]} {pos[2]}">
          <freejoint/>
          <geom name="cup_geom{i}" type="cylinder" size="{CUP_R} {CUP_H/2}" mass=".13"
                rgba=".87 .88 .94 1" friction="1 0.01 0.001"/>
        </body>""")
        coasters.append(f"""
        <body name="coaster{i}" mocap="true" pos=".9 .4 {TABLE_Z+0.007}">
          <geom name="coaster_geom{i}" type="cylinder" size="{COASTER_R} .007"
                rgba="{.2+.2*i} {.44+.1*i} {.65-.1*i} 1"
                contype="1" conaffinity="1" friction="1 0.01 0.001"/>
        </body>""")
        welds.append(f'<weld name="grasp{i}" body1="gripper" body2="cup{i}" active="false" relpose="0 0 0 1 0 0 0" solref="0.006 1"/>')
    return f"""
<mujoco model="I1_V2">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual><global offwidth="960" offheight="640"/></visual>
  <default><joint damping="2" armature=".04"/><geom condim="3" friction="1 .03 .002"/></default>
  <worldbody>
    <light pos="0.45 -.1 1.8" directional="false"/>
    <camera name="overview" pos="1.1 -1.6 1.45" xyaxes=".85 .52 0 -.30 .49 .82"/>
    <geom name="floor" type="plane" size="3 3 .1" rgba=".88 .9 .91 1"/>
    <geom name="table" type="box" pos=".43 .04 .21" size=".55 .49 .195"
          rgba=".58 .47 .39 1"/>
    <body name="robot" pos="{BASE[0]} {BASE[1]} {BASE[2]}">
      <geom type="cylinder" size=".09 .075" pos="0 0 -.06" mass="3" rgba=".28 .32 .4 1"/>
      <joint name="yaw" type="hinge" axis="0 0 1" range="-2.9 2.9" limited="true"/>
      <body name="upper">
        <joint name="shoulder" type="hinge" axis="0 -1 0" range="-2.7 2.7" limited="true"/>
        <geom type="capsule" fromto="0 0 0 {L1} 0 0" size=".035" mass="1.1"
              rgba=".22 .4 .70 1"/>
        <body pos="{L1} 0 0" name="forearm">
          <joint name="elbow" type="hinge" axis="0 -1 0" range="-2.6 2.6" limited="true"/>
          <geom type="capsule" fromto="0 0 0 {L2} 0 0" size=".028" mass=".7"
                rgba=".23 .55 .76 1"/>
          <body name="gripper" pos="{L2} 0 0">
            <geom type="sphere" size=".032" mass=".08" rgba=".15 .75 .47 1"/>
            <site name="tip" size=".012"/>
          </body>
        </body>
      </body>
    </body>
    <body name="human_hand" mocap="true" pos=".8 .3 .6">
      <geom name="human_palm" type="sphere" size=".047" rgba=".95 .62 .42 1"
            contype="0" conaffinity="0"/>
    </body>
    {''.join(cups)}
    {''.join(coasters)}
  </worldbody>
  <actuator>
    <position joint="yaw" kp="110"/>
    <position joint="shoulder" kp="125"/>
    <position joint="elbow" kp="95"/>
  </actuator>
  <equality>{''.join(welds)}</equality>
</mujoco>"""

class I1Scene:
    def __init__(self,seed:int):
        self.clip=HumanClip.sample(seed)
        self.model=mujoco.MjModel.from_xml_string(scene_xml())
        self.data=mujoco.MjData(self.model)
        self.data.qpos[:3] = [np.pi/2,1.1,-1.85]
        self.data.ctrl[:3] = self.data.qpos[:3]
        self.hand_mocap=self.model.body("human_hand").mocapid[0]
        self.coaster_mocap=[self.model.body(f"coaster{i}").mocapid[0] for i in range(3)]
        self.weld_ids=[self.model.equality(f"grasp{i}").id for i in range(3)]
        self.gripper_id=self.model.body("gripper").id
        self.cup_ids=[self.model.body(f"cup{i}").id for i in range(3)]
        self.held=None
        self.used=set()
        self.advance_human(0)
        mujoco.mj_forward(self.model,self.data)
        self.initial_hash=hashlib.sha256(
            self.data.qpos.tobytes()+self.clip.targets.tobytes()
            +self.clip.order.tobytes()).hexdigest()[:16]

    def advance_human(self,t:float):
        _,_,hand=self.clip.pose(t)
        self.data.mocap_pos[self.hand_mocap]=hand
        for i,pos in enumerate(self.clip.coaster_poses(t)):
            self.data.mocap_pos[self.coaster_mocap[i]]=pos

    def tip(self):
        return self.data.site("tip").xpos.copy()

    def cup_xyz(self,i):
        return self.data.xpos[self.cup_ids[i]].copy()

    def attach(self,i):
        if self.held is not None or i in self.used:
            raise ValueError("Cup already used or robot already holding a cup")
        # Set the weld reference pose to the current measured relative transform.
        b1=self.gripper_id
        b2=self.cup_ids[i]
        q1=self.data.xquat[b1]
        q2=self.data.xquat[b2]
        qi=np.zeros(4)
        mujoco.mju_negQuat(qi,q1)
        rel=np.zeros(3)
        mujoco.mju_rotVecQuat(rel,self.data.xpos[b2]-self.data.xpos[b1],qi)
        qr=np.zeros(4)
        mujoco.mju_mulQuat(qr,qi,q2)
        self.model.eq_data[self.weld_ids[i],:3]=rel
        self.model.eq_data[self.weld_ids[i],3:7]=qr
        self.data.eq_active[self.weld_ids[i]]=1
        self.held=i
        mujoco.mj_forward(self.model,self.data)

    def release(self):
        if self.held is None: raise ValueError("No cup held")
        i=self.held
        self.data.eq_active[self.weld_ids[i]]=0
        self.held=None
        self.used.add(i)
        mujoco.mj_forward(self.model,self.data)
        return i

    def step(self,q_cmd):
        self.advance_human(self.data.time)
        self.data.ctrl[:3]=np.asarray(q_cmd)
        mujoco.mj_step(self.model,self.data)
