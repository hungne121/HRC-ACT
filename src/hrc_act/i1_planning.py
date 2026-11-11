"""Joint-space motion primitives and observation-limited target selectors for I1.

No future human clip, target oracle, coaster IDs as labels or case metadata
is supplied to expert action selection. Only observed mocap object locations,
robot proprioception and a causal wrist-position history are used here.

The causal linear prediction is a VISUAL PREVIEW PROXY, NOT ManiCast.
"""
from collections import deque
import numpy as np
from .i1_scene import BASE,L1,L2,TABLE_Z,COASTER_R

class JointPlanner:
    def __init__(self,speed=1.2):
        self.speed=float(speed)
        self.start=None
        self.end=None
        self.begin=0.
        self.duration=1.

    @staticmethod
    def ik(xyz,seed=None):
        p=np.asarray(xyz,dtype=float)-BASE
        yaw=np.arctan2(p[1],p[0])
        r=np.hypot(p[0],p[1])
        z=p[2]
        d=np.hypot(r,z)
        d=np.clip(d,abs(L1-L2)+1e-3,L1+L2-0.01)
        c=np.clip((d*d-L1*L1-L2*L2)/(2*L1*L2),-1,1)
        # elbow-up configuration: shoulder stays above workspace
        elbow=-np.arccos(c)
        shoulder=np.arctan2(z,r)-np.arctan2(L2*np.sin(elbow),L1+L2*np.cos(elbow))
        wrist=-(shoulder+elbow)  # world-level grasp frame: cup stays vertical
        q=np.array([yaw,shoulder,elbow,wrist])
        if not np.isfinite(q).all(): raise ValueError("Invalid IK")
        return q

    def target(self,now,q,xyz):
        goal=self.ik(xyz)
        self.start=np.asarray(q,dtype=float).copy()
        self.end=goal
        self.begin=float(now)
        self.duration=max(0.65,1.875*np.max(np.abs(goal-self.start))/self.speed)

    def sample(self,now,q_fallback):
        if self.start is None: return np.asarray(q_fallback).copy(),True
        u=np.clip((now-self.begin)/self.duration,0,1)
        s=u*u*u*(10+u*(-15+6*u))
        return self.start+s*(self.end-self.start),u>=1

class WristForecast:
    def __init__(self,horizon=0.75):
        self.history=deque(maxlen=16)
        self.horizon=horizon

    def push(self,t,xyz):
        self.history.append((float(t),np.asarray(xyz).copy()))

    def projected(self):
        if len(self.history)<4: return None
        now,p=self.history[-1]
        # strictly past data; no access to clip targets or future poses.
        older=[(t,v) for t,v in self.history if now-t>=0.30]
        if not older: return None
        t0,x0=older[-1]
        velocity=(p-x0)/(now-t0)
        if np.linalg.norm(velocity[:2])<.07: return None
        return p+velocity*self.horizon

def infer_target(scene,expert,used_coasters,forecast):
    if expert not in ("observed","intent"):
        raise ValueError(expert)
    pos=np.array([scene.data.mocap_pos[j] for j in scene.coaster_mocap])
    hand=scene.data.mocap_pos[scene.hand_mocap].copy()
    candidates=[]
    for i,p in enumerate(pos):
        if i in used_coasters: continue
        # A coaster is a candidate only once it is visible in working area;
        # the unused coasters stay in the human-side source.
        if .11 <= p[0] <= .69 and .11 <= p[1] <= .31:
            candidates.append((i,p))
    if expert=="observed":
        for i,p in candidates:
            if p[2] <= TABLE_Z+.028:
                return i,p[:2].copy(),"observation"
        return None,None,"waiting"
    pred=forecast.projected()
    if pred is None:
        return None,None,"waiting"
    # Only use extrapolated wrist, and only when its target lies near a
    # currently observed coaster. No ground-truth future target access.
    for i,p in candidates:
        if np.linalg.norm(pred[:2]-p[:2]) <= .17:
            return i,pred[:2].copy(),"proxy_forecast"
    return None,None,"waiting"

def safe_to_place(scene,coaster_idx,placement_xyz):
    if coaster_idx is None: return False
    coaster=scene.data.mocap_pos[scene.coaster_mocap[coaster_idx]]
    hand=scene.data.mocap_pos[scene.hand_mocap]
    on_table=abs(coaster[2]-(TABLE_Z+.007)) < .012
    hand_clear=np.linalg.norm(hand[:2]-coaster[:2])>.16
    aligned=np.linalg.norm(np.asarray(placement_xyz)[:2]-coaster[:2])<COASTER_R*.6
    return bool(on_table and hand_clear and aligned)
