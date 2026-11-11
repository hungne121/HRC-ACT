import numpy as np
import pytest
from hrc_act.planner import MotionPlanner
from hrc_act.experts import Prediction, choose_target

def test_minimum_jerk_stops_at_both_ends():
    p=MotionPlanner()
    s=p.plan([0,0],[0.4,-0.5])
    q0,v0=s.sample(0)
    q1,v1=s.sample(s.duration)
    np.testing.assert_allclose(q0,[0,0])
    np.testing.assert_allclose(q1,[0.4,-0.5])
    np.testing.assert_allclose(v0,[0,0],atol=1e-12)
    np.testing.assert_allclose(v1,[0,0],atol=1e-12)

def test_joint_speed_bound():
    s=MotionPlanner(max_joint_speed=0.8).plan([0,0],[1,-1])
    for t in np.linspace(0,s.duration,40):
        _,v=s.sample(t)
        assert np.max(np.abs(v))<=0.8+1e-9

def test_ik_reaches_target():
    p=MotionPlanner()
    goal=np.array([0.45,0.12])
    q=p.ik(goal,[0.5,-0.7])
    np.testing.assert_allclose(p.fk(q),goal,atol=0.015)

def test_observed_waits_and_intent_falls_back():
    wrist=[0.4,0,0.5]
    assert choose_target("observed",wrist,False,None)[0] is None
    assert choose_target("intent",wrist,False,None)[0] is None
    assert choose_target("intent",wrist,True,None)[1]=="observed"

def test_intent_prefers_stable_prediction():
    t=np.arange(1,11)*0.1
    p=np.tile([0.45,0,0.52],(10,1))
    target,source=choose_target("intent",[0.3,0,0.5],False,Prediction(t,p))
    assert source=="prediction"
    np.testing.assert_allclose(target,p[-1])

def test_bad_prediction_rejected():
    with pytest.raises(ValueError):
        Prediction(np.array([0.2,0.1]),np.zeros((2,3)))

def test_mujoco_scene_loads_and_steps():
    from hrc_act.sim import I1Sim
    s=I1Sim()
    s.tick(s.data.qpos[:2].copy())
    assert s.data.time>0
