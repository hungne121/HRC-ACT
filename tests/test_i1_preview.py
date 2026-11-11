"""I1 V2 scene, paired seed, cue and safety gate checks."""
import numpy as np
import pytest
from hrc_act.i1_scene import I1Scene,HumanClip,TABLE_Z
from hrc_act.i1_planning import JointPlanner,WristForecast,infer_target,safe_to_place

def test_human_clip_repeatability():
    a=HumanClip.sample(300100)
    b=HumanClip.sample(300100)
    assert a.clip_id==b.clip_id
    np.testing.assert_array_equal(a.order,b.order)
    np.testing.assert_allclose(a.targets,b.targets)
    assert len(set(a.order))==3

def test_clip_all_three_placed():
    clip=HumanClip.sample(300100)
    for t in (4.5,9.5,14.5):
        n=int(t//5)+1
        for i in clip.order[:n]:
            np.testing.assert_allclose(
                clip.coaster_poses(t)[int(i)],clip.targets[int(i)])
    assert clip.coaster_poses(16).shape==(3,3)

def test_common_motion_planner_has_zero_endpoint_speed():
    p=JointPlanner()
    p.target(0.,[0,0,0,0],[.4,.15,.7])
    assert p.duration>=.65
    q0,done=p.sample(0.,[0,0,0,0])
    np.testing.assert_allclose(q0,[0,0,0,0])
    assert not done
    q1,done=p.sample(p.duration,[0,0,0,0])
    np.testing.assert_allclose(q1,p.end)
    assert done

def test_ik_forward_kinematics():
    p=JointPlanner()
    target=np.array([.48,.19,.68])
    yaw,shoulder,elbow,wrist=p.ik(target)
    assert abs(shoulder+elbow+wrist)<1e-10
    from hrc_act.i1_scene import BASE,L1,L2
    radial=L1*np.cos(shoulder)+L2*np.cos(shoulder+elbow)
    z=L1*np.sin(shoulder)+L2*np.sin(shoulder+elbow)
    measured=BASE+np.array([radial*np.cos(yaw),radial*np.sin(yaw),z])
    np.testing.assert_allclose(measured,target,atol=.012)

def test_scene_loads_and_reproducible_initial():
    a=I1Scene(300100)
    b=I1Scene(300100)
    assert a.initial_hash==b.initial_hash
    assert len(a.cup_ids)==3
    assert len(a.coaster_mocap)==3
    for _ in range(10):
        a.step(a.data.qpos[:4])
    assert a.data.time>.0

def test_observed_cannot_see_unplaced_coaster_target():
    scene=I1Scene(300100)
    scene.advance_human(0.0)
    idx,xy,source=infer_target(scene,"observed",set(),WristForecast())
    assert idx is None and xy is None and source=="waiting"

def test_preview_proxy_uses_past_motion_only():
    f=WristForecast()
    for i in range(8):
        f.push(i*.1,[.8-i*.04,.30-i*.02,.6])
    projected=f.projected()
    assert projected is not None
    assert projected[0]<.8-7*.04

def test_release_gate_rejects_human_near_and_unplaced():
    s=I1Scene(300100)
    i=int(s.clip.order[0])
    target=s.clip.targets[i]
    s.advance_human(2.7)
    assert not safe_to_place(s,i,target)
    s.advance_human(4.5)
    assert safe_to_place(s,i,target)

def test_attach_cannot_reuse_cup():
    s=I1Scene(300100)
    s.attach(0)
    with pytest.raises(ValueError):
        s.attach(0)
    assert s.release()==0
    with pytest.raises(ValueError):
        s.attach(0)
