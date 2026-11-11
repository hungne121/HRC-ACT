"""Render paired I1 previews in MuJoCo, with optional LeRobot episode recording.

Intent mode's causal wrist extrapolation is explicitly a PREVIEW PROXY, not
ManiCast. Never use these traces as ManiCast/ACT research results.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio

from .i1_scene import I1Scene,SOURCE,TABLE_Z,CUP_H
from .i1_planning import JointPlanner,WristForecast,infer_target,safe_to_place

def make_dataset(path:Path,fps:int):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset.create(
        repo_id="local/i1-preview",root=path,fps=fps,
        robot_type="simulated_3dof",use_videos=True,
        features={
            "observation.state":{"dtype":"float32","shape":(6,),
                                 "names":["yaw","shoulder","elbow","yaw_v","shoulder_v","elbow_v"]},
            "action":{"dtype":"float32","shape":(3,),"names":["yaw","shoulder","elbow"]},
            "observation.images.overview":{"dtype":"video","shape":(640,960,3),
                                            "names":["height","width","channel"]},
        },
    )

def rollout(expert:str,seed:int,out:Path,fps:int=10,
            max_seconds:float=65.,dataset_enabled:bool=True):
    if expert not in ("observed","intent"): raise ValueError(expert)
    scene=I1Scene(seed)
    planner=JointPlanner(speed=1.25)
    forecast=WristForecast()
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    dt=float(scene.model.opt.timestep)
    substeps=int(round(1/(fps*dt)))
    if not np.isclose(substeps*dt,1/fps): raise ValueError("FPS incompatible with dt")
    dataset=make_dataset(out/"dataset",fps) if dataset_enabled else None
    renderer=mujoco.Renderer(scene.model,height=640,width=960)

    used_coasters=set()
    events=[{"event":"task_active","t":0.}]
    frames=[]
    trace=[]
    safety_overrides=0
    max_penetration=0.0
    worst_contact=None
    state="above_source"
    turn=0
    current_cup=0
    candidate_coaster=None
    target_xy=None
    state_enter=0.
    placed=[]

    def event(name,t,**kw):
        events.append({"event":name,"t":round(float(t),3),**kw})

    def move(xyz):
        planner.target(scene.data.time,scene.data.qpos[:3],xyz)

    def transition(to,t):
        nonlocal state,state_enter
        state=to
        state_enter=t

    move(SOURCE[0]+np.array([0,0,.23]))
    event("prep_onset",0.,cup=0)
    human_was_moving=False

    try:
        with imageio.get_writer(out/f"I1_{expert}.mp4",fps=fps,
                                codec="libx264",quality=7,macro_block_size=1) as video:
            n_frames=int(max_seconds*fps)
            for frame in range(n_frames):
                for _ in range(substeps):
                    t=float(scene.data.time)
                    scene.advance_human(t)
                    hand=scene.data.mocap_pos[scene.hand_mocap].copy()
                    if int(round(t/dt))%20==0:
                        forecast.push(t,hand)
                    moving=np.linalg.norm(hand-scene.clip.pose(max(0,t-.1))[2])>.012
                    if moving and not human_was_moving:
                        event("human_motion_onset",t)
                    human_was_moving=moving
                    q,done=planner.sample(t,scene.data.qpos[:3])
                    # Use measured position for posture convergence, not just timeline.
                    settled=bool(done and np.max(np.abs(scene.data.qpos[:3]-planner.end))<.09)
                    if state=="above_source" and settled:
                        move(SOURCE[current_cup]+np.array([0,0,CUP_H/2+.014]))
                        transition("descend_source",t)
                    elif state=="descend_source" and settled:
                        distance=np.linalg.norm(scene.tip()-(scene.cup_xyz(current_cup)+[0,0,CUP_H/2]))
                        if distance<.072 and np.linalg.norm(scene.tip()[:2]-scene.cup_xyz(current_cup)[:2])<.025:
                            scene.attach(current_cup)
                            event("grasp",t,cup=current_cup)
                            move(SOURCE[current_cup]+np.array([0,0,.25]))
                            transition("lift",t)
                        else:
                            event("failed_grasp",t,distance=float(distance))
                            transition("failed",t)
                    elif state=="lift" and settled:
                        transition("await_target",t)
                    elif state=="await_target":
                        idx,xy,source=infer_target(scene,expert,used_coasters,forecast)
                        if idx is not None and idx not in used_coasters:
                            candidate_coaster=int(idx)
                            target_xy=np.asarray(xy).copy()
                            event("robot_commit",t,coaster=candidate_coaster,source=source)
                            event("target_observed" if source=="observation" else "forecast_update",t)
                            move([target_xy[0],target_xy[1],TABLE_Z+.32])
                            transition("route_to_target",t)
                    elif state=="route_to_target" and settled:
                        transition("hover",t)
                    elif state=="hover":
                        if candidate_coaster is not None:
                            target=scene.data.mocap_pos[scene.coaster_mocap[candidate_coaster]]
                            hand_clear=np.linalg.norm(hand[:2]-target[:2])>.16
                            stationary=abs(target[2]-(TABLE_Z+.007))<.012
                            if stationary and hand_clear:
                                # Actual observed pose always overrides proxy forecast.
                                target_xy=target[:2].copy()
                                event("human_ready",t,coaster=candidate_coaster)
                                held_offset=scene.cup_xyz(current_cup)[:2]-scene.tip()[:2]
                                grip_xy=target_xy-held_offset
                                move([grip_xy[0],grip_xy[1],TABLE_Z+.16])
                                transition("approach",t)
                    elif state=="approach" and settled:
                        if safe_to_place(scene,candidate_coaster,target_xy):
                            event("robot_ready",t,coaster=candidate_coaster)
                            held_offset=scene.cup_xyz(current_cup)[:2]-scene.tip()[:2]
                            grip_xy=target_xy-held_offset
                            move([grip_xy[0],grip_xy[1],TABLE_Z+.014+CUP_H+.014])
                            transition("lower",t)
                        else:
                            safety_overrides+=1
                    elif state=="lower" and settled:
                        cup=scene.cup_xyz(current_cup)
                        goal=scene.data.mocap_pos[scene.coaster_mocap[candidate_coaster]]
                        placement_height=TABLE_Z+.014+CUP_H/2
                        if (safe_to_place(scene,candidate_coaster,cup)
                            and abs(cup[2]-placement_height)<.022):
                            released=scene.release()
                            used_coasters.add(candidate_coaster)
                            placed.append((released,candidate_coaster))
                            event("release",t,cup=released,coaster=candidate_coaster)
                            transition("settle_cup",t)
                        else:
                            safety_overrides+=1
                            # Correct the measured held-cup pose instead of
                            # trusting the planned tool-tip target as ground truth.
                            error=goal[:2]-cup[:2]
                            if np.linalg.norm(error)<.10:
                                tip=scene.tip()
                                move([tip[0]+error[0],tip[1]+error[1],
                                      TABLE_Z+.014+CUP_H+.014+(placement_height-cup[2])])
                            else:
                                event("unsafe_alignment",t,error=float(np.linalg.norm(error)))
                                transition("failed",t)
                    elif state=="settle_cup" and t-state_enter>.45:
                        i,j=placed[-1]
                        cup=scene.cup_xyz(i)
                        coaster=scene.data.mocap_pos[scene.coaster_mocap[j]]
                        error=np.linalg.norm(cup[:2]-coaster[:2])
                        vertical=abs(cup[2]-(TABLE_Z+.007+.007+CUP_H/2))
                        event("place_complete" if error<.043 and vertical<.06 else "place_unstable",
                              t,cup=i,coaster=j,xy_error=float(error),vertical_error=float(vertical))
                        if error>=.043 or vertical>=.06:
                            transition("failed",t)
                        elif len(placed)==3:
                            event("episode_end",t)
                            transition("complete",t)
                        else:
                            current_cup+=1
                            next_pos=SOURCE[current_cup]+np.array([0,0,.23])
                            move(next_pos)
                            event("prep_onset",t,cup=current_cup)
                            candidate_coaster=None
                            target_xy=None
                            transition("above_source",t)
                    # collision depth is recorded; only target release gate uses a strict hold.
                    for c in scene.data.contact:
                        depth=max(0.,-float(c.dist))
                        if depth>max_penetration:
                            max_penetration=depth
                            worst_contact={"t":round(t,3),"geom1":scene.model.geom(c.geom1).name,
                                           "geom2":scene.model.geom(c.geom2).name,"depth_m":depth}
                    scene.step(q)

                renderer.update_scene(scene.data,camera="overview")
                rgb=renderer.render().copy()
                video.append_data(rgb)
                trace.append({"t":round(float(scene.data.time),3),
                              "state":state,"q":scene.data.qpos[:3].round(5).tolist(),
                              "action":np.asarray(q).round(5).tolist()})
                if dataset is not None:
                    dataset.add_frame({
                        "task":"I1 place three cups on three human coasters",
                        "observation.state":np.concatenate([scene.data.qpos[:3],scene.data.qvel[:3]]).astype(np.float32),
                        "action":np.asarray(q,dtype=np.float32),
                        "observation.images.overview":rgb})
                if state in ("complete","failed"):
                    break
        if dataset is not None:
            dataset.save_episode()
    finally:
        renderer.close()
        if dataset is not None:
            dataset.finalize()

    cup_positions=np.array([scene.cup_xyz(i) for i in range(3)])
    final_coaster=np.array([scene.data.mocap_pos[k] for k in scene.coaster_mocap])
    matches=[np.linalg.norm(cup_positions[i,:2]-final_coaster[j,:2])<.045
             for i,j in placed]
    success=bool(state=="complete" and len(placed)==3 and all(matches))
    metadata={
        "task":"I1","case":"nominal","expert":expert,"forecast_source":
            "causal_linear_proxy_NOT_ManiCast" if expert=="intent" else "none",
        "seed":seed,"human_clip_id":scene.clip.clip_id,
        "scene_init_hash":scene.initial_hash,
        "coaster_order":scene.clip.order.tolist(), # evaluator metadata, never policy input
        "coaster_targets":scene.clip.targets.tolist(), # evaluator-only
        "events":events,"state":state,"success":success,"placed":placed,
        "max_contact_penetration_m":max_penetration,
        "worst_contact":worst_contact,
        "safety_override_count":safety_overrides,
        "manicast_integrated":False,
        "robot_action_trace":trace,
    }
    (out/f"I1_{expert}.json").write_text(json.dumps(metadata,indent=2))
    return {"expert":expert,"seed":seed,"success":success,
            "state":state,"video":str(out/f"I1_{expert}.mp4"),
            "clip_id":scene.clip.clip_id,"placed":len(placed)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=300100)
    ap.add_argument("--out",type=Path,default=Path("outputs/i1_preview"))
    ap.add_argument("--expert",choices=["observed","intent","paired"],default="paired")
    ap.add_argument("--fps",type=int,default=10)
    ap.add_argument("--max-seconds",type=float,default=65.)
    ap.add_argument("--skip-dataset",action="store_true")
    args=ap.parse_args()
    runs=["observed","intent"] if args.expert=="paired" else [args.expert]
    results=[]
    for mode in runs:
        results.append(rollout(mode,args.seed,args.out,args.fps,
                               args.max_seconds,not args.skip_dataset))
    print(json.dumps(results,indent=2))
if __name__=="__main__":
    main()
