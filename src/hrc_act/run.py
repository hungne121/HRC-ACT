"""Reproducible I1 paired-mode command line runner."""
import argparse
import json
from pathlib import Path
import numpy as np
from .sim import I1Sim
from .planner import MotionPlanner
from .experts import Prediction,choose_target

def record(expert, out, forecast=None, demo_forecast=False, fps=20, seconds=5.0):
    import mujoco
    import imageio.v2 as imageio
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    sim=I1Sim()
    planner=MotionPlanner()
    dt=sim.model.opt.timestep
    steps_per_frame=round(1/(fps*dt))
    if not np.isclose(steps_per_frame*dt,1/fps):
        raise ValueError("FPS must divide physics rate exactly")
    if demo_forecast:
        if forecast is not None:
            raise ValueError("Cannot combine synthetic forecast with real forecast")
        times=np.arange(1,11,dtype=float)*0.1
        # Explicitly synthetic, not ManiCast inference.
        positions=np.tile(sim.human_xyz(2.5),(len(times),1))
        forecast=Prediction(times,positions)
    if expert=="intent" and forecast is None:
        raise ValueError("Intent mode requires --forecast or --demo-forecast")

    dataset=LeRobotDataset.create(
        repo_id=f"local/i1-{expert}",root=out/"dataset",fps=fps,
        robot_type="simulated_2dof_arm",use_videos=True,
        features={
            "observation.state":{"dtype":"float32","shape":(2,),"names":["shoulder","elbow"]},
            "action":{"dtype":"float32","shape":(2,),"names":["shoulder","elbow"]},
            "observation.images.overview":{"dtype":"video","shape":(480,640,3),
                "names":["height","width","channel"]},
        },
    )
    segment=None
    onset=None
    sources=[]
    renderer=mujoco.Renderer(sim.model,height=480,width=640)
    try:
        with imageio.get_writer(out/"overview.mp4",fps=fps,codec="libx264",macro_block_size=1) as video:
            for k in range(round(seconds*fps)):
                t=sim.data.time
                hand=sim.human_xyz(t)
                ready=t>=2.0
                target,source=choose_target(expert,hand,ready,forecast)
                sources.append(source)
                if segment is None and target is not None:
                    # The planar arm lives in x-z with base z=.39.
                    goal=planner.ik([target[0],target[2]-.39],sim.data.qpos[:2])
                    segment=planner.plan(sim.data.qpos[:2],goal)
                    onset=t
                if segment is None:
                    cmd=sim.data.qpos[:2].copy()
                else:
                    cmd,_=segment.sample(t-onset)
                for _ in range(steps_per_frame):
                    sim.tick(cmd)
                rgb=sim.frame(renderer)
                video.append_data(rgb)
                dataset.add_frame({
                    "task":"I1 human hand presentation",
                    "observation.state":sim.data.qpos[:2].astype(np.float32).copy(),
                    "action":np.asarray(cmd,dtype=np.float32),
                    "observation.images.overview":rgb,
                })
        dataset.save_episode()
    finally:
        renderer.close()
        dataset.finalize()
    metrics={"expert":expert,"onset_s":onset,"episode_s":seconds,
             "source":next((s for s in sources if s!="waiting"),"waiting"),
             "forecast_type":"synthetic" if demo_forecast else ("external" if forecast is not None else "none"),
             "task_success_validated":False}
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2))
    return metrics

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--expert",choices=["observed","intent"],required=True)
    p.add_argument("--forecast",type=Path)
    p.add_argument("--demo-forecast",action="store_true")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--fps",type=int,default=20)
    p.add_argument("--seconds",type=float,default=5.0)
    args=p.parse_args()
    forecast=Prediction.load(args.forecast) if args.forecast else None
    print(json.dumps(record(args.expert,args.out,forecast,args.demo_forecast,args.fps,args.seconds),indent=2))

if __name__=="__main__":
    main()
