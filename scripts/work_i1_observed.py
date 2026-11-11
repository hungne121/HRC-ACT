"""I1 Observed preview using the approved Work UR3e + SusGrip MuJoCo scene.

This is a small adapter, not another robot or scene. No Intent or ACT training.
Run from the *source checkout* of hungne121/intent_policy at the pinned commit:
    PYTHONPATH=. python ../HRC-ACT/scripts/work_i1_observed.py --out ../HRC-ACT/outputs/work_i1

The source checkout is read-only. Its scene, human geometry, arm/gripper, I1
scripted expert, joint controller and physical contact logic remain unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import imageio.v2 as imageio

from intent_policy.benchmark.runner import ExpertAgent, ObservationConfig, run_episode
from intent_policy.scenarios.scenario_registry import make_scenario
from intent_policy.sim.restricted_action import RestrictedActionConfig


def run(seed: int, out: Path, speedup: float = 1.0):
    # Film cadence is 20 physical decision samples/sec. A playback speedup
    # affects ONLY the preview MP4 timestamps, never simulator timing, safety
    # distance, controller speed limits, or demonstration labels.
    out.mkdir(parents=True,exist_ok=True)
    # Observed preview rule: keep the UR3e arm completely stationary until
    # the human has physically released the current coaster onto the table.
    # Applies at EACH of the three rounds. Do not change human motion timing.
    from intent_policy.experts.i1_expert import I1Expert
    from intent_policy.sim.restricted_action import RestrictedAction
    original_act = I1Expert.act

    def act_only_after_coaster_placed(self):
        current = "coaster_placed_" + self.coaster
        if current not in self.sc.protocol_done:
            return int(RestrictedAction.HOLD)
        return original_act(self)

    I1Expert.act = act_only_after_coaster_placed
    scene=make_scenario("i1_coaster_sequence", {"expert":{"rule":"observed"},
        # Generated HY-Motion clips were not checked into the old repository.
        # Use its own deterministic minimum-jerk scripted human instead.
        "human_behavior":{"motion_bank":None}})
    controller=RestrictedActionConfig.load("configs/controller/continuous_motion.yaml")
    obs_cfg=ObservationConfig(cameras=("high",),width=640,height=480)
    observed=[]
    video=out/"I1_observed_work.mp4"
    # One large fixed high camera; images are RGB, not keypoint overlays.
    with imageio.get_writer(video,fps=round(20*speedup),codec="libx264",
                            macro_block_size=1,quality=8) as writer:
        def record_frame(obs,decision,target,s):
            frame=np.asarray(obs["observation.images.high"],dtype=np.uint8)
            writer.append_data(frame)
            observed.append(float(s.time))
        try:
            result=run_episode(scene,ExpertAgent(),seed,controller,
                               obs_cfg=obs_cfg,on_frame=record_frame,
                               keep_trace=False)
        finally:
            scene.close()
    summary={"scenario":"i1_coaster_sequence",
             "expert":"observed","seed":seed,
             "work_source":"hungne121/intent_policy@428e8fcb9d7834eb6e6be98ab72b1ccf9c71ab3a",
             "robot":"UR3e 6-DOF + SusGrip physical fingers",
             "scene":"Work original MuJoCo MJCF and original I1 human motion",
             "controller":"configs/controller/continuous_motion.yaml (20Hz)",
             "success":bool(result["success"]),
             "failure":result.get("failure"),
             "physics_duration_s":result.get("duration_s"),
             "preview_playback_factor":speedup,
             "preview_frame_count":len(observed),
             "robot_start_rule":"hold all arm motion until current coaster is placed",
             "recorded_cadence_hz":20,
             "video":"I1_observed_work.mp4",
             "note":"Playback speed does not change simulated physics or training actions"}
    (out/"I1_observed_work.json").write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str),flush=True)
    if not result["success"]:
        raise SystemExit("Observed I1 failed; see JSON and video")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=300100)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--speedup",type=float,default=1.0)
    a=p.parse_args()
    if not (1<=a.speedup<=1.5):raise ValueError("speedup must be 1..1.5")
    run(a.seed,a.out,a.speedup)

if __name__=="__main__":main()
