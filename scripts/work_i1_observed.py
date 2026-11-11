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


def run(seed: int, out: Path, speedup: float = 1.0, expert: str = 'observed'):
    # Film cadence is 20 physical decision samples/sec. A playback speedup
    # affects ONLY the preview MP4 timestamps, never simulator timing, safety
    # distance, controller speed limits, or demonstration labels.
    out.mkdir(parents=True,exist_ok=True)
    if expert not in ("observed", "early"):
        raise ValueError("expert must be observed or early")
    # Public Cornell ManiCast checkpoint, not the absent Work fine-tune.
    if expert == "early":
        import pif.config
        original_load = pif.config.load_config
        def released_config(*args, **kwargs):
            cfg = original_load(*args, **kwargs)
            if "v6_expert" in cfg:
                cfg["v6_expert"]["checkpoint_path"] = None
            return cfg
        pif.config.load_config = released_config
    # Reach cue unlocks preparation and grasp.
    from intent_policy.experts.i1_expert import I1Expert
    from intent_policy.sim.restricted_action import RestrictedAction
    original_act = I1Expert.act
    original_destination = I1Expert._destination

    def reached_for_this_coaster(self):
        idx = self.sc.sequence_index
        return any(
            e["event_type"] == "human_cue_onset"
            and int((e.get("payload") or {}).get("round", 0)) == idx
            for e in self.sc.get_events()
        )

    def act_after_reach_cue(self):
        if not reached_for_this_coaster(self):
            return int(RestrictedAction.HOLD)
        return original_act(self)

    def destination_after_place(self):
        if expert == "observed" and "coaster_placed_" + self.coaster not in self.sc.protocol_done:
            return None
        return original_destination(self)

    I1Expert.act = act_after_reach_cue
    I1Expert._destination = destination_after_place
    scene=make_scenario("i1_coaster_sequence", {"expert":{"rule":expert},
        # Generated HY-Motion clips were not checked into the old repository.
        # Use its own deterministic minimum-jerk scripted human instead.
        "human_behavior":{"motion_bank":None}})
    controller=RestrictedActionConfig.load("configs/controller/continuous_motion.yaml")
    obs_cfg=ObservationConfig(cameras=("high",),width=640,height=480)
    observed=[]
    carried_frames={}
    weld_seen=set()
    video=out/f"I1_{expert}_work.mp4"
    # One large fixed high camera; images are RGB, not keypoint overlays.
    with imageio.get_writer(video,fps=round(20*speedup),codec="libx264",
                            macro_block_size=1,quality=8) as writer:
        def record_frame(obs,decision,target,s):
            for item in s.rounds:
                key = item["coaster"]
                if s.human.holding == key:
                    carried_frames[key] = carried_frames.get(key, 0) + 1
                    if s.env.data.eq_active[s.weld_id[key]]:
                        weld_seen.add(key)
            frame=np.asarray(obs["observation.images.high"],dtype=np.uint8)
            writer.append_data(frame)
            observed.append(float(s.time))
        try:
            result=run_episode(scene,ExpertAgent(),seed,controller,
                               obs_cfg=obs_cfg,on_frame=record_frame,
                               keep_trace=False)
        finally:
            scene.close()
    if result["success"]:
        required={item["coaster"] for item in scene.rounds}
        assert required <= weld_seen, "No verified human weld: " + repr(required - weld_seen)
        assert all(carried_frames.get(k,0)>1 for k in required), "Human did not carry all coasters"
    summary={"scenario":"i1_coaster_sequence",
             "expert":expert,"seed":seed,
             "forecast_source":"Cornell released ManiCast" if expert=="early" else "none",
             "work_source":"hungne121/intent_policy@428e8fcb9d7834eb6e6be98ab72b1ccf9c71ab3a",
             "robot":"UR3e 6-DOF + SusGrip physical fingers",
             "scene":"Work original MuJoCo MJCF and original I1 human motion",
             "controller":"configs/controller/continuous_motion.yaml (20Hz)",
             "success":bool(result["success"]),
             "failure":result.get("failure"),
             "physics_duration_s":result.get("duration_s"),
             "preview_playback_factor":speedup,
             "preview_frame_count":len(observed),
             "robot_start_rule":"reach cue -> prepare; forecast -> early approach" if expert=="early" else "reach cue -> prepare; coaster placed -> deliver",
             "human_carried_frames":carried_frames,
             "human_active_welds":sorted(weld_seen),
             "recorded_cadence_hz":20,
             "video":f"I1_{expert}_work.mp4",
             "note":"Playback speed does not change simulated physics or training actions"}
    (out/f"I1_{expert}_work.json").write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str),flush=True)
    if not result["success"]:
        raise SystemExit("Observed I1 failed; see JSON and video")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=300100)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--speedup",type=float,default=1.0)
    p.add_argument("--expert",choices=["observed","early"],default="observed")
    a=p.parse_args()
    if not (1<=a.speedup<=1.5):raise ValueError("speedup must be 1..1.5")
    run(a.seed,a.out,a.speedup,a.expert)

if __name__=="__main__":main()
