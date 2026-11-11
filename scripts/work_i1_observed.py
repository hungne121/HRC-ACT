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


def run(seed: int, out: Path, speedup: float = 1.0, expert: str = 'observed', tempo: str = 'balanced', robot_profile: str = 'baseline', layout: str = 'row'):
    # Film cadence is 20 physical decision samples/sec. A playback speedup
    # affects ONLY the preview MP4 timestamps, never simulator timing, safety
    # distance, controller speed limits, or demonstration labels.
    out.mkdir(parents=True,exist_ok=True)
    if robot_profile not in ("baseline", "optimized"):
        raise ValueError("robot profile must be baseline or optimized")
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
        # Simulation-only controlled timing ablation. On robot-only prep
        # moves, speed up to 0.50 m/s iff human hand is > 0.40m from TCP.
        # In the shared workspace keep exactly the original 0.39m/s cap.
        # This heuristic is NOT a safety certification or speed approval.
        if robot_profile == "optimized":
            clearance = float(np.linalg.norm(self.sc.human.pos - self.sc.tcp()))
            prep = self.stage in ("open", "above_grasp",
                "settle_above_grasp", "descend", "settle_before_grasp",
                "close", "settle_grasp", "lift")
            self.mapper.cfg.continuous["tcp_speed"] = (
                0.50 if prep and clearance > 0.40 else 0.39)
        if not reached_for_this_coaster(self):
            return int(RestrictedAction.HOLD)
        return original_act(self)

    def destination_after_place(self):
        if expert == "observed" and "coaster_placed_" + self.coaster not in self.sc.protocol_done:
            return None
        return original_destination(self)

    I1Expert.act = act_after_reach_cue
    I1Expert._destination = destination_after_place
    # The restricted action interface moves arm OR gripper, never both.
    # Convert the original expert's discrete plan into a *continuous 7D*
    # target so the gripper can change aperture simultaneously with TCP travel.
    # This supplies genuine synchronized demonstrations for ACT.
    from intent_policy.experts.scripted_expert import Step
    from intent_policy.benchmark.runner import Decision
    original_grasp_steps = I1Expert._grasp_steps
    original_home_steps = I1Expert._home_steps
    full_open_m = 0.085
    cruise_open_m = 0.055     # visibly pre-shape in free space, away from cup
    grasp_clear_m = 0.079     # cup outer diameter ~72mm + 7mm clearance

    def fluid_grasp_steps(self, key):
        steps = original_grasp_steps(self, key)
        # No blocking gripper-only preparation phase: shape fingers
        # while approaching. Keep a guard before descending onto the cup.
        steps = steps[1:]
        index=next(i for i,v in enumerate(steps) if v.name == "descend")
        steps.insert(index, Step("wait",
            until=lambda: float(self.sc.env.data.qpos[self.sc.env.gid]) >= 0.076,
            name="verify_grasp_clearance"))
        return steps

    def open_fully_only_at_rest(self):
        tail = original_home_steps(self)
        return tail[:-1] + [
            Step("wait", until=lambda: float(self.sc.env.data.qpos[self.sc.env.gid]) >= 0.083,
                 name="open_wide_at_rest"),
            tail[-1]]

    I1Expert._grasp_steps = fluid_grasp_steps
    I1Expert._home_steps = open_fully_only_at_rest

    class FluidExpertAgent(ExpertAgent):
        action_mode = "continuous"

        def reset(self, scenario, mapper):
            super().reset(scenario, mapper)
            self.mapper = mapper

        def act(self, obs):
            d = super().act(obs)
            stage = self.expert.stage
            command = self.mapper.map(d.action_id)
            target = np.asarray(command.joint_target,dtype=float).copy()
            if stage == "above_grasp" and not reached_for_this_coaster(self.expert):
                opening = full_open_m if self.expert.sc.sequence_index == 0 else grasp_clear_m
            elif stage == "above_grasp":
                obj = self.expert.obj
                above=np.r_[self.expert.sc.grasp_point(obj)[:2], self.expert.travel]
                remaining=float(np.linalg.norm(self.expert.sc.tcp() - above))
                opening = cruise_open_m if remaining > 0.13 else grasp_clear_m
            elif stage in ("settle_above_grasp","verify_grasp_clearance",
                           "descend","settle_before_grasp"):
                opening = grasp_clear_m
            elif stage in ("close","settle_grasp","lift","approach_spot",
                           "await_coaster","transport","lower","settle_before_release",
                           "back_off"):
                opening = 0.0
            elif stage in ("open_wide_at_rest","done"):
                opening = full_open_m
            else:
                # Following a release, preserve working aperture rather than
                # opening to rest. For later rounds pre-shape on the next reach.
                opening = grasp_clear_m
            # Never open while holding a cup, including stages overwritten
            # by the planner's safety / retreat logic.
            if any(self.expert.sc.grasp_state.get(k,False) for k in ("C1","C2","C3")):
                opening=0.0
            target[6] = opening
            self.mapper.gripper_target = opening
            return Decision(joint_target=target, extras={
                **d.extras,"gripper_stage":stage,"gripper_target_m":opening,
                "arm_action_id":d.action_id})

        def describe(self):
            return dict(type="fluid_scripted_expert",action_mode="continuous",
                        gripper_target_in_7d=True,training_demos=True)
    # Fixed, evenly spaced coaster targets (one straight row). The per-seed
    # permutation assigns each incoming coaster to left/centre/right once.
    # The expert NEVER reads this mapping; it still uses visible coaster
    # positions (Observed) or forecasts (Intent). This changes scenario
    # variation only, not Work's robot or MJCF physics.
    from intent_policy.scenarios.i1_coaster_sequence import CoasterSequenceTask
    original_sample = CoasterSequenceTask.sample_episode_variation
    x_slots = [0.155, 0.295, 0.435]
    y_row = -0.130

    def sample_row_variation(self, episode_seed, spec=None):
        variation = original_sample(self, episode_seed, spec)
        if layout == 'original':
            return variation
        rng = np.random.default_rng([int(episode_seed), 811])
        order = rng.permutation(3)
        for i, slot in enumerate(order):
            variation["coaster_sequence"][i]["destination_xy"] = [
                x_slots[int(slot)], y_row
            ]
            variation["coaster_sequence"][i]["display_slot"] = (
                "left", "middle", "right"
            )[int(slot)]
        return variation

    CoasterSequenceTask.sample_episode_variation = sample_row_variation
    # Same sampled human phase timings for both policies, unchanged robot limits.
    tempo_cfg = {
        "original": {},
        "balanced": {"take_reach_s": [0.9, 1.2], "carry_s": [1.3, 1.7],
                     "lower_s": [0.45, 0.65], "speed_scale": [0.9, 1.1]}
    }[tempo]
    scene=make_scenario("i1_coaster_sequence", {"expert":{"rule":expert},
        # Generated HY-Motion clips were not checked into the old repository.
        # Use its own deterministic minimum-jerk scripted human instead.
        "human_behavior":{"motion_bank":None, **tempo_cfg}})
    controller=RestrictedActionConfig.load("configs/controller/continuous_motion.yaml")
    controller.gripper["open"]=grasp_clear_m
    obs_cfg=ObservationConfig(cameras=("high",),width=640,height=480)
    observed=[]
    carried_frames={}
    weld_seen=set()
    stage_timeline=[]
    last_stage=None
    event_times={}
    hand_speeds=[]
    last_hand=None
    gripper_stage_samples=[]
    demonstration_states=[]
    demonstration_images=[]
    demonstration_actions=[]
    demonstration_stages=[]
    demonstration_times=[]
    video=out/f"I1_{expert}_{robot_profile}_work.mp4"
    # One large fixed high camera; images are RGB, not keypoint overlays.
    with imageio.get_writer(video,fps=round(20*speedup),codec="libx264",
                            macro_block_size=1,quality=8) as writer:
        def record_frame(obs,decision,target,s):
            nonlocal last_stage, last_hand
            stage=decision.extras.get("gripper_stage",decision.extras.get("expert_stage"))
            if stage in ("above_grasp","verify_grasp_clearance","close","release","open_wide_at_rest"):
                gripper_stage_samples.append({"t":round(float(s.time),2),"stage":stage,
                    "width_m":round(float(s.env.data.qpos[s.env.gid]),5),
                    "target_m":round(float(target[6]),5)})
            if stage!=last_stage:
                stage_timeline.append({"t":round(float(s.time),3),"round":int(s.sequence_index),"stage":stage})
                last_stage=stage
            for e in s.get_events():
                typ=e["event_type"]
                val=e.get("payload") or {}
                if typ in ("human_cue_onset","object_grasp","object_release","protocol_step_complete"):
                    key=f"{typ}/{val.get('step',val.get('object',''))}/{val.get('actor','')}"
                    event_times.setdefault(key,round(float(e["timestamp"]),3))
            hand=np.asarray(s.human.pos,float)
            if last_hand is not None:
                hand_speeds.append(float(np.linalg.norm(hand-last_hand)/s.dt))
            last_hand=hand.copy()
            for item in s.rounds:
                key = item["coaster"]
                if s.human.holding == key:
                    carried_frames[key] = carried_frames.get(key, 0) + 1
                    if s.env.data.eq_active[s.weld_id[key]]:
                        weld_seen.add(key)
            frame=np.asarray(obs["observation.images.high"],dtype=np.uint8)
            # Genuine ACT-format supervised targets: causally observed RGB and
            # proprioception paired with expert's executed 7D joint command.
            # The gripper opening target is the 7th action component.
            demonstration_states.append(np.asarray(obs["observation.state"],dtype=np.float32).copy())
            demonstration_images.append(frame[::5,::5].copy())  # 640x480 -> 128x96
            demonstration_actions.append(np.asarray(target,dtype=np.float32).copy())
            demonstration_stages.append(stage or "unknown")
            demonstration_times.append(float(s.time))
            writer.append_data(frame)
            observed.append(float(s.time))
        try:
            result=run_episode(scene,FluidExpertAgent(),seed,controller,
                               obs_cfg=obs_cfg,on_frame=record_frame,
                               keep_trace=False)
        finally:
            scene.close()
    if result["success"]:
        required={item["coaster"] for item in scene.rounds}
        assert required <= weld_seen, "No verified human weld: " + repr(required - weld_seen)
        assert all(carried_frames.get(k,0)>1 for k in required), "Human did not carry all coasters"
    if result["success"]:
        # In MuJoCo's action interface, idx 6 is the gripper target that ACT
        # learns as a continuous joint target; do not store the stage as input.
        demo_file=out/f"I1_{expert}_{robot_profile}_demo.npz"
        np.savez_compressed(
            demo_file,
            observation_state=np.asarray(demonstration_states,dtype=np.float32),
            observation_images_high=np.asarray(demonstration_images,dtype=np.uint8),
            action=np.asarray(demonstration_actions,dtype=np.float32),
            expert_stage=np.asarray(demonstration_stages,dtype="U48"),
            timestamp_s=np.asarray(demonstration_times,dtype=np.float64),
        )
    else:
        demo_file=None
    summary={"scenario":"i1_coaster_sequence",
             "expert":expert,"seed":seed,
             "robot_profile":robot_profile,
             "coaster_layout":layout,
             "gripper_cruise_open_m":cruise_open_m,
             "gripper_grasp_clear_m":grasp_clear_m,
             "gripper_rest_open_m":full_open_m,
             "gripper_simultaneous_arm_motion":True,
             "gripper_samples":gripper_stage_samples,
             "demonstration_npz":demo_file.name if demo_file else None,
             "demonstration_format":"observation_state[T,10], observation_images_high[T,96,128,3], action[T,7] (joint targets), expert_stage[T] diagnostic only",
             "robot_speed_rule":"0.50m/s only robot prep and hand-TCP >0.40m; otherwise 0.39m/s" if robot_profile=="optimized" else "original 0.39m/s",
             "human_tempo":tempo,"human_tempo_config":tempo_cfg,
             "coaster_row":[{"slot":x,"xy_m":[v,y_row]} for x,v in zip(["left","middle","right"],x_slots)],
             "placement_order":[{"coaster":v["coaster"],"slot":v.get("display_slot","free"),"xy_m":v["destination_xy"]} for v in scene.rounds],
             "human_hand_speed_p95_mps":round(float(np.percentile(hand_speeds,95)),3) if hand_speeds else None,
             "human_hand_speed_p50_mps":round(float(np.percentile(hand_speeds,50)),3) if hand_speeds else None,
             "stage_timeline":stage_timeline,"event_times":event_times,
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
             "video":f"I1_{expert}_{robot_profile}_work.mp4",
             "note":"Playback speed does not change simulated physics or training actions"}
    (out/f"I1_{expert}_{robot_profile}_work.json").write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str),flush=True)
    if not result["success"]:
        raise SystemExit("Observed I1 failed; see JSON and video")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=300100)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--speedup",type=float,default=1.0)
    p.add_argument("--expert",choices=["observed","early"],default="observed")
    p.add_argument("--tempo",choices=["original","balanced"],default="balanced")
    p.add_argument("--robot-profile",choices=["baseline","optimized"],default="baseline")
    p.add_argument("--layout",choices=["original","row"],default="row")
    a=p.parse_args()
    if not (1<=a.speedup<=1.5):raise ValueError("speedup must be 1..1.5")
    run(a.seed,a.out,a.speedup,a.expert,a.tempo,a.robot_profile,a.layout)

if __name__=="__main__":main()
