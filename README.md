# HRC-ACT · I1 clean foundation

A new I1-only prototype: MuJoCo simulation; a shared minimum-jerk joint-space planner; two experts with identical dynamics, task and camera; and LeRobot dataset + standalone MP4 recording. No code inherited from `hri_new` or `v11_current`.

**Scope/limitations:** This is a *minimal simulated reaching/handoff readiness* task, not a validated pick-and-place or a complete human robot collaboration benchmark. Robot uses a 2-DOF planar arm with position actuators. Human hand is a kinematic mocap proxy. No grasp, object transfer, force control, learned ACT policy or task-success benchmark is claimed. The Intent expert **requires externally supplied ManiCast trajectories**; `--demo-forecast` is explicitly synthetic smoke-test data, **not** ManiCast. Do not treat synthetic forecast results as experimental evidence.

## Install

```bash
python -m pip install -e '.[record,test]'
MUJOCO_GL=egl hrc-i1 --expert observed --out outputs/observed
MUJOCO_GL=egl hrc-i1 --expert intent --forecast path/to/manicast_prediction.npz --out outputs/intent
pytest -q
```

For a quick rendering/pipeline smoke test without ManiCast: `MUJOCO_GL=egl hrc-i1 --expert intent --demo-forecast --out outputs/intent_demo`.

Each run saves `overview.mp4`, `metrics.json` and an episode in `dataset/` using LeRobot v3 with a video feature. To compare experts fairly, run the same seed, frame rate and human trajectory. Each mode writes independently; never reuse one episode file for both modes.

## ManiCast forecast contract

NPZ keys: `times` shape (N,) seconds **relative to prediction instant**, strictly increasing and positive, and `positions` shape (N,3) in **MuJoCo world coordinates/meters**, representing predicted *human right wrist positions*. Recommended sampling: 10 points at 0.1…1.0 s. A real ManiCast integration must perform the trained model's normalization, camera/world transform and confidence calibration upstream. This prototype deliberately does not silently invent a model/checkpoint. Intent uses the earliest sufficiently stable predicted wrist position as a goal cue; it does not yet exploit all 10 timesteps/multi-joint forecasts. Expanding the adapter is future work.

## Planner

One planner for both experts: Cartesian target -> damped least-squares inverse kinematics -> joint-space minimum-jerk quintic trajectory with bounded speed, velocity continuity on replans and no per-task motion rules. Conservative target workspace clipping; not a certified collision-free planner. Targets are in a fixed vertical planar workspace.

Observed waits for measured wrist to reach a readiness threshold, then plans to that position. Intent can plan earlier from the external future position and falls back to observed if no credible forecast. Both use the same robot initial pose, controller, and camera.

## Reproducibility / status

`pytest` covers jerk endpoints, IK convergence and intent fallback. No physical feasibility validation, benchmark, genuine ManiCast inference or recorded MP4 artifacts are bundled in Git. LeRobot and offscreen video require ffmpeg and a functioning EGL/OSMesa backend; execution environment dependent.

The `--forecast` NPZ is read once at episode start: this is a single prediction snapshot, **not** an online streaming predictor. Before reporting early-react gains, add timestamped online forecast updates, paired evaluation and safety/collision tests.
