# HRC-ACT — I1 Preview (V2)

This is a **new clean-room implementation** of I1 from the user-approved `HRI_TASK_SPECS_V2` (10 November 2026). No code from `hri_new` or `v11_current` is inherited. Only **I1 nominal** is implemented; do not begin I2 or other tasks before preview review.

## What is implemented

- A MuJoCo table scene containing **3 distinct physical cups** and **3 distinct coasters**; an articulated 3-DOF robot arm and a kinematic human hand/coaster demonstration.
- Seeded deterministic human clip with varied coaster order/positions, identical for Observed and Intent.
- Shared joint-space quintic trajectory planner and IK, source-cup preparation for both experts, pick with switchable MuJoCo weld, transport, placement readiness check, release and state/event/penetration logs.
- Observed uses only currently visible coaster positions. Intent uses only a causal wrist history through the current instant.
- Two videos with the *same fixed overview camera*: `I1_observed.mp4` and `I1_intent.mp4`, plus JSON traces per run.
- Optional LeRobot episode dataset/video recording, and GitHub Actions paired preview artifact.

**Critical research limitation**: The Intent preview uses causal linear extrapolation as a **test surrogate**. It is **not ManiCast**. The actual network, trained checkpoint, coordinate transform and complete ten-step bi-arm forecast are **not integrated**; onset differences from these preview videos are **not experimental ManiCast→ACT claims**. Robot gripper is a suction-like weld attachment model, not a realistic parallel-jaw grasp. Human body and coasters are kinematic proxies, not fully simulated biomechanics.

## Run a preview

```bash
python -m pip install -e '.[test]'
MUJOCO_GL=egl hrc-i1 --expert paired --seed 300100 --out outputs/I1 --fps 10 --skip-dataset
```

Output includes separate MP4s and machine-readable event traces. To record LeRobot demonstrations:

```bash
python -m pip install -e '.[record,test]'
MUJOCO_GL=egl hrc-i1 --expert paired --seed 300100 --out outputs/I1_record --fps 10
```

## Test

```bash
MUJOCO_GL=egl pytest -q
```

The GitHub Actions workflow runs scene tests, renders the two MP4s and **fails** if both experts don't place all three cups. Download the `I1-seed-300100-preview` artifact from the latest successful workflow run to review videos and event logs.

## Status and boundaries

This is a **preview for user review**, not a finished I1 expert/data-collection pipeline. Preview data cannot be used as scientific evidence until it passes the V2 acceptance criteria in `specs/I1_ACCEPTANCE.md`, including actual ManiCast, safety validation, paired trials across seeds and absence of privileged information. Collision penetration, place errors and task outcomes are reported in JSON to prevent a generated MP4 being mistaken for successful manipulation.

Main branch is not modified by development; all implementation is under `feat/i1-clean-foundation` in draft PR #1.
