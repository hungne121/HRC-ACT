# I1 acceptance gate (HRI_TASK_SPECS_V2, 2026-11-10)

**Status: BLOCKED — current implementation is not an I1 preview.** The old 2-DOF hand-reach prototype is a disposable plumbing smoke test, not a cup-placing task; do not use its videos or timing in experimental claims.

## Mandatory scenario
- **Exactly 3 identical cups** at robot-side source. Cups have evaluator-only object IDs and are never reused.
- **Exactly 3 coasters** carried one at a time by a human; placed in randomized unused left, middle and right workspace regions, with seed-dependent exact positions, order and trajectory.
- Coaster readiness requires real contact with the table, stable pose, and observed human hand clearance; object release must not happen before all three.
- Robot may prepare for pickup independently of destination in **both** experts.
- Observed commits to a target from current observations; Intent may commit earlier using **causal real ManiCast** forecasts computed only from history through t, never simulator future/ground truth.
- Both experts use identical scene/clip/controller/collision wrapper. No oracle goal, task_id, case_id, future human clip or hidden metadata fed to policy/expert.
- Complete episode means 3 distinct stable cups resting on 3 distinct coasters, robot withdrawn, with no unsafe release.

## Preview delivery gate
- [ ] MuJoCo physically simulates cups, coasters, table, arm/gripper, human hand (and observable motion): not storyboard or decorative animation
- [ ] Scene camera can see the coasters, hand trajectory, all cup interactions, and gripper
- [ ] Seeded human generator yields SAME human clip and exact initial object poses across Observed / Intent
- [ ] Unified planner avoids path discontinuities, object/table/hand collision and satisfies motion limits
- [ ] Pickup and transport are physically represented; no teleporting cups
- [ ] Release gate checks coaster contact+stability+human-clearance in **both** modes
- [ ] ManiCast actual inference adapter: causal full 10-step left/right wrists, elbows, shoulders, timestamps, latency and valid mask; no synthetic/oracle input in claimed Intent results
- [ ] Produce independent `I1_observed.mp4`, `I1_intent.mp4`, with same static overview camera
- [ ] Per-run events: task_active, human_motion_onset, target_observed, prep_onset, robot_commit, human_ready, robot_ready, place_complete ×3, episode_end
- [ ] Metrics and logs: seed, human_clip_id, scene_init_hash, action trace, forecast timestamp, safety_override_count, outcome, coaster_stable time, cup_place time and useful onset time
- [ ] Visual and automatic checks over multiple paired seeds: 3/3 success, contact-penetration report, safety report, no privileged information leak, exact same human clip
- [ ] User reviews preview; **do not start I2 or other tasks before approval**

## Why blocked
Existing `src/hrc_act/sim.py` has no cups/coasters or gripper. `run.py` has no pickup or release, and `--demo-forecast` is synthetic, not ManiCast. These must be replaced, not promoted as complete I1. CI unit tests only verify generic helpers. The local environment used for this review lacks MuJoCo / LeRobot and cannot install packages (network unavailable), therefore neither physically valid preview nor video has been generated or verified.

## Source-of-truth
The user-supplied `HRI_TASK_SPECS_V2(1).zip` (2026-11-10), especially `01_INSTRUCTOR/I1_OBSERVED.md`, `01_INSTRUCTOR/I1_INTENT.md` and `00_COMMON/01_RULES_AND_INTERFACES.md`, is the governing source. It is currently available as a conversation upload, **not yet checked into the repository**. Preserve all task requirements and flag engineering assumptions explicitly.
