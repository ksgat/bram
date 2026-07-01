# Bram Current Policy Package - 2026-07-01

This folder is the compact policy snapshot to share/push.

## Reviewable Controller

Current usable policy is the exact component controller:

- forward: deterministic CPG gait in `gaits/forward_best_params.json`
- backward: deterministic CPG gait in `gaits/backward_best_params.json`
- left yaw: exported PPO action table in `yaw_tables/yaw_left_policy_table.json`
- right yaw: exported PPO action table in `yaw_tables/yaw_right_policy_table.json`

Visual review command from repo root:

```bash
.venv-rl/bin/mjpython train_cpg_modulator.py \
  --teacher-controller \
  --forward-gait pushed_runs/current_policy_20260701/gaits/forward_best_params.json \
  --backward-gait pushed_runs/current_policy_20260701/gaits/backward_best_params.json \
  --yaw-left-table pushed_runs/current_policy_20260701/yaw_tables/yaw_left_policy_table.json \
  --yaw-right-table pushed_runs/current_policy_20260701/yaw_tables/yaw_right_policy_table.json \
  --view \
  --view-command primitives \
  --speed 0.8
```

Headless check:

```bash
.venv-rl/bin/python -u train_cpg_modulator.py \
  --teacher-controller \
  --forward-gait pushed_runs/current_policy_20260701/gaits/forward_best_params.json \
  --backward-gait pushed_runs/current_policy_20260701/gaits/backward_best_params.json \
  --yaw-left-table pushed_runs/current_policy_20260701/yaw_tables/yaw_left_policy_table.json \
  --yaw-right-table pushed_runs/current_policy_20260701/yaw_tables/yaw_right_policy_table.json \
  --eval-episodes 1
```

## Status

Primitive commands are reviewable: idle, forward, backward, left yaw, right yaw.

Mixed forward+yaw arcs are not final. Naive blending can terminate or underperform, so do not present that as solved Xbox control yet.

The learned CPG-modulator checkpoint in `experimental_cpg_modulator/` is preserved for reference, but it is not the current best visual policy because it loses too much left-yaw authority.
