# Movement V2

This folder is a BRAM-specific pass at the approach from Inoue et al.,
`Body Design and Gait Generation of Chair-Type Asymmetrical Tripedal Low-rigidity Robot`.

The important idea is not their exact servo angles. Their robot has six servos
and BRAM has three. The useful part to copy is the control structure:

- define a few essential postures
- linearly interpolate between them
- repeat those transitions for walking
- handle self-righting as a separate behavior
- use IMU orientation to decide when to enter or exit recovery

## What Their Repo Actually Has

- `connect_walk.py`: open-loop walking by repeated keyframe interpolation.
- `connect_stand.py` / `connect_once_stand.py`: keyframed stand-up motions.
- `rl_walk.py`: learned ONNX walking policy using IMU quaternion plus recent
  action history.
- `rl_stand.py`: learned ONNX stand-up policy using the same observation style.

They do not have joystick-style walking control. Their walking scripts are
forward-motion behaviors, not full directional teleop controllers.

## BRAM V2 Plan

For BRAM, the practical split should be:

- forward: one PPO-trained specialist primitive
- backward: one separate PPO-trained specialist primitive
- yaw: one PPO-trained yaw-only generalist conditioned on yaw command
- self-righting: separate keyframed recovery first, RL only after hardware
  keyframes exist
- mixed joystick arcs: disabled for demos unless the isolated primitives are
  visually reliable

This intentionally avoids one full joystick policy. Forward/back do not share a
command-conditioned policy, and yaw is the only generalist.

The file `bram_v2_keyframes.json` stores normalized servo commands in BRAM's
servo order:

```text
front, back_left, back_right
```

Values are normalized controller actions in `[-1, 1]`, not microseconds.

## Smoke Test

From the repo root:

```bash
python3 software/movement_v2/keyframe_controller.py \
  --sequence walk_forward \
  --steps 20
```

Backward has a separate seeded sequence:

```bash
python3 software/movement_v2/keyframe_controller.py \
  --sequence walk_backward \
  --steps 20
```

To export a longer command table:

```bash
python3 software/movement_v2/keyframe_controller.py \
  --sequence walk_forward \
  --cycles 10 \
  --csv /tmp/bram_walk_v2.csv
```

Run the MuJoCo primitive smoke suite:

```bash
python3 software/movement_v2/smoke_primitives.py --seconds 4
```

It reports forward/back distance, commanded yaw radians, drift, max tilt, min
chassis height, action delta RMS, and early termination for idle, random-small,
forward, backward, yaw `+1`, yaw `-1`, and half yaw commands.

## Primitive PPO

Use the V2 primitive PPO environment for all three base primitives. The reward
shape follows the paper's walking reward weights:

- progress: `30`
- chassis height: `20`
- upright seat: `5`
- heading/alignment: `2`
- alive: `1`
- death/reset: `-1`
- action delta: `-2`
- hinge velocity: `-2`

BRAM centers the height, upright, and heading/alignment terms at zero and adds a
small active-command stall penalty so standing still does not beat command
progress. The upright term is implemented as a tilt corridor: tilt up to about
`0.45 rad` is allowed because the 1-DOF legs need body rocking, then the penalty
ramps up through about `0.80 rad` and becomes harsher beyond that.

Forward/back specialists use a broad direction reward instead of exact
chassis-axis tracking. Net motion inside a `45 deg` cone gets full progress
credit and fades out by `65 deg`; path waste, wrong-way motion, yaw drift, and
excessive tilt are penalized separately. A slightly skewed but straight,
repeatable crawl is acceptable for the base primitive.

Policy observation:

- previous `4` IMU quaternion frames
- previous `6` normalized servo command frames
- yaw command scalar in `[-1, 1]` for yaw only; forward/back specialists receive
  `0`

Policy output:

- `3` normalized servo targets
- `20 Hz` by default with `--frame-skip 25`
- ESP32 runtime runs the actor online at `20 Hz` and sends normal `50 Hz` PWM
  pulses to the servos

Train the forward specialist:

```bash
.venv-rl/bin/python software/movement_v2/train_primitive_ppo.py \
  --primitive forward \
  --total-steps 300000 \
  --num-envs 4 \
  --episode-seconds 20 \
  --no-randomize-reset \
  --entropy-coef 0.0 \
  --log-std-init -0.8 \
  --hidden-size 128
```

Train the backward specialist separately:

```bash
.venv-rl/bin/python software/movement_v2/train_primitive_ppo.py \
  --primitive backward \
  --total-steps 200000 \
  --num-envs 4 \
  --episode-seconds 8
```

Train the yaw-only generalist:

```bash
.venv-rl/bin/python software/movement_v2/train_primitive_ppo.py \
  --primitive yaw \
  --total-steps 200000 \
  --num-envs 4 \
  --episode-seconds 8
```

Do not enable domain randomization until the nominal primitives move cleanly.
The trainer writes `policy.pt`, `policy_best.pt`, and `metrics.csv` under
`software/movement_v2/runs/rl_primitives/`.

Current corrected-geometry checkpoints use the `15.0 cm` leg endpoint in
`software/gait_discovery/bram.xml`:

- forward: `runs/rl_primitives/forward_primitive_20260706_173603/policy_best.pt`
- backward: `runs/rl_primitives/backward_primitive_20260706_173609/policy_best.pt`
- yaw: `runs/rl_primitives/yaw_primitive_20260706_174051/policy_best.pt`

The forward/back checkpoints are stable 20 second crawl primitives. A free PPO
fine-tune from those seeds was tested and rejected because deterministic eval
collapsed toward stationary behavior; keep the selected `policy_best.pt` files
unless a new run beats them in closed-loop smoke. Yaw was fine-tuned after the
leg-length change and selected with a balanced score over `-1`, `-0.5`, `0.5`,
and `1` yaw commands.

Yaw training now uses a stricter `gait_discovery`-style yaw-in-place eval:
signed yaw progress must be balanced against planar drift throughout the
rollout, action delta, action acceleration, roll/pitch rate, support loss,
contact slip, and chassis height warnings. The per-step yaw reward keeps the
paper's progress/height/up/action/velocity structure, but adds lighter versions
of those yaw-in-place terms so PPO sees the same failure modes without exploding
the value loss.

The old `gait_discovery` yaw tables are useful diagnostics, but they are not a
clean teacher on the corrected `15.0 cm` model: full negative yaw can flip sign
and full-rate tables drift heavily. Do not export a new yaw checkpoint unless it
beats the current strict eval and the visual smoke test.

Closed-loop checkpoint smoke test:

```bash
.venv-rl/bin/python software/movement_v2/smoke_policy_primitives.py --seconds 8
.venv-rl/bin/python software/movement_v2/smoke_policy_primitives.py --seconds 20
```

Visualize the same online policies in MuJoCo:

```bash
.venv-rl/bin/python software/movement_v2/visualize_policy_primitives.py --case suite --seconds 8
```

On macOS the script relaunches itself under `mjpython` when opening the viewer.
Use `--case forward`, `--case backward`, `--case yaw_pos`, or `--case yaw_neg`
to inspect one primitive at a time; add `--repeat` to keep looping.

Compare the current yaw actor against the saved degraded yaw actor:

```bash
.venv-rl/bin/python software/movement_v2/visualize_yaw_compare.py --case suite --seconds 8
```

Known corrected-geometry smoke results:

- `8s`: forward `0.1666 m`; backward broad-direction primary `0.0114 m`
  (`x=-0.068 m`, `y=0.125 m`); yaw `+1=2.378 rad`, `-1=1.054 rad`;
  no terminations.
- `20s`: forward `0.5151 m`; backward `0.3886 m`; yaw `+1=2.317 rad`,
  `-1=0.892 rad`; no terminations. Negative yaw drifts more than positive yaw.

## ESP32 Binding

The ESP32 runtime uses a hybrid primitive stack by default:

- forward/back run actor policies online
- yaw uses the base yaw table controller (`kUseBaseControllerForYaw = true`)

This avoids promoting the 10 Hz yaw PPO fine-tune, which did not beat the
baseline strict eval. The online actor path still maintains the same observation
history as the MuJoCo env:

- previous `4` IMU quaternions, `wxyz`
- previous `6` emitted servo actions
- yaw command scalar for the yaw policy; `0` for forward/back

Export actor weights from the selected checkpoints:

```bash
.venv-rl/bin/python software/movement_v2/export_policy_header.py
```

This writes
`software/firmware/bram_esp32_controller/bram_policy_data.hpp`. The firmware
uses `bram_policy_controller.hpp` for tiny MLP inference and
`kUseOnlinePolicies = true` in
`software/firmware/bram_esp32_controller/bram_esp32_controller.ino`.
Yaw-only commands are routed through `bram_controller.hpp` while
`kUseBaseControllerForYaw` remains enabled.

Optional BNO08x IMU support is behind `BRAM_ENABLE_BNO08X_POLICY_IMU=1`. With
that flag enabled, the firmware reads the BNO080/BNO085 rotation vector as
`w,x,y,z` for the policy. Without it, the policy sees identity quaternion, which
is only useful for compile tests.

Then compile-check firmware:

```bash
arduino-cli compile --fqbn esp32:esp32:XIAO_ESP32C3 \
  software/firmware/bram_esp32_controller
```

## Runtime Arbitration

The ESP32 firmware maps joystick input to primitives with deadbands:

- left stick Y / `f` command: forward/back primitive
- right stick X / `y` command: yaw primitive
- both active: translation priority by default

Set `kBlendMixedCommands = true` in
`software/firmware/bram_esp32_controller/bram_esp32_controller.ino` only after
the isolated forward/back/yaw checks look clean.
