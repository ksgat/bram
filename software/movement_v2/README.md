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

- walking: keyframe/CPG-style deterministic controller
- yaw: keep the current learned yaw primitive if it remains useful
- self-righting: separate keyframed recovery first, RL only after hardware
  keyframes exist

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

To export a longer command table:

```bash
python3 software/movement_v2/keyframe_controller.py \
  --sequence walk_forward \
  --cycles 10 \
  --csv /tmp/bram_walk_v2.csv
```

