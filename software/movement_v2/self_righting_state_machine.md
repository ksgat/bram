# Self-Righting State Machine Sketch

This is the movement-v2 equivalent of the paper's stand-up split.

The paper's useful pattern:

1. Use IMU quaternion to estimate whether the body is down or upright.
2. If down, run a recovery behavior.
3. If the seat/body becomes upright enough, transition to a normal standing
   posture.
4. Keep stand-up separate from walking.

For BRAM:

```text
state = upright

if up_projection < fallen_threshold:
    state = fallen

fallen:
    choose recovery sequence based on roll/pitch quadrant
    play keyframed kick sequence
    if up_projection > recovered_threshold:
        state = recover_to_neutral

recover_to_neutral:
    interpolate current servo commands to neutral
    state = upright
```

The first real hardware task is not RL. It is to discover three or four
`self_right_*` keyframe sequences by testing the physical robot:

- fallen on left side
- fallen on right side
- upside down / battery-side-up safe case
- awkward partial side case

Once those work manually, an RL policy can be trained around those same motion
families instead of searching from nothing.

