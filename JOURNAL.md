# Bram Journal

All detailed gait-discovery approaches, failed experiments, run IDs, and controller notes are documented in [`software/gait_discovery/gait_discovery_journal.md`](software/gait_discovery/gait_discovery_journal.md). The root journal is intentionally kept as a brief day-level synthesis.

## 6/19

Initial concept: a low-sensor tripod robot using large servos and a compact onboard controller, originally framed around learning locomotion and deciding what hardware/sensors were actually necessary.

## 6/29-6/30

Set up the Python/MuJoCo gait-discovery project, built the first usable tripod model from measured geometry, added rough hardware constraints, and trained early PPO policies. Forward locomotion became the first solved behavior, while broad joystick control, yaw, and aggressive sim-to-real noise exposed reward hacking, arcing, stationary local optima, and the need for stricter world-frame command metrics.

## 6/30

Reworked the environment and training process around command obedience, per-command evals, gated curricula, distillation, and straight-line polish. Fixed forward/back/yaw primitives proved individually learnable, but one shared PPO actor was unstable for full joystick control, which pushed the project toward expert distillation, deterministic CPG search, yaw action tables, and hybrid composition rather than a single end-to-end learned policy.

## 7/1

Corrected yaw scoring to use real planar heading instead of body gyro wobble, retrained/exported learned yaw primitives, searched deterministic forward/back CPGs and mixed-arc blend grids, and settled on the current deployable architecture: deterministic forward/back CPGs plus learned left/right yaw tables plus a tuned grid blend for mixed commands. The standalone controller export now lives under the gait-discovery software folder and is the source for the future ESP32 port.
