6/19

so the idea is this

<img width="572" height="371" alt="image" src="https://github.com/user-attachments/assets/5bea86bb-be6d-4429-af32-2960aed3dcdc" />

basically im gonna rl walking or somehting idk ima do some research bu im gonna cad first im gonna use like big servos and mag encoders and my pi 3 and probably a pico also bu idk ill see type

night 6/29-30

- Set up the Python/MuJoCo project around `bram.xml`.
  Added Python 3 env tooling, MuJoCo viewer entrypoints, PPO training, policy visualization, and PufferLib tests.

- Tried different approaches for the MuJoCo model.
  Codex wrote all the models conversions, after 2 failed attempts at getting it to pull from cad (spatial reasoning isnt there yet) I just fed it the hinge offsets and it worked fine.

- Added realistic-ish hardware constraints.
  Servos were limited to weakened torque around `3.0 Nm`, actuator range was set to about 160 degrees total, and the observation model was kept close to IMU plus servo command history.

- Built a low-sensor Gym environment.
  Observations use gyro, accel, gravity estimate, command, heading error, recent servo actions, and command deltas; reward can use privileged sim state because it is not deployed.

- Started with simple forward reward.
  It learned movement, but body-frame forward reward let the robot arc and still score well.

- Tried heavier sim-to-real noise early.
  Full randomization made learning much worse and exposed reward hacking before the base gait was stable.

- Tried directional/vector command ideas.
  Full direction control was too much too early and confused "move along vector" with "rotate heading"; forward-plus-yaw became the better intermediate target.

- Tried forward/yaw mixed policies.
  Mixed command training exposed catastrophic command failures, especially yaw/rotate cases, so the work narrowed back to forward first.

- Found `command_distance` was not enough for checkpoint selection.
  It was useful as a diagnostic, but it could reward arcing or body-frame progress instead of straight world-frame progress.

- Added heading-aware world-frame reward.
  Forward progress became line progress along the desired heading, with explicit penalties for lateral drift, yaw error, heading error, jitter, airtime, and falling.

- Hit an early-death exploit.
  The policy learned that dying fast could accumulate less negative reward than surviving badly, so terminal penalties and survival-aware eval scoring were added.

- Hit a stochastic lunge failure.
  Some policies moved only with sampled actions while deterministic mean actions collapsed, so deployable deterministic eval became the main sanity check.

- Switched to "gentle parenting" reward shaping.
  The working split was large praise for alive/upright forward progress, firm fall penalties, and ramped-in penalties for heading, drift, jitter, airtime, and tracking.

- Best clean forward policy so far: `runs/ppo_20260630_002841`.
  It reached about `1.93 m` command progress over a full 400-step episode and looked visually pretty straight despite some measured yaw/side drift. 

- Added mild domain-randomization strength.
  Full DR was too harsh, so randomization now scales around nominal values with `--domain-randomization-strength`, defaulting to `0.45`.

- Added late-episode polish costs.
  Smoothness, jerk, hinge velocity, and airtime penalties now ramp later so exploration is not crushed before a gait emerges.

- Best mild domain-randomized policy so far: `runs/ppo_20260630_004158`.
  It reached about `1.19 m` clean eval distance over 400 steps and survived mild randomized tests, but was slower/uglier than the clean demo gait.

- Current conclusion.
  The clean gait is decent; the mild-DR gait is the sim-to-real starting point; the next best step is checkpoint warm-start and fine-tuning the clean policy with mild DR around `0.25-0.35` and lower learning rate. And then after that Yaw policy and then merging them!


Overall productive day, probably could have gone faster had I "multi-agent-orchestrachted" (yuck) but still I got pretty far.
By tomorrow, I want to get the yaw policy working and merging and then I will run a larger run on my old laptop which has a 3050

The architecture has changed since my pitch so I will be running the policy's actor weights onboard the esp32 and that will be it, trying to keep the model pretty small as a result of that (cant find a pi zero 2 w and its more overhead for like marginally better inference / headroom idk pretty stupid on my part) total bom is looking like
servos: 25$ (3x): 75
xiao esp32c3 off amazook: 10
bec: ~7-10$
imu off adafruit: 20
misc hardware (M3 hardware kit): 15

6/30 afternoon

- Made chassis, battery, and servo cans non-contact in MuJoCo.
  Since I do not trust a random PLA/battery friction estimate, the body now acts as visual/mass geometry and the floor interaction is mostly feet/legs instead of fake chassis sledding.

- Changed "alive" into a liberal limit gate.
  The env no longer pays the robot for being upright/alive; it only terminates for battery-side-down-ish orientation, extreme low body height, or invalid sim state.

- Added retroactive training visualization.
  `train_ppo.py` can save snapshots with `--snapshot-interval`, and `visualize_training_run.py` replays those checkpoints in order after the run.

- Found the stationary local optimum.
  With the v4 reward, standing still on forward command scored around `-7`, so PPO learned quiet non-motion instead of risking movement.

- Reworked exploration/reward into `forward_yaw_heading_v5`.
  Added command stall penalties, immediate command-motion bonus, higher movement payoff, lower early regularization, and higher PPO entropy/action std so standing still gets punished harder.

- Current reward direction.
  Command obedience stays primary; next polish should target crawl quality with foot-slip penalties and progress-per-effort rather than generic prettiness.

6/30 evening general-policy rewrite

- Compared the old movers against the newer generalist runs.
  The successful runs were forward specialists with dense progress reward; the failed generalist runs mostly learned lower-penalty stationary behavior.

- Identified forward as the only solved primitive.
  `runs/ppo_20260630_002841` and related v3 runs moved well, while yaw/back/mixed command training still had near-zero command distance.

- Rewrote the env as `forward_yaw_heading_v9`.
  The reward now pays dense forward/yaw progress first, then applies wrong-way, stall, extra-motion, heading, smoothness, and crawl-polish penalties after the robot has a chance to move.

- Slowed the command curriculum.
  Training now stages positive forward, forward/back, yaw-only, gentle arcs, then full joystick commands instead of blending the whole command space too early.

- Updated checkpoint selection.
  Eval defaults to the full 13-command suite, and score now includes command distance so lower-penalty non-motion is less likely to look like the best policy.

- Smoke check result.
  Zero action is now clearly bad for active commands, idle is rewarded for staying still, and a 4k PPO smoke completed with positive early curriculum returns but no learned full general gait yet.
