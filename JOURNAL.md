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

6/30 gated joystick-policy environment work

- Added per-command deterministic eval metrics.
  `metrics.csv` now logs each command separately (`fwd1`, `back1`, yaw signs, half commands, arcs) so average reward cannot hide a broken direction.

- Replaced time-only curriculum with competence gates.
  Randomized-command training now holds stages until deterministic eval passes required command thresholds for consecutive evals.

- Added weak-command oversampling.
  The trainer samples failed command families more often after eval, so backward/yaw failures get more rollout data instead of being hidden by easy forward commands.

- Changed command stages to isolated primitives first.
  The staged order is now forward, backward with forward replay, yaw-left, yaw-right, primitive mix, arcs, then full joystick.

- Rewrote command-sign punishment into `forward_yaw_heading_v10`.
  Wrong signed movement is punished strongly from the start of the episode, because joystick command obedience matters more than crawl smoothness right now.

- Medium-run result.
  Gated 50k runs now expose the bottleneck clearly: forward passes and unlocks stage 1, but backward does not become positive within the remaining 50k budget.

- Backward feasibility check.
  Fixed backward v10 still learns strongly in 50k (`runs/ppo_20260630_185845`, about `1.02 m`), so the issue is curriculum/transfer, not backward being physically impossible.

- Current conclusion.
  The environment is much better instrumented and stricter, but the one-policy curriculum is not solved yet; the next work should focus on longer stage budgets, warm-start/resume support, or stage-specific optimizer handling so backward can be learned without collapsing the forward primitive.

6/30 follow-up on backward transfer

- Tried weak backward contrast during the forward stage.
  A small backward sample rate in stage 0 did not stop the policy from learning a mostly command-agnostic forward gait.

- Tried balanced forward/backward from scratch.
  The policy stayed near stationary over 50k, so the first gate is too hard when both translation signs are equally weighted from random initialization.

- Added `--init-checkpoint` support.
  The trainer can now initialize a run from a compatible checkpoint while using a fresh optimizer, which is useful for staged experiments and later resume/warm-start work.

- Tried warm-starting from a fixed backward expert.
  Starting from `runs/ppo_20260630_185845/policy_best.pt` preserved backward initially but did not learn forward; both signs decayed toward non-motion by 50k.

- Tried command-phase observation features.
  Extra command-times-gait-phase features increased obs dim to 25 but did not improve the 50k translation-sign curriculum and caused less stable eval, so the env was reverted to v10/21-observation.

- Current blocker.
  Fixed forward and fixed backward both learn, but one shared PPO actor does not yet learn reliable signed translation under the mixed command distribution. The next promising direction is probably stage-specific training mechanics (longer stage budgets, lower LR after unlock, or explicit distillation/replay from fixed-command experts), not more reward scalar tweaking.

6/30 distill-then-anchor breakthrough

- Added explicit expert distillation.
  `train_ppo.py` can now collect expert rollouts, behavior-clone one command-conditioned actor, and optionally run BC-only to test representation before PPO.

- Added PPO anti-forgetting controls.
  PPO now supports command-family advantage normalization, expert replay BC, and a frozen post-BC action anchor so updates do not immediately erase learned command modes.

- Verified one compact actor can represent the primitive gaits.
  Primitive-only BC (`runs/ppo_20260630_200631`) fit four experts to very low MSE and passed forward, backward, yaw-left, and yaw-right without PPO.

- PPO from primitive BC finally stayed stable.
  Anchored PPO (`runs/ppo_20260630_200728`) kept all four primitives passing through stage 4, proving the previous collapse was PPO forgetting, not model capacity.

- Added optional arc experts.
  The old v9 arc-forward-left policy still works in v10, so `train_ppo.py` now accepts arc expert checkpoints for distillation and replay.

- Five-expert BC fixed the left-forward arc.
  Adding the arc-forward-left expert (`runs/ppo_20260630_201507`) raised `arc_fl` from failing to about `0.269`; only `arc_bl` remained slightly weak.

- Reached the first full-suite pass.
  Anchored PPO from five-expert BC (`runs/ppo_20260630_201558`) advanced to stage 5 and had a full-pass snapshot at update 40: primitives and all arcs cleared their gates.

- Current best visual checkpoint.
  Use `runs/ppo_20260630_201558/policy_full_pass.pt`; it is copied from the update-40 snapshot because the final scalar-best checkpoint later regressed `arc_bl`.

- Fixed checkpoint scoring.
  Eval score now includes full-command deficit, so future `policy_best.pt` selection should prefer complete command obedience over higher average distance with one failed command.

- Added single-policy command-suite visualization.
  `visualize_policy.py --command-suite` cycles idle, forward/back, yaw, half commands, and arcs for one checkpoint instead of relaunching the viewer per command.

- Headless visualizer verification.
  `runs/ppo_20260630_201558/policy_full_pass.pt` clears the command suite with mean command progress about `0.408`; all 13 commands ran full length with no weak command under the current thresholds.

- Current visual command.
  Run `.venv-rl/bin/python visualize_policy.py --checkpoint runs/ppo_20260630_201558/policy_full_pass.pt --command-suite --episodes 13` to inspect the general joystick policy in the MuJoCo viewer.

6/30 forward/back line polish

- Added a straight-line polish mode.
  `train_ppo.py --straight-line-polish` samples mostly pure forward/back commands and adds lateral velocity, cross-track error, heading drift, and yaw-rate penalties only for those commands.

- Added line-quality eval metrics.
  Per-command metrics now include cross-track and heading error, and checkpoint scoring can include a straight-line score so average command reward does not hide drift.

- First polish run improved the full-suite checkpoint.
  `runs/ppo_20260630_210517` improved straight-line score from about `-80.6` to `-31.6`, with especially better backward tracking.

- Second polish run is the current forward/back candidate.
  `runs/ppo_20260630_210825/policy_best.pt` reached mean command progress about `0.437` across the full command suite and straight-line score about `-26.9`; it is better but not yet rail-straight.

- Current polished visual command.
  Run `.venv-rl/bin/python visualize_policy.py --checkpoint runs/ppo_20260630_210825/policy_straight_polish.pt --command-suite --episodes 13` to inspect the forward/back-polished general policy.

6/30 smooth gait-generator search

- Added black-box gait search.
  `search_gait.py` optimizes a smooth sinusoidal three-servo gait with frequency, per-leg center, amplitude, phase, and second-harmonic terms.

- First forward search found a usable open-loop crawl.
  `runs/gait_search_forward_refine/best_params.json` gets about `0.343 m` in 4 seconds with about `4 mm` terminal cross-track error in the nominal sim.

- Longer forward eval exposes heading drift.
  The same gait gets about `0.622 m` in 8 seconds but drifts to about `0.59 rad` heading error, so hardware should use IMU heading trim rather than pure open-loop.

- Basic robustness check passed.
  With reset/domain randomization enabled, the forward gait averaged about `0.327 m` over 4 seconds across three episodes without terminating.

- Added heading-corrected gait parameters.
  `search_gait.py` now supports heading/yaw-rate feedback terms that trim the smooth gait using the same heading error and gyro yaw rate a 9DoF IMU would provide.

- First heading-corrected forward gait improved the 8-second line.
  `runs/gait_search_forward_heading/best_params.json` gets about `0.800 m` in 8 seconds with about `5.9 cm` cross-track error and `0.10 rad` final heading error.

- Current correction note.
  The short search mostly selected yaw-rate damping instead of absolute heading Kp; on hardware, heading Kp should remain available and be tuned against the real steering direction.

- Added inverse-seeded backward search.
  `search_gait.py --init-inverse` can seed backward from a forward gait by phase-inverting the smooth waveform, then refine it independently.

- Current backward gait.
  `runs/gait_search_backward_heading_refine/best_params.json` gets about `1.686 m` backward in 8 seconds with about `4.8 cm` cross-track error and `0.31 rad` heading error; randomized 4-second eval averaged about `0.565 m`.

- Current yaw-left gait.
  `runs/gait_search_yaw_left/best_params.json` rotates about `4.28 rad` in 4 seconds with about `8.6 cm` planar drift; randomized eval averaged about `4.54 rad`.

- Current yaw-right gait.
  `runs/gait_search_yaw_right/best_params.json` rotates about `3.68 rad` in 4 seconds with about `11.9 cm` planar drift; randomized eval averaged about `3.63 rad`.

- Tried cloning PPO yaw into the smooth sine gait.
  `fit_policy_gait.py` records PPO actions and fits the smooth gait formula, but the fit collapses toward the mean action because PPO yaw is high-variance/feedback-like rather than a clean time-only waveform.

- Added direct PPO action-table export.
  `export_policy_table.py` records deterministic PPO actions into a 50 Hz table that can be replayed or ported to firmware as a lookup/interpolation gait.

- Current PPO yaw table result.
  From `runs/ppo_20260630_210825/policy_straight_polish.pt`, yaw-right table replay gets about `5.14 rad` in 4 seconds with about `3.7 cm` planar drift; yaw-left gets about `1.20 rad` but drifts about `36 cm`, so yaw-left still needs a better source policy or a separate mirror/refinement.

- Tried mirroring yaw-right into yaw-left.
  `mirror_policy_table.py` brute-forces servo sign flips, rear swaps, time reversal, phase shifts, and optional full servo permutations from the good yaw-right table.

- Best mirrored yaw-left result.
  `runs/policy_table_yaw_left_mirror/yaw-left_mirrored_table.json` gets about `0.98 rad` in 4 seconds with about `13 cm` planar drift; this is cleaner than the bad yaw-left PPO table but far weaker than yaw-right.

6/30 hybrid CPG + learned residual BC

- Added BC hybrid controller.
  `train_hybrid_bc.py` keeps deterministic CPG forward/back as the base action and trains a small residual network that is hard-gated by yaw command, so pure forward/back and idle cannot be modified by the learned layer.

- First obs-based residual was too weak.
  `runs/hybrid_bc_merged_v1` fit the dataset but yaw collapsed in rollout, likely because small action errors changed the env observations the residual depended on.

- Switched to phase/command residual features.
  The better version uses command, base action, previous action, and phase harmonics instead of full IMU/env obs, making the learned yaw residual behave like a compact learned waveform.

- Current BC baseline.
  `runs/hybrid_bc_phase_v2/hybrid_bc_current.pt` completes the command suite with no terminations: forward `0.408 m`, backward `0.795 m`, yaw-left `5.15 rad`, yaw-right `2.43 rad` over 4 seconds.

- Balanced variant was not selected.
  `runs/hybrid_bc_phase_v4` made pure yaw-left/right more balanced (`3.31/2.58 rad`) but terminated on `arc_bl`, so it is less safe as the current checkpoint.

- Remaining BC issues.
  Half-speed forward/back are still weak because amplitude-scaled CPGs do not crawl well at low magnitude, and yaw-right still under-reproduces the raw PPO yaw-right table (`5.14 rad`).

- RL fine-tune decision.
  Do not start residual RL yet; first fix low-speed CPG scaling or discover dedicated half-speed gaits, then add better arc teachers so RL is not asked to repair bad supervised targets.

- Fixed hybrid viewer freeze.
  `train_hybrid_bc.py --view` now resets the same MuJoCo env instead of recreating it under an active viewer, and `--view-command` can jump directly to a specific command like `yaw_l1` or `fwd1`.
