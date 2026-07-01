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

- Added pure-yaw direct bypass.
  Pure yaw commands now bypass the residual network and use the best direct yaw sources: searched yaw-left and PPO-table yaw-right.

- Current pure-yaw result after bypass.
  With `runs/hybrid_bc_phase_v2/hybrid_bc_current.pt`, yaw-left is about `4.28 rad` in 4 seconds with about `8.6 cm` planar drift, and yaw-right is about `5.14 rad` with about `3.7 cm` planar drift.

6/30 15 cm symmetric model retest

- Fixed the remaining leg-length asymmetry.
  `bram.xml` now uses the same 15 cm tube/tip geometry for all three legs; the old back-right leg was longer than the others.

- Fresh PPO yaw-left still failed from scratch.
  `runs/ppo_yaw_left_15cm/ppo_20260630_233657` reached only about `0.88 rad` yaw in direct rollout with about `30 cm` planar drift, so the geometry bug was not the full left-yaw problem.

- Fresh PPO yaw-right learned strongly.
  `runs/ppo_yaw_right_15cm/ppo_20260630_233657` reached about `12.12 rad` yaw in direct rollout with about `27 cm` planar drift; high yaw rate, but still too much absolute drift for a clean firmware primitive.

- Old yaw artifacts changed under the corrected model.
  The searched yaw-left gait improved to about `4.19 rad` with about `5.4 cm` drift, while the old yaw-right action table degraded to about `2.14 rad` with about `9 cm` drift.

- Current conclusion.
  The asymmetric leg was a real model bug, but left/right yaw learning is still not equivalent; do not keep training yaw-left PPO from scratch without seeding it from the searched gait or using a residual/teacher setup.

7/1 body-height limit

- Added an aggressive no-body-drag constraint.
  The MuJoCo env now terminates when chassis center height drops below `3.5 cm` and starts penalizing below `5.0 cm`.

- Wired height into gait scoring.
  `search_gait.py` and `mirror_policy_table.py` now track min height plus warning/hard height deficits, so search and table audits reject low chassis gaits.

- Invalidated the latest strict left-yaw candidate.
  `runs/gait_search_yaw_left_strict_15cm/best_params.json` now terminates after `22` steps with `min_height=0.03475 m`, so it was relying on dropping too low.

7/1 height-limited yaw retraining

- Added target-rate yaw scoring.
  Yaw search now scores against the commanded yaw distance instead of rewarding unbounded spin, so overspin cannot hide drift.

- Kept right yaw PPO.
  The right yaw PPO remains the trusted right primitive; the partial right search was stopped rather than replacing a visually good policy.

- Dedicated left-yaw PPO still failed.
  `runs/ppo_yaw_left_height_v1/ppo_20260701_085732` was warm-started from the good right-yaw PPO, but ended at only about `0.020` command distance after 50k steps.

- Left yaw v1/v2 search found feasibility.
  `runs/gait_search_yaw_left_height_v1` found safe but weak yaw (`2.19 rad`), then `height_v2` found target-speed yaw but with too much drift (`5.15 rad`, `13.8 cm` planar drift).

- Current left yaw primitive.
  `runs/gait_search_yaw_left_height_v3_lowdrift/best_params.json` gets about `4.94 rad` against a `4.8 rad` target over 4 seconds, with about `3.0 cm` final planar drift, `5.9 cm` max drift, `min_height=3.66 cm`, and no hard height violation.

- Updated hybrid defaults.
  `train_hybrid_bc.py` now defaults yaw-left to the height-limited low-drift gait and yaw-right to the 15 cm PPO-exported table.

- Longer left-yaw follow-up.
  `height_v4_8s` and `height_v5_lowpathdrift` searched longer 8-second holds; v5 survives 8 seconds with about `8.96 rad` yaw, `5.3 cm` final drift, and `6.5 cm` max drift, but its 4-second endpoint drift is about `11.3 cm`.

- Mixed-horizon search was not selected.
  `height_v6_mixed` evaluated both 4-second and 8-second horizons, but ended with worse endpoint drift (`8.8 cm` at 4 seconds and `10.5 cm` at 8 seconds), so it is not the current default.

- Current left-yaw recommendation.
  Use `height_v3_lowdrift` for short viewer/command-suite tests and compare `height_v5_lowpathdrift` only if testing sustained 8-second yaw holds.

7/1 left PPO level-penalty test

- Added explicit levelness reward pressure.
  `bram_env.py` now penalizes chassis tilt angle directly via `level_penalty`, with a stronger cost after about `0.18 rad` of roll/pitch tilt.

- Fresh left-yaw PPO still failed.
  `runs/ppo_yaw_left_level_v1/ppo_20260701_113309` trained for 50k steps from scratch under the level penalty, but best headless eval only reached about `0.047` command progress over a full 400-step episode.

- Conclusion.
  Levelness penalty reduces the incentive to twitch/tilt, but by itself it makes PPO settle into weak near-stationary motion instead of discovering left yaw; left yaw likely needs an action prior/residual/teacher from a deterministic gait rather than plain PPO from scratch.

7/1 planar-yaw metric fix

- Found the yaw-score bug.
  The env was integrating MuJoCo gyro `z` as yaw progress; chassis rocking/tilting can create body-frame gyro-z without meaningful planar heading change.

- Changed yaw progress to projected heading delta.
  `bram_env.py` now computes `planar_yaw_rate` from wrapped chassis heading change between steps and uses that for `yaw_progress`, `yaw_error`, `yaw_distance`, and command distance; gyro-z is kept only as `gyro_yaw_rate` diagnostics.

- Invalidated previous left-yaw gait scores.
  `runs/gait_search_yaw_left_height_v3_lowdrift/best_params.json` drops from the old apparent `4.94 rad` to only about `0.33 rad` of real planar yaw over 4 seconds, confirming the visual wobble complaint.

- Right yaw remains real.
  `runs/policy_table_yaw_right_15cm/yaw-right_policy_table.json` still replays about `7.07 rad` of planar yaw over 4 seconds under the fixed metric, though it drifts about `15 cm`.

- Removed invalid left-yaw default.
  `train_hybrid_bc.py` no longer defaults to the old left-yaw gait; left yaw needs retraining under the planar-yaw metric before being used as a direct primitive.

7/1 planar-yaw left retraining

- Broad deterministic search failed under real planar yaw.
  `runs/gait_search_yaw_left_planar_v1` mostly terminated early and found only about `0.011 rad` of real left yaw, confirming the old sine gait was not a valid seed.

- Mirroring right yaw was not enough.
  `runs/policy_table_yaw_left_mirror_planar_v2_allperm` tried all servo permutations/signs/time shifts from the real right-yaw table; the best result was only about `1.17 rad` left yaw against a `4.8 rad` target.

- Boosted PPO yaw reward after fixing the metric.
  `bram_env.py` now uses `forward_yaw_heading_v14_planar_yaw_boost`, with stronger reward for real planar yaw, stronger wrong-way yaw penalty, and more translation penalty during rotate-in-place.

- PPO finally learned real left yaw.
  `runs/ppo_yaw_left_planar_v1/ppo_20260701_114808/policy_best.pt` reaches about `0.428` command progress, meaning about `4.28 rad` real planar left yaw over 8 seconds.

- Short fine-tune barely improved it.
  `runs/ppo_yaw_left_planar_v2_finetune/ppo_20260701_115304/policy_best.pt` reaches about `0.433` command progress, or about `4.33 rad` over 8 seconds, but still drifts about `17 cm`; this is real yaw but not yet a clean primitive.

7/1 CPG-modulator policy attempt

- Added a CPG-modulator training/view script.
  `train_cpg_modulator.py` trains a single command-conditioned controller where forward/back are deterministic CPG carriers and yaw is learned as gated CPG parameter/action modulation.

- Exported the 300k left-yaw PPO teacher.
  `runs/policy_table_yaw_left_planar_300k_8s/yaw-left_policy_table.json` replays about `4.62 rad` of real left yaw over 8 seconds.

- Found the previous backward default was stale.
  `runs/gait_search_backward_heading_refine` now moves the wrong way under the current model; `runs/gait_search_backward_rough` still moves backward about `0.90 m` over 8 seconds and is now the modulator script default.

- CPG-param NN compression is not good enough yet.
  Runs `cpg_param_v1` through `v4` reduce action clone error, but the learned approximation loses too much yaw authority; v4 only reaches about `2.31 rad` left yaw versus the exact teacher's `4.62 rad`.

- Exact component-controller review mode works for primitives, not arcs.
  `train_cpg_modulator.py --teacher-controller` runs exact forward/back CPG plus exact yaw tables; idle, forward, full backward, and pure yaw are reviewable, but naive mixed forward+yaw blending still terminates or underperforms.

7/1 residual PPO over component controller

- Added residual PPO trainer.
  `train_residual_ppo.py` freezes the exact component controller as the base action and trains a small PPO policy that outputs only a gated residual; the residual gate is zero for pure forward/back/yaw so solved primitives stay unchanged.

- Made checkpoint scoring arc-focused.
  Eval now reports `arc_score`, `arc_cmd`, and weighted `score`; this avoids hiding bad mixed-command behavior behind good primitive scores.

- Smoke and checkpoint reload pass.
  `runs/residual_ppo_smoke/residual_ppo_20260701_140757/residual_policy_best.pt` reloads and evaluates correctly, proving the training/save/view path works.

- Short 4k pilot shows the right direction but is not solved.
  `runs/residual_ppo_pilot/residual_ppo_20260701_140851/residual_policy_best.pt` improved arc score from about `-0.068` to `-0.026` at the best checkpoint while preserving primitives, but arcs are still not visually final.

- Next larger run command.
  `.venv-rl/bin/python -u train_residual_ppo.py --total-steps 50000 --num-envs 4 --rollout-steps 256 --eval-interval 10 --eval-episodes 1 --output-dir runs/residual_ppo_arc --torch-threads 2 --snapshot-interval 10`

7/1 residual PPO carrier fix

- Naive mixed base was the main arc crash.
  The original residual PPO used the naive component blend for arcs, so `arc_fl` hit the body-height termination at step 24; diagnostics showed `below_body_limit=True` with height around `0.034 m`.

- Switched mixed commands to a safe CPG carrier.
  `train_residual_ppo.py` now uses pure forward/back CPG as the base for mixed forward+yaw commands and leaves pure yaw on the exact yaw tables; zero-residual mixed commands now survive all four arc evals for 400 steps.

- Added focused arc sampling and arc-score checkpointing.
  Training now oversamples the exact four eval arcs, weights `arc_fl` highest, and selects best checkpoints by `arc_score` instead of primitive-dominated aggregate score.

- Best current residual checkpoint.
  `runs/residual_ppo_arc_command/residual_ppo_20260701_141701/residual_policy_best.pt` reaches `arc_score=0.0108`, `arc_cmd=0.0343`, and all arcs survive 400 steps; this is a real improvement over crashy arcs but still weak joystick obedience.

- Longer resume did not help.
  `runs/residual_ppo_arc_command_resume/residual_ppo_20260701_141847` degraded and reintroduced some arc termination, so do not promote it over the shorter command-focused best checkpoint.

7/1 residual scaled-arc controller

- Scalar yaw-table residuals are the best mixed-command baseline so far.
  A safe forward/back CPG carrier plus command-specific scaled yaw-table residuals keeps all four arc commands alive for 400 steps and reaches `arc_score=0.1598`, with scales `fl=-0.20`, `fr=-0.50`, `bl=-0.40`, `br=-0.40`.

- Yaw action must be an explicit policy feature.
  BC without the yaw-table action could not clone the scaled residual timing; adding `yaw_action` to the residual policy observation made supervised cloning work.

- Best learned residual checkpoint is the yaw-feature BC model.
  `runs/residual_bc_scaled_arc_yawfeat/residual_ppo_20260701_142533/residual_policy_best.pt` reaches `arc_score=0.1391`, `arc_cmd=0.1619`, and preserves the solved primitive commands.

- PPO fine-tuning did not improve the BC model.
  `runs/residual_ppo_arc_yawfeat_finetune/residual_ppo_20260701_142606/residual_policy_best.pt` tops out at `arc_score=0.1324`, so keep the BC checkpoint as the current learned best and the scaled controller as the current teacher.

7/1 improved arc teacher and DAgger distillation

- Added arc-controller JSON support.
  `train_residual_ppo.py` can now load `--arc-controller` files with per-quadrant `base_scale`, three servo-specific `yaw_scales`, and `step_offset`; primitives still use the exact forward/back CPG and yaw tables.

- Added `search_arc_controller.py`.
  The search optimizes the small mixed-command teacher directly instead of asking PPO to rediscover arcs; it keeps the controller ESP32-friendly because the policy surface is still tiny.

- Improved the deterministic mixed-command teacher.
  `runs/arc_controller_search/arc_controller_20260701_143909/best_arc_controller.json` reaches `arc_score=0.2906`, `arc_cmd=0.2996`, up from the scalar teacher's `arc_score=0.1598`.

- Plain BC was not enough for the richer teacher.
  A 96-hidden clone reached only `arc_score=0.0765`, and a longer 160-hidden clone reached `arc_score=0.1341`; low supervised loss did not prevent closed-loop rollout drift.

- Added explicit command features and DAgger-style residual distillation.
  The residual policy input now includes the command, and pretraining can collect learner-visited states labeled by the deterministic teacher via `--residual-dagger-rounds`.

- Best current learned residual policy.
  `runs/residual_dagger_arc_controller_weighted/residual_ppo_20260701_144648/residual_policy_best.pt` reaches `arc_score=0.2736`, `arc_cmd=0.2834`, close to the improved teacher while preserving the solved primitive commands.

7/1 broad joystick-grid eval and training

- Added broad mixed-command eval coverage.
  `train_residual_ppo.py --eval-suite broad` now evaluates a 4x4 magnitude grid over all forward/yaw sign quadrants, and `--view-command broad` can cycle the same commands visually.

- The corner-focused policy does not generalize enough.
  The previous best learned residual scores `arc_score=0.2736` on the core four arcs but only `arc_score=0.0638` on the broad joystick grid.

- Added broad quadrant search.
  `search_arc_controller.py --command broad_all` optimizes each quadrant's tiny arc-controller parameters over all mixed magnitudes instead of only the `0.7/0.7` corner.

- Best current broad deterministic teacher.
  `runs/arc_controller_search/arc_controller_20260701_145334/best_arc_controller.json` improves broad teacher score from `arc_score=0.1040` to `arc_score=0.1503`, but its core four-arc score is lower than the corner-focused teacher.

- Best current broad learned policy.
  `runs/residual_dagger_broad_arc_controller/residual_ppo_20260701_151038/residual_policy_best.pt` reaches broad `arc_score=0.1279`, up from the previous learned broad `0.0638`; its core four-arc score is `0.1901`, so keep the earlier `144648` checkpoint for sharp corner-arc demos.

- Dataset cap cleanup.
  `--residual-dataset-max-samples` now caps the initial BC dataset as well as DAgger-augmented datasets, which keeps broad-grid clone runs from silently collecting oversized supervised buffers.

7/1 magnitude-aware grid controller

- Added magnitude-aware arc-controller grid support.
  Arc-controller JSON entries can now include per-quadrant `grid` params keyed like `f0p70_y0p90`; `train_residual_ppo.py` interpolates grid params for arbitrary joystick magnitudes.

- Added grid-point search mode.
  `search_arc_controller.py --command grid_all` optimizes individual mixed-command grid points while preserving the older quadrant-level defaults for interpolation fallback.

- Best current deterministic joystick teacher.
  `runs/arc_controller_grid/arc_controller_20260701_151532/best_arc_controller.json` reaches broad `arc_score=0.2110`, up from the broad quadrant teacher's `0.1503`; it also reaches core `arc_score=0.3634`, beating the earlier corner-focused deterministic teacher.

- Learned grid clone improved but lost too much authority.
  `runs/residual_dagger_grid_arc_controller/residual_ppo_20260701_152801/residual_policy_best.pt` reaches broad `arc_score=0.1512`, better than the previous broad learned `0.1279`, but core `arc_score=0.1704`; for visual review and near-term control, prefer the deterministic grid controller.

- Current interpretation.
  The best ESP32-friendly path is now command -> magnitude-grid interpolation -> CPG/yaw-table function -> servo output; a learned residual clone is optional polish, not the best controller yet.

7/1 standalone controller export

- Added standalone deterministic runtime.
  `bram_controller.py` computes normalized servo commands from `(forward, yaw, step, heading_error, yaw_rate)` using only JSON data and NumPy; it does not require MuJoCo.

- Verified runtime parity.
  Sampled mixed, pure yaw, pure forward, and idle commands match the `train_residual_ppo.py` deterministic grid-controller path with `max_diff=0.0`.

- Added combined export artifact.
  `exports/bram_grid_controller_export.json` bundles forward/back gait params, yaw tables, grid arc-controller params, dt, scaling, and command metadata into one file for later ESP32/C++ translation.

- Current deployable control shape.
  Runtime loop is `read command + IMU -> BramGridController.action(...) -> map [-1, 1] to servo pulse/range`; heading correction is only active for pure forward/back at the moment.

7/1 right-yaw retraining

- Retrained right yaw under the planar-yaw PPO reward.
  `runs/ppo_yaw_right_planar_300k/ppo_20260701_154021/policy_best.pt` was trained with the same 300k-step settings as the good left-yaw run, but the raw exported table overspun at about `15.15 rad` over 8 seconds.

- Chose a scaled right-yaw table for review.
  `runs/policy_table_yaw_right_planar_300k_8s_scaled_0p4/yaw-right_policy_table.json` gives about `7.04 rad` over 8 seconds with about `3.4 cm` planar drift, much cleaner than the old right table.

- Retuned right-side mixed arcs for the new table.
  `runs/arc_controller_grid_right_yaw_0p40/arc_controller_20260701_155840/best_arc_controller.json` updates the `arc_fr` and `arc_br` grid points; core right arcs improved, but broad joystick score is still below the older all-grid controller.

- Updated defaults.
  `bram_controller.py`, `train_cpg_modulator.py`, and `train_hybrid_bc.py` now point right yaw at the scaled planar table; `bram_controller.py` also points at the right-yaw-retuned arc grid.
