# JOURNAL.md


## 6/19

Initial mockup!
<img width="572" height="371" alt="image" src="https://github.com/user-attachments/assets/5bea86bb-be6d-4429-af32-2960aed3dcdc" />

basically im gonna rl walking or somehting idk ima do some research bu im gonna cad first im gonna use like big servos and mag encoders and my pi 3 and probably a pico also but idk ill see type

in between this was cad and parts, tuff 👍

## 6/29-6/30

Set up the Python/MuJoCo gait-discovery project, built the first usable tripod model from measured geometry, added rough hardware constraints, and trained early PPO policies. Forward locomotion became the first solved behavior, while broad joystick control, yaw, and aggressive sim-to-real noise exposed reward hacking, arcing, stationary local optima, and the need for stricter world-frame command metrics.

## 6/30

Reworked the environment and training process around command obedience, per-command evals, gated curricula, distillation, and straight-line polish. Fixed forward/back/yaw primitives proved individually learnable, but one shared PPO actor was unstable for full joystick control, which pushed the project toward expert distillation, deterministic CPG search, yaw action tables, and hybrid composition rather than a single end-to-end learned policy.

## 7/1

Corrected yaw scoring to use real planar heading instead of body gyro wobble, retrained/exported learned yaw primitives, searched deterministic forward/back CPGs and mixed-arc blend grids, and settled on the current deployable architecture: deterministic forward/back CPGs plus learned left/right yaw tables plus a tuned grid blend for mixed commands. The standalone controller export now lives under the gait_discovery folder and is the source for the future ESP32 port.

7/1 THE EVENING!

Did some minor cad (recessed where the bolts go), added screw holes, to keep the robot together, as my 3d printer is broken and my friend whos printer works leaves on a trip tomorrow, so I had to get my friend to print the parts today.

On architecture: I went through a lot of pivots and eventually landed on a hybrid controller instead of a single end-to-end learned policy. In retrospect, forward and backward crawling made much more sense as deterministic gaits, which I found out empirically after fighting the general PPO setup for a while. The learned PPO-derived yaw behavior was stellar though, so I kept that part and built the final controller around combining deterministic translation with learned turning.

I also dropped the pizero as it was unneccesary the intial policy setup I had planned was already small enough to run on an esp32 like raw, and the one I have right now also is, and it just makes my life easier and I cant even get a pi zero they are like 60 dollars its crazy!
prints!
sim:
<img width="800" height="1200" alt="image" src="https://github.com/user-attachments/assets/f7719e8e-e1ab-4766-afab-3919750d9aa5" />
prints:
<img width="1172" height="1562" alt="image" src="https://github.com/user-attachments/assets/5aa46017-ec23-4df2-b49c-fdb74fbe0b16" />
<img width="1172" height="1562" alt="image" src="https://github.com/user-attachments/assets/367fb9c7-6547-4e72-9973-ba87fbf192cc" />

## 7/13 movement_v2 yaw polish

Okay so the old "just train one yaw generalist" thing was kinda cooked for what I actually need on hardware. I only really need base primitives I can bind to the ESP32, and the robot has an IMU, so exporting open-loop tables is not the move. The move is online tiny policies: previous 4 IMU quats + previous 6 servo actions + command/base features in, 3 servo targets/residual out, low-rate policy, servo smoothing outside the policy.

First important thing: the left yaw table export was lying to me. The searched left CPG params were good when held at the original 10 Hz timing, but the interpolated table replay changed the contact timing and drifted like trash. So I changed the residual yaw env so the base can be either a JSON table or the original searched CPG params directly. That mattered a lot.

Left yaw method:

- base: `yaw-left_base_params.json`, direct searched CPG, held at 10 Hz
- residual: small PPO policy around the base, `residual_limit=0.12`, no extra residual slew
- train command: only `yaw_pos1`, not all yaw commands
- run: `software/movement_v2/runs/yaw_left_residual_cpg/yaw_residual_20260713_185146`
- promoted checkpoint: `software/movement_v2/exports/final_yaw_20260713/yaw-left_residual_policy_best.pt`

Left got way cleaner:

- raw direct CPG: `2.297 rad`, `2.70 cm` final drift, `1.11 cm` mean, `3.58 cm` max
- residual best: `2.285 rad`, `0.77 cm` final drift, `1.09 cm` mean, `1.97 cm` max

Then I tried to get right yaw up to that standard. First instinct was "do what worked on left" and try right CPG params directly. I tested the old right search seeds through the same residual wrapper. Seed 41 had strong yaw and good final position (`3.13 rad`, `1.15 cm` final drift), but it wandered mid-rollout (`2.79 cm` mean, `4.99 cm` max), so raw CPG right was not actually the best teacher. Tight-gate residual around that CPG also wasted compute and kept jumping into 5-9 cm peak drift. Basically, do not throw a new CPG search at right yaw just because it worked for left.

The good move was the less glamorous one: take the already-good right residual checkpoint and fine-tune it as a right-only specialist. Starting point was the old promoted right residual:

- old right: `2.341 rad`, `1.63 cm` final drift, `1.40 cm` mean, `2.79 cm` max

Fine-tune method:

- init checkpoint: `yaw-right_residual_policy_best.pt`
- base: `yaw-right_residual_base_table.json`
- train command: only `yaw_neg1`
- `frame_skip=10` / 50 Hz residual env, because this right artifact was table/residual based already
- `residual_limit=0.18`
- `slew_limit=0.25`
- low LR: `5e-5`
- tighter training gate than the old accept gate: `2.0 cm` final, `1.2 cm` mean, `2.2 cm` max
- run: `software/movement_v2/runs/yaw_right_residual_finetune_tight/yaw_residual_20260713_193828`
- promoted checkpoint: `software/movement_v2/exports/final_yaw_20260713/yaw-right_residual_policy_finetuned_best.pt`

That hit the money checkpoint at about 10k steps:

- new right: `2.385 rad`, `0.53 cm` final drift, `1.07 cm` mean, `2.01 cm` max

Compared to left:

- left residual: `2.285 rad`, `0.77 cm` final, `1.09 cm` mean, `1.97 cm` max
- right residual: `2.385 rad`, `0.53 cm` final, `1.07 cm` mean, `2.01 cm` max

So right now beats left on yaw amount, final drift, and mean drift. Left only barely wins max drift by like `0.04 cm`, which is basically tiny but I wrote it down because the metric says it. I also tried a peak-polish run from the new right best with an even tighter max target, but it did not beat the saved checkpoint, so I did not promote it. Good reminder that "stricter" is not automatically better; once the policy is clean, PPO can just trade one metric for another.

The final yaw package is now:

- `software/movement_v2/exports/final_yaw_20260713/manifest.json`
- left full yaw: `yaw-left_residual_policy_best.pt` on direct CPG params
- right full yaw: `yaw-right_residual_policy_finetuned_best.pt` on the right residual base table

Verified with:

```bash
.venv-rl/bin/python software/movement_v2/visualize_final_yaw.py --headless --case suite --seconds 8
```

The actual lesson is kinda important: left wanted CPG-param residual because table export broke timing, but right wanted checkpoint polish because the existing table residual already encoded a good contact pattern. Same residual idea, different best base. This is probably the ontology going forward: primitive-specific base source first, then narrow residual PPO around the best reproducible base, not broad generalist PPO unless there is no other option.
