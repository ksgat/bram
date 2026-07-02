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

