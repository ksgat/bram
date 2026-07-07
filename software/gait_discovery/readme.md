3 legged robo

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

PufferLib currently depends on Torch. If Torch has a wheel for your `python3`,
install the RL stack with:

```bash
python -m pip install -e ".[rl]"
```

For the working RL setup in this directory, use the Python 3.12 venv:

```bash
source .venv-rl/bin/activate
python puffer_smoke.py
```

## Run

Smoke-test the MuJoCo model without opening a window:

```bash
python main.py --headless --seconds 3
```

Open the interactive viewer frozen so you can inspect and tweak the model:

```bash
.venv/bin/mjpython main.py --seconds 0 --pause
```

Run the interactive viewer with a simple sinusoidal starter gait:

```bash
.venv/bin/mjpython main.py
```

Check the Gymnasium environment scaffold that PufferLib can wrap later:

```bash
python main.py --env-check --seconds 3
```

## Movement V2 Training

The current control direction is under `software/movement_v2`: separate
forward/back RL specialist primitives plus a yaw-only command-conditioned RL
generalist. Do not train the old full joystick PPO policy for the base demo.

Train each primitive from the repo root:

```bash
.venv-rl/bin/python software/movement_v2/train_primitive_ppo.py \
  --primitive forward \
  --total-steps 200000 \
  --num-envs 4

.venv-rl/bin/python software/movement_v2/train_primitive_ppo.py \
  --primitive backward \
  --total-steps 200000 \
  --num-envs 4

.venv-rl/bin/python software/movement_v2/train_primitive_ppo.py \
  --primitive yaw \
  --total-steps 200000 \
  --num-envs 4
```

Smoke-test the current primitive controller from the repo root:

```bash
python3 software/movement_v2/smoke_primitives.py --seconds 4
```

The older CAD-mesh attempt is saved as `bram_cad_attempt.xml`. The active
`bram.xml` is a measured hand-written model with a 7.5 inch equilateral
triangle chassis and primitive servo/leg bodies for simulation and RL.

On macOS, MuJoCo's viewer should be launched with:

```bash
.venv/bin/mjpython main.py
```
