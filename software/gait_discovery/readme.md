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

The older CAD-mesh attempt is saved as `bram_cad_attempt.xml`. The active
`bram.xml` is a measured hand-written model with a 7.5 inch equilateral
triangle chassis and primitive servo/leg bodies for simulation and RL.

On macOS, MuJoCo's viewer should be launched with:

```bash
.venv/bin/mjpython main.py
```
