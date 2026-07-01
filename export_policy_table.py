from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bram_env import BramTripodEnv, ENV_COMMAND_MODE
from search_gait import command_for_primitive
from train_ppo import ActorCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export or replay a deterministic servo action table from a PPO policy."
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--table", type=Path, default=None)
    parser.add_argument(
        "--primitive",
        choices=("forward", "backward", "yaw-left", "yaw-right", "idle"),
        default="yaw-right",
    )
    parser.add_argument("--episode-seconds", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--allow-mode-mismatch", action="store_true")
    return parser.parse_args()


def maybe_relaunch_with_mjpython(args: argparse.Namespace) -> None:
    if not args.view or args.headless or platform.system() != "Darwin":
        return
    if Path(sys.executable).name == "mjpython" or os.environ.get("MJPYTHON_BIN"):
        return
    mjpython = Path(sys.executable).with_name("mjpython")
    if mjpython.exists():
        os.execv(str(mjpython), [str(mjpython), *sys.argv])


def load_agent(
    checkpoint: Path,
    env: BramTripodEnv,
    allow_mode_mismatch: bool,
) -> ActorCritic:
    payload = torch.load(checkpoint, map_location="cpu")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    checkpoint_obs_dim = int(payload.get("obs_dim", -1))
    checkpoint_action_dim = int(payload.get("action_dim", -1))
    if (checkpoint_obs_dim, checkpoint_action_dim) != (obs_dim, action_dim):
        raise ValueError(
            f"{checkpoint} has obs/action dims "
            f"{checkpoint_obs_dim}/{checkpoint_action_dim}; current env has "
            f"{obs_dim}/{action_dim}."
        )
    checkpoint_mode = payload.get("env_command_mode")
    if checkpoint_mode != ENV_COMMAND_MODE and not allow_mode_mismatch:
        raise ValueError(
            f"{checkpoint} has env_command_mode={checkpoint_mode!r}; "
            f"current env uses {ENV_COMMAND_MODE!r}."
        )
    hidden_size = int(payload.get("args", {}).get("hidden_size", 64))
    agent = ActorCritic(obs_dim, action_dim, hidden_size)
    agent.load_state_dict(payload["model_state_dict"])
    agent.eval()
    return agent


def policy_action(agent: ActorCritic, obs: np.ndarray) -> np.ndarray:
    obs_tensor = torch.as_tensor(obs[None, :], dtype=torch.float32)
    with torch.no_grad():
        action = agent.deterministic_action(obs_tensor)
    return action.cpu().numpy()[0].astype(np.float32)


def make_env(primitive: str, episode_seconds: float) -> BramTripodEnv:
    forward_command, yaw_command = command_for_primitive(primitive)
    return BramTripodEnv(
        frame_skip=10,
        episode_seconds=episode_seconds,
        randomize_reset=False,
        domain_randomization=False,
        randomize_command=False,
        command_forward=forward_command,
        command_yaw_rate=yaw_command,
    )


def command_options(primitive: str) -> dict[str, float]:
    forward_command, yaw_command = command_for_primitive(primitive)
    return {"forward_command": forward_command, "yaw_rate_command": yaw_command}


def export_table(args: argparse.Namespace) -> Path:
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required when exporting a table.")
    out_dir = args.out_dir or Path("runs") / (
        "policy_table_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args.primitive, args.episode_seconds)
    agent = load_agent(args.checkpoint, env, args.allow_mode_mismatch)
    obs, _ = env.reset(seed=args.seed, options=command_options(args.primitive))
    actions: list[list[float]] = []
    infos: list[dict[str, float]] = []
    for _ in range(env.max_steps):
        action = policy_action(agent, obs)
        actions.append([float(value) for value in action])
        obs, _, terminated, truncated, info = env.step(action)
        infos.append(scalar_info(info))
        if terminated or truncated:
            break
    payload = {
        "primitive": args.primitive,
        "checkpoint": str(args.checkpoint),
        "dt": env.dt,
        "control_hz": 1.0 / env.dt,
        "episode_seconds": args.episode_seconds,
        "seed": args.seed,
        "command": command_options(args.primitive),
        "actions": actions,
        "final_info": infos[-1] if infos else {},
    }
    path = out_dir / f"{args.primitive}_policy_table.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    env.close()
    print(f"table={path}")
    return path


def scalar_info(info: dict[str, Any]) -> dict[str, float]:
    keep = (
        "command_distance",
        "line_distance",
        "yaw_distance",
        "x_distance",
        "y_distance",
        "cross_track_error",
        "heading_error",
        "yaw_rate",
        "planar_speed",
    )
    return {key: float(info.get(key, 0.0)) for key in keep}


def load_table(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if "actions" not in payload:
        raise ValueError(f"{path} does not contain an actions table.")
    return payload


def table_action(table: dict[str, Any], step: int) -> np.ndarray:
    actions = table["actions"]
    return np.asarray(actions[step % len(actions)], dtype=np.float32)


def replay_table(path: Path, args: argparse.Namespace) -> None:
    table = load_table(path)
    primitive = str(table["primitive"])
    env = make_env(primitive, args.episode_seconds)
    if args.view and not args.headless:
        run_viewer(env, table, args)
    else:
        run_headless(env, table, args)
    env.close()


def run_headless(env: BramTripodEnv, table: dict[str, Any], args: argparse.Namespace) -> None:
    results = []
    for episode in range(args.episodes):
        env.reset(seed=args.seed + episode, options=command_options(table["primitive"]))
        final_info: dict[str, Any] = {}
        terminated = False
        truncated = False
        for step in range(env.max_steps):
            _, _, terminated, truncated, final_info = env.step(table_action(table, step))
            if terminated or truncated:
                break
        result = scalar_info(final_info)
        result["length"] = float(step + 1)
        result["terminated"] = float(terminated)
        results.append(result)
        print(
            f"episode={episode + 1} "
            f"primitive={table['primitive']} "
            f"command_progress={result['command_distance']:.4f} "
            f"yaw_distance={result['yaw_distance']:.4f} "
            f"planar=({result['x_distance']:.4f},{result['y_distance']:.4f}) "
            f"length={step + 1} term={terminated}"
        )
    keys = results[0].keys()
    mean = {key: float(np.mean([result[key] for result in results])) for key in keys}
    print(json.dumps(mean, indent=2))


def run_viewer(env: BramTripodEnv, table: dict[str, Any], args: argparse.Namespace) -> None:
    import mujoco.viewer

    episode = 1
    step = 0
    env.reset(seed=args.seed, options=command_options(table["primitive"]))
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running() and episode <= args.episodes:
            start = time.perf_counter()
            _, _, terminated, truncated, _ = env.step(table_action(table, step))
            viewer.sync()
            step += 1
            if terminated or truncated:
                episode += 1
                step = 0
                if episode <= args.episodes:
                    env.reset(
                        seed=args.seed + episode,
                        options=command_options(table["primitive"]),
                    )
                    time.sleep(0.4)
            sleep_time = (env.dt / max(args.speed, 1e-6)) - (time.perf_counter() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)


def main() -> None:
    args = parse_args()
    maybe_relaunch_with_mjpython(args)
    table_path = args.table
    if table_path is None:
        table_path = export_table(args)
    if args.view or args.headless:
        replay_table(table_path, args)


if __name__ == "__main__":
    main()
