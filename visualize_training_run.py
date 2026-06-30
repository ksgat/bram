from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np

from bram_env import BramTripodEnv
from visualize_policy import (
    configure_viewer_visuals,
    distance_from_info,
    load_agent,
    policy_action,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay saved PPO checkpoints from one Bram training run."
    )
    parser.add_argument("--run", type=Path, default=None, help="Path like runs/ppo_...")
    parser.add_argument("--episodes-per-checkpoint", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--pause", type=float, default=0.4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-checkpoints", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--include-best-final", action="store_true")
    parser.add_argument("--forward-command", type=float, default=1.0)
    parser.add_argument("--yaw-rate-command", type=float, default=0.0)
    parser.add_argument(
        "--command-suite",
        action="store_true",
        help="Cycle forward, backward, yaw, and arc commands for each checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run or latest_run_dir()
    checkpoints = checkpoints_for_run(run_dir, args)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")

    env = BramTripodEnv(
        randomize_reset=False,
        randomize_command=False,
        command_forward=args.forward_command,
        command_yaw_rate=args.yaw_rate_command,
    )
    print(f"run={run_dir}")
    print(f"checkpoints={len(checkpoints)}")

    if args.headless:
        run_headless(env, checkpoints, args)
        return

    run_viewer(env, checkpoints, args)


def latest_run_dir() -> Path:
    runs = sorted(
        (path for path in Path("runs").glob("ppo_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    if not runs:
        raise FileNotFoundError("No runs/ppo_* directories found.")
    return runs[-1]


def checkpoints_for_run(run_dir: Path, args: argparse.Namespace) -> list[Path]:
    snapshots = sorted(
        (run_dir / "snapshots").glob("policy_update_*.pt"),
        key=checkpoint_sort_key,
    )
    checkpoints = snapshots
    if args.include_best_final or not checkpoints:
        checkpoints = checkpoints + [
            path
            for path in [run_dir / "policy_best.pt", run_dir / "policy.pt"]
            if path.exists()
        ]
    checkpoints = checkpoints[:: max(1, args.stride)]
    if args.max_checkpoints > 0:
        checkpoints = checkpoints[: args.max_checkpoints]
    return checkpoints


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"policy_update_(\d+)", path.name)
    if match:
        return int(match.group(1)), path.name
    if path.name == "policy_best.pt":
        return 10**9 - 1, path.name
    if path.name == "policy.pt":
        return 10**9, path.name
    return 10**9 + 1, path.name


def run_headless(
    env: BramTripodEnv,
    checkpoints: list[Path],
    args: argparse.Namespace,
) -> None:
    for index, checkpoint in enumerate(checkpoints, start=1):
        agent = load_agent(checkpoint, env)
        stats = evaluate_checkpoint(env, agent, args, index)
        print(
            f"{index:03d}/{len(checkpoints):03d} "
            f"{checkpoint.name} "
            f"reward={stats['reward']:.3f} "
            f"command_progress={stats['distance']:.4f} "
            f"length={stats['length']:.1f}"
        )


def run_viewer(
    env: BramTripodEnv,
    checkpoints: list[Path],
    args: argparse.Namespace,
) -> None:
    import mujoco.viewer

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        configure_viewer_visuals(viewer)
        for index, checkpoint in enumerate(checkpoints, start=1):
            if not viewer.is_running():
                break
            agent = load_agent(checkpoint, env)
            print(f"replay={index}/{len(checkpoints)} checkpoint={checkpoint}")
            for episode in range(args.episodes_per_checkpoint):
                if not viewer.is_running():
                    break
                command = command_for_episode(args, index, episode)
                obs, _ = env.reset(
                    seed=args.seed + index * 1000 + episode,
                    options=command,
                )
                total_reward = 0.0
                final_info = {"command_distance": 0.0}
                while viewer.is_running():
                    step_started = time.monotonic()
                    action = policy_action(agent, obs, args.stochastic)
                    obs, reward, terminated, truncated, final_info = env.step(action)
                    total_reward += reward
                    viewer.sync()
                    if terminated or truncated:
                        break
                    sleep_time = (env.dt / max(args.speed, 1e-6)) - (
                        time.monotonic() - step_started
                    )
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                print(
                    f"episode={episode + 1} "
                    f"reward={total_reward:.3f} "
                    f"command_progress={distance_from_info(final_info):.4f} "
                    f"forward_cmd={command['forward_command']:.2f} "
                    f"yaw_cmd={command['yaw_rate_command']:.2f} "
                    f"length={env.steps}"
                )
            if args.pause > 0:
                time.sleep(args.pause)


def evaluate_checkpoint(
    env: BramTripodEnv,
    agent,
    args: argparse.Namespace,
    checkpoint_index: int,
) -> dict[str, float]:
    rewards = []
    distances = []
    lengths = []
    for episode in range(args.episodes_per_checkpoint):
        command = command_for_episode(args, checkpoint_index, episode)
        obs, _ = env.reset(
            seed=args.seed + checkpoint_index * 1000 + episode,
            options=command,
        )
        total_reward = 0.0
        final_info = {"command_distance": 0.0}
        for length in range(1, env.max_steps + 1):
            action = policy_action(agent, obs, args.stochastic)
            obs, reward, terminated, truncated, final_info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
        distances.append(distance_from_info(final_info))
        lengths.append(length)
    return {
        "reward": float(np.mean(rewards)),
        "distance": float(np.mean(distances)),
        "length": float(np.mean(lengths)),
    }


def command_for_episode(
    args: argparse.Namespace,
    checkpoint_index: int,
    episode: int,
) -> dict[str, float]:
    if not args.command_suite:
        return {
            "forward_command": float(np.clip(args.forward_command, -1.0, 1.0)),
            "yaw_rate_command": float(np.clip(args.yaw_rate_command, -1.0, 1.0)),
        }

    commands = [
        (0.0, 0.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.5, 0.0),
        (-0.5, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (0.0, 0.5),
        (0.0, -0.5),
        (0.7, 0.7),
        (0.7, -0.7),
        (-0.7, 0.7),
        (-0.7, -0.7),
    ]
    forward, yaw_rate = commands[(checkpoint_index + episode - 1) % len(commands)]
    return {"forward_command": forward, "yaw_rate_command": yaw_rate}


if __name__ == "__main__":
    main()
