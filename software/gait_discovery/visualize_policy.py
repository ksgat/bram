from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from bram_env import BramTripodEnv, ENV_COMMAND_MODE
from train_ppo import ActorCritic


COMMAND_SUITE = (
    ("idle", 0.0, 0.0),
    ("fwd1", 1.0, 0.0),
    ("back1", -1.0, 0.0),
    ("fwd05", 0.5, 0.0),
    ("back05", -0.5, 0.0),
    ("yaw_l1", 0.0, 1.0),
    ("yaw_r1", 0.0, -1.0),
    ("yaw_l05", 0.0, 0.5),
    ("yaw_r05", 0.0, -0.5),
    ("arc_fl", 0.7, 0.7),
    ("arc_fr", 0.7, -0.7),
    ("arc_bl", -0.7, 0.7),
    ("arc_br", -0.7, -0.7),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a trained Bram PPO policy.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--forward-command", type=float, default=1.0)
    parser.add_argument("--yaw-rate-command", type=float, default=0.0)
    parser.add_argument("--random-command", action="store_true")
    parser.add_argument(
        "--command-suite",
        action="store_true",
        help="Cycle through idle, forward/back, yaw, and arc commands.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    maybe_relaunch_with_mjpython(args)
    if args.command_suite:
        args.episodes = max(args.episodes, len(COMMAND_SUITE))
    checkpoint = args.checkpoint or latest_checkpoint()
    env = BramTripodEnv(
        randomize_reset=False,
        randomize_command=args.random_command and not args.command_suite,
        command_forward=args.forward_command,
        command_yaw_rate=args.yaw_rate_command,
    )
    agent = load_agent(checkpoint, env)
    print(f"checkpoint={checkpoint}")

    if args.headless:
        stats = run_headless(env, agent, args)
        print(
            f"episodes={args.episodes} "
            f"mean_reward={stats['reward']:.3f} "
            f"mean_command_progress={stats['distance']:.4f} "
            f"mean_length={stats['length']:.1f}"
        )
        return

    run_viewer(env, agent, args)


def maybe_relaunch_with_mjpython(args: argparse.Namespace) -> None:
    if args.headless or platform.system() != "Darwin" or os.environ.get("MJPYTHON_BIN"):
        return

    mjpython = Path(sys.executable).with_name("mjpython")
    if not mjpython.exists():
        raise RuntimeError(
            "MuJoCo viewer requires mjpython on macOS. Run this command with "
            f"`mjpython {' '.join(sys.argv)}`."
        )

    print(f"macOS MuJoCo viewer requires mjpython; relaunching with {mjpython}")
    os.execv(str(mjpython), [str(mjpython), *sys.argv])


def latest_checkpoint() -> Path:
    checkpoints = sorted(
        Path("runs").glob("ppo_*/policy_best.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    if not checkpoints:
        raise FileNotFoundError("No policy_best.pt found under runs/ppo_*/")
    return checkpoints[-1]


def load_agent(checkpoint: Path, env: BramTripodEnv) -> ActorCritic:
    payload = torch.load(checkpoint, map_location="cpu")
    hidden_size = int(payload.get("args", {}).get("hidden_size", 64))
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    checkpoint_obs_dim, checkpoint_action_dim = checkpoint_dims(payload)
    if (checkpoint_obs_dim, checkpoint_action_dim) != (obs_dim, action_dim):
        raise ValueError(
            f"{checkpoint} was trained for obs_dim={checkpoint_obs_dim}, "
            f"action_dim={checkpoint_action_dim}, but the current env is "
            f"obs_dim={obs_dim}, action_dim={action_dim}. Retrain after the "
            "environment rewrite before visualizing this policy."
        )
    checkpoint_command_mode = payload.get("env_command_mode")
    if checkpoint_command_mode != ENV_COMMAND_MODE:
        raise ValueError(
            f"{checkpoint} was trained with env_command_mode={checkpoint_command_mode!r}, "
            f"but the current env uses {ENV_COMMAND_MODE!r}. Retrain before visualizing."
        )
    agent = ActorCritic(obs_dim, action_dim, hidden_size)
    agent.load_state_dict(payload["model_state_dict"])
    agent.eval()
    print(
        "checkpoint_eval "
        f"reward={payload.get('eval_reward', float('nan')):.3f} "
        f"distance={payload.get('eval_distance', float('nan')):.4f} "
        f"length={payload.get('eval_length', float('nan')):.1f}"
    )
    return agent


def checkpoint_dims(payload: dict) -> tuple[int, int]:
    if "obs_dim" in payload and "action_dim" in payload:
        return int(payload["obs_dim"]), int(payload["action_dim"])

    state_dict = payload["model_state_dict"]
    actor_input = state_dict["actor.0.weight"]
    actor_output = state_dict["actor.4.weight"]
    return int(actor_input.shape[1]), int(actor_output.shape[0])


def run_headless(
    env: BramTripodEnv,
    agent: ActorCritic,
    args: argparse.Namespace,
) -> dict[str, float]:
    rewards = []
    distances = []
    lengths = []
    for episode in range(args.episodes):
        command_name, command = command_for_episode(args, episode)
        _, total_reward, final_info, length = rollout_episode(
            env,
            agent,
            args,
            args.seed + episode,
            command,
        )
        print(
            f"episode={episode + 1} "
            f"command={command_name} "
            f"reward={total_reward:.3f} "
            f"command_progress={distance_from_info(final_info):.4f} "
            f"forward_cmd={command['forward_command']:.2f} "
            f"yaw_cmd={command['yaw_rate_command']:.2f} "
            f"length={length}"
        )
        rewards.append(total_reward)
        distances.append(distance_from_info(final_info))
        lengths.append(length)
    return {
        "reward": float(np.mean(rewards)),
        "distance": float(np.mean(distances)),
        "length": float(np.mean(lengths)),
    }


def run_viewer(env: BramTripodEnv, agent: ActorCritic, args: argparse.Namespace) -> None:
    import mujoco.viewer

    episode = 1
    command_name, command = command_for_episode(args, episode - 1)
    obs, _ = env.reset(seed=args.seed, options=command)
    total_reward = 0.0
    started = time.monotonic()
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        configure_viewer_visuals(viewer)
        while viewer.is_running() and episode <= args.episodes:
            step_started = time.monotonic()
            action = policy_action(agent, obs, args.stochastic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            viewer.sync()

            if terminated or truncated:
                print(
                    f"episode={episode} "
                    f"command={command_name} "
                    f"reward={total_reward:.3f} "
                    f"command_progress={distance_from_info(info):.4f} "
                    f"forward_cmd={command['forward_command']:.2f} "
                    f"yaw_cmd={command['yaw_rate_command']:.2f} "
                    f"length={env.steps}"
                )
                episode += 1
                total_reward = 0.0
                if episode <= args.episodes:
                    command_name, command = command_for_episode(args, episode - 1)
                    obs, _ = env.reset(seed=args.seed + episode, options=command)
                    viewer.sync()
                    time.sleep(0.5)

            sleep_time = (env.dt / max(args.speed, 1e-6)) - (time.monotonic() - step_started)
            if sleep_time > 0:
                time.sleep(sleep_time)

    elapsed = time.monotonic() - started
    print(f"viewer_elapsed_sec={elapsed:.1f}")


def rollout_episode(
    env: BramTripodEnv,
    agent: ActorCritic,
    args: argparse.Namespace,
    seed: int,
    command: dict[str, float],
) -> tuple[np.ndarray, float, dict[str, float], int]:
    obs, _ = env.reset(seed=seed, options=command)
    total_reward = 0.0
    final_info = {"command_distance": 0.0}
    for length in range(1, env.max_steps + 1):
        action = policy_action(agent, obs, args.stochastic)
        obs, reward, terminated, truncated, final_info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return obs, total_reward, final_info, length


def command_for_episode(
    args: argparse.Namespace,
    episode: int,
) -> tuple[str, dict[str, float]]:
    if args.command_suite:
        name, forward, yaw_rate = COMMAND_SUITE[episode % len(COMMAND_SUITE)]
    else:
        name = "fixed"
        forward = float(np.clip(args.forward_command, -1.0, 1.0))
        yaw_rate = float(np.clip(args.yaw_rate_command, -1.0, 1.0))
    return name, {"forward_command": forward, "yaw_rate_command": yaw_rate}


def policy_action(agent: ActorCritic, obs: np.ndarray, stochastic: bool) -> np.ndarray:
    obs_tensor = torch.as_tensor(obs[None, :], dtype=torch.float32)
    with torch.no_grad():
        if stochastic:
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
        else:
            action = agent.deterministic_action(obs_tensor)
    return action.cpu().numpy()[0].astype(np.float32)


def distance_from_info(info: dict) -> float:
    return float(info.get("command_distance", info.get("x_distance", 0.0)))


def configure_viewer_visuals(viewer) -> None:
    for flag in [
        mujoco.mjtVisFlag.mjVIS_INERTIA,
        mujoco.mjtVisFlag.mjVIS_SCLINERTIA,
        mujoco.mjtVisFlag.mjVIS_CONTACTPOINT,
        mujoco.mjtVisFlag.mjVIS_CONTACTFORCE,
    ]:
        viewer.opt.flags[int(flag)] = 0


if __name__ == "__main__":
    main()
