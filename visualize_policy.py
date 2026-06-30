from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from bram_env import BramTripodEnv
from train_ppo import ActorCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a trained Bram PPO policy.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--command-angle-deg", type=float, default=0.0)
    parser.add_argument("--random-command", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint or latest_checkpoint()
    command_angle = None if args.random_command else np.deg2rad(args.command_angle_deg)
    env = BramTripodEnv(
        randomize_reset=False,
        randomize_command=args.random_command,
        command_angle=command_angle,
    )
    agent = load_agent(checkpoint, env)
    print(f"checkpoint={checkpoint}")

    if args.headless:
        stats = run_headless(env, agent, args)
        print(
            f"episodes={args.episodes} "
            f"mean_reward={stats['reward']:.3f} "
            f"mean_command_distance={stats['distance']:.4f} "
            f"mean_length={stats['length']:.1f}"
        )
        return

    run_viewer(env, agent, args)


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
        _, total_reward, final_info, length = rollout_episode(env, agent, args, args.seed + episode)
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

    obs, _ = env.reset(seed=args.seed)
    episode = 1
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
                    f"reward={total_reward:.3f} "
                    f"command_distance={distance_from_info(info):.4f} "
                    f"command_angle_deg={np.rad2deg(info.get('command_angle', 0.0)):.1f} "
                    f"length={env.steps}"
                )
                episode += 1
                total_reward = 0.0
                if episode <= args.episodes:
                    obs, _ = env.reset(seed=args.seed + episode)
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
) -> tuple[np.ndarray, float, dict[str, float], int]:
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    final_info = {"command_distance": 0.0}
    for length in range(1, env.max_steps + 1):
        action = policy_action(agent, obs, args.stochastic)
        obs, reward, terminated, truncated, final_info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return obs, total_reward, final_info, length


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
