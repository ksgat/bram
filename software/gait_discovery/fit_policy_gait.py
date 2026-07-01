from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from bram_env import BramTripodEnv, ENV_COMMAND_MODE
from search_gait import (
    PARAM_HIGH,
    PARAM_LOW,
    PARAM_NAMES,
    clip_params,
    command_for_primitive,
    evaluate_candidate,
    gait_action,
    initial_mean,
    initial_std,
    save_params,
)
from train_ppo import ActorCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a smooth deterministic gait to a PPO policy action trace."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--primitive",
        choices=("forward", "backward", "yaw-left", "yaw-right", "idle"),
        required=True,
    )
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--episode-seconds", type=float, default=4.0)
    parser.add_argument("--fit-iterations", type=int, default=30)
    parser.add_argument("--population", type=int, default=384)
    parser.add_argument("--elite-frac", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--allow-mode-mismatch", action="store_true")
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--domain-randomization-strength", type=float, default=0.25)
    parser.add_argument("--randomize-reset", action="store_true")
    return parser.parse_args()


def load_agent(checkpoint: Path, env: BramTripodEnv, allow_mode_mismatch: bool) -> ActorCritic:
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
            f"current env uses {ENV_COMMAND_MODE!r}. Pass --allow-mode-mismatch "
            "only if you know the obs/action layout is compatible."
        )
    hidden_size = int(payload.get("args", {}).get("hidden_size", 64))
    agent = ActorCritic(obs_dim, action_dim, hidden_size)
    agent.load_state_dict(payload["model_state_dict"])
    agent.eval()
    return agent


def deterministic_action(agent: ActorCritic, obs: np.ndarray) -> np.ndarray:
    obs_tensor = torch.as_tensor(obs[None, :], dtype=torch.float32)
    with torch.no_grad():
        action = agent.deterministic_action(obs_tensor)
    return action.cpu().numpy()[0].astype(np.float32)


def collect_policy_trace(
    agent: ActorCritic,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    forward_command, yaw_command = command_for_primitive(args.primitive)
    env = BramTripodEnv(
        frame_skip=10,
        episode_seconds=args.episode_seconds,
        randomize_reset=args.randomize_reset,
        domain_randomization=args.domain_randomization,
        domain_randomization_strength=args.domain_randomization_strength,
        randomize_command=False,
        command_forward=forward_command,
        command_yaw_rate=yaw_command,
    )
    times: list[float] = []
    actions: list[np.ndarray] = []
    for episode in range(args.episodes):
        obs, _ = env.reset(
            seed=args.seed + episode,
            options={
                "forward_command": forward_command,
                "yaw_rate_command": yaw_command,
                "randomize": args.randomize_reset,
            },
        )
        for step in range(env.max_steps):
            action = deterministic_action(agent, obs)
            times.append(step * env.dt)
            actions.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    env.close()
    return np.asarray(times, dtype=np.float64), np.asarray(actions, dtype=np.float64)


def batch_gait_actions(params: np.ndarray, times: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            gait_action(params, float(t), use_heading_correction=False)
            for t in times
        ],
        axis=0,
    ).astype(np.float64)


def fit_score(params: np.ndarray, times: np.ndarray, target_actions: np.ndarray) -> float:
    pred = batch_gait_actions(params, times)
    action_mse = float(np.mean(np.square(pred - target_actions)))
    if len(pred) > 1:
        delta_mse = float(
            np.mean(np.square(np.diff(pred, axis=0) - np.diff(target_actions, axis=0)))
        )
    else:
        delta_mse = 0.0
    saturation_cost = float(np.mean(np.maximum(0.0, np.abs(pred) - 0.96)))
    return -(action_mse + 0.25 * delta_mse + 0.04 * saturation_cost)


def trace_seed(target_actions: np.ndarray) -> np.ndarray:
    mean = initial_mean()
    centers = np.mean(target_actions, axis=0)
    amplitudes = np.sqrt(2.0) * np.std(target_actions, axis=0)
    mean[0] = 1.40
    mean[1:4] = np.clip(centers, PARAM_LOW[1:4], PARAM_HIGH[1:4])
    mean[4:7] = np.clip(amplitudes, PARAM_LOW[4:7], PARAM_HIGH[4:7])
    mean[10:13] = 0.0
    mean[13:18] = 0.0
    mean[18:21] = 0.0
    return clip_params(mean)


def fit_params(
    times: np.ndarray,
    target_actions: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(args.seed)
    mean = trace_seed(target_actions)
    std = initial_std()
    std[0] = 0.35
    std[1:4] = 0.20
    std[4:7] = 0.24
    std[7:10] = 1.35
    std[10:13] = 0.22
    std[13:18] = 0.0
    std[18:21] = 1.20
    min_std = (PARAM_HIGH - PARAM_LOW) * 0.01
    min_std[13:18] = 0.0
    elite_count = max(2, int(round(args.population * args.elite_frac)))
    best_params = mean.copy()
    best_score = fit_score(best_params, times, target_actions)

    for iteration in range(args.fit_iterations):
        candidates = []
        for candidate_index in range(args.population):
            if iteration == 0 and candidate_index == 0:
                params = mean.copy()
            else:
                params = clip_params(rng.normal(mean, std))
            score = fit_score(params, times, target_actions)
            candidates.append((score, params))
        candidates.sort(key=lambda item: item[0], reverse=True)
        elites = np.stack([params for _, params in candidates[:elite_count]])
        mean = clip_params(np.mean(elites, axis=0))
        std = np.maximum(np.std(elites, axis=0), min_std)
        std[13:18] = 0.0
        if candidates[0][0] > best_score:
            best_score = float(candidates[0][0])
            best_params = candidates[0][1].copy()
        print(
            f"iter={iteration + 1:03d}/{args.fit_iterations:03d} "
            f"best_clone_loss={-best_score:.6f} "
            f"iter_clone_loss={-candidates[0][0]:.6f}"
        )
    return best_params, best_score


def make_eval_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        primitive=args.primitive,
        episodes=max(1, args.episodes),
        episode_seconds=args.episode_seconds,
        domain_randomization=args.domain_randomization,
        domain_randomization_strength=args.domain_randomization_strength,
        randomize_reset=args.randomize_reset,
    )


def write_fit_metrics(
    path: Path,
    times: np.ndarray,
    target_actions: np.ndarray,
    params: np.ndarray,
) -> None:
    pred = batch_gait_actions(params, times)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "t",
                "target_front",
                "target_back_left",
                "target_back_right",
                "fit_front",
                "fit_back_left",
                "fit_back_right",
            ],
        )
        writer.writeheader()
        for t, target, fitted in zip(times, target_actions, pred, strict=True):
            writer.writerow(
                {
                    "t": float(t),
                    "target_front": float(target[0]),
                    "target_back_left": float(target[1]),
                    "target_back_right": float(target[2]),
                    "fit_front": float(fitted[0]),
                    "fit_back_left": float(fitted[1]),
                    "fit_back_right": float(fitted[2]),
                }
            )


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or Path("runs") / (
        "gait_fit_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    forward_command, yaw_command = command_for_primitive(args.primitive)
    env = BramTripodEnv(
        episode_seconds=args.episode_seconds,
        randomize_reset=False,
        randomize_command=False,
        command_forward=forward_command,
        command_yaw_rate=yaw_command,
    )
    agent = load_agent(args.checkpoint, env, args.allow_mode_mismatch)
    env.close()

    times, target_actions = collect_policy_trace(agent, args)
    params, clone_score = fit_params(times, target_actions, args)
    eval_result = evaluate_candidate(params, make_eval_args(args), args.seed + 1_000_000)
    save_params(out_dir / "best_params.json", params, make_eval_args(args), eval_result)
    write_fit_metrics(out_dir / "fit_trace.csv", times, target_actions, params)
    payload = {
        "checkpoint": str(args.checkpoint),
        "primitive": args.primitive,
        "clone_loss": -clone_score,
        "eval": asdict(eval_result),
        "best_params": str(out_dir / "best_params.json"),
        "fit_trace": str(out_dir / "fit_trace.csv"),
        "param_names": PARAM_NAMES,
    }
    (out_dir / "fit_summary.json").write_text(
        __import__("json").dumps(payload, indent=2) + "\n"
    )
    print(__import__("json").dumps(payload, indent=2))


if __name__ == "__main__":
    main()
