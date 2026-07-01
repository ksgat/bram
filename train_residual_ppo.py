from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from bram_env import BramTripodEnv, ENV_COMMAND_MODE
from train_cpg_modulator import (
    DEFAULT_BACKWARD_GAIT,
    DEFAULT_FORWARD_GAIT,
    DEFAULT_YAW_LEFT_TABLE,
    DEFAULT_YAW_RIGHT_TABLE,
    COMMAND_SUITE,
    TeacherLibrary,
    command_options,
)
from train_ppo import ActorCritic


EXACT_ARC_COMMANDS = (
    (0.7, 0.7),
    (0.7, -0.7),
    (-0.7, 0.7),
    (-0.7, -0.7),
)
MIXED_MAGNITUDES = (0.35, 0.50, 0.70, 0.90)
BROAD_ARC_COMMANDS = tuple(
    (forward_sign * forward_mag, yaw_sign * yaw_mag)
    for forward_mag in MIXED_MAGNITUDES
    for yaw_mag in MIXED_MAGNITUDES
    for forward_sign in (1.0, -1.0)
    for yaw_sign in (1.0, -1.0)
)
EXACT_ARC_WEIGHTS = np.array([0.35, 0.25, 0.20, 0.20], dtype=np.float64)
EXACT_ARC_WEIGHTS /= EXACT_ARC_WEIGHTS.sum()
ARC_COMMAND_NAMES = {
    (1, 1): "arc_fl",
    (1, -1): "arc_fr",
    (-1, 1): "arc_bl",
    (-1, -1): "arc_br",
}
BROAD_COMMAND_SUITE = COMMAND_SUITE + tuple(
    (
        (
            f"{ARC_COMMAND_NAMES[(1 if forward >= 0.0 else -1, 1 if yaw >= 0.0 else -1)]}"
            f"_{abs(forward):.2f}_{abs(yaw):.2f}"
        ).replace(".", "p"),
        forward,
        yaw,
    )
    for forward, yaw in BROAD_ARC_COMMANDS
    if (round(abs(forward), 2), round(abs(yaw), 2)) != (0.70, 0.70)
)


@dataclass
class ControllerState:
    step: int = 0
    heading_error: float = 0.0
    yaw_rate: float = 0.0


@dataclass(frozen=True)
class EvalResult:
    command: str
    reward: float
    score: float
    command_distance: float
    line_distance: float
    yaw_distance: float
    x_distance: float
    y_distance: float
    cross_track_error: float
    heading_error: float
    length: int
    terminated: bool


@dataclass(frozen=True)
class EvalStats:
    reward: float
    score: float
    arc_score: float
    arc_command_distance: float
    arc_length: float
    command_distance: float
    length: float
    per_command: tuple[EvalResult, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Residual PPO over Bram's exact CPG/table component controller."
    )
    parser.add_argument("--forward-gait", type=Path, default=DEFAULT_FORWARD_GAIT)
    parser.add_argument("--backward-gait", type=Path, default=DEFAULT_BACKWARD_GAIT)
    parser.add_argument("--yaw-left-table", type=Path, default=DEFAULT_YAW_LEFT_TABLE)
    parser.add_argument("--yaw-right-table", type=Path, default=DEFAULT_YAW_RIGHT_TABLE)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/residual_ppo"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Initialize a new residual PPO run from an existing residual checkpoint.",
    )
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.18)
    parser.add_argument("--entropy-coef", type=float, default=0.004)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.05)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--log-std-init", type=float, default=-1.6)
    parser.add_argument("--residual-limit", type=float, default=0.35)
    parser.add_argument("--arc-yaw-scale", type=float, default=0.65)
    parser.add_argument("--base-scaling", choices=("linear", "gait-speed"), default="gait-speed")
    parser.add_argument("--base-speed-min", type=float, default=0.35)
    parser.add_argument("--base-action-min", type=float, default=0.60)
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--domain-randomization-strength", type=float, default=0.35)
    parser.add_argument("--episode-seconds", type=float, default=8.0)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--eval-suite", choices=("core", "broad"), default="core")
    parser.add_argument("--view", action="store_true")
    parser.add_argument(
        "--view-command",
        choices=(
            "suite",
            "broad",
            "primitives",
        )
        + tuple(command[0] for command in BROAD_COMMAND_SUITE),
        default="suite",
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--snapshot-interval", type=int, default=0)
    parser.add_argument("--residual-pretrain-steps", type=int, default=0)
    parser.add_argument("--residual-pretrain-batch-size", type=int, default=256)
    parser.add_argument("--residual-pretrain-episodes", type=int, default=4)
    parser.add_argument("--residual-dagger-rounds", type=int, default=0)
    parser.add_argument("--residual-dagger-steps", type=int, default=700)
    parser.add_argument("--residual-dagger-episodes", type=int, default=2)
    parser.add_argument("--residual-dataset-max-samples", type=int, default=50_000)
    parser.add_argument("--residual-train-broad-mixed", action="store_true")
    parser.add_argument("--residual-repeat-arc-fl", type=int, default=1)
    parser.add_argument("--residual-repeat-arc-fr", type=int, default=1)
    parser.add_argument("--residual-repeat-arc-bl", type=int, default=1)
    parser.add_argument("--residual-repeat-arc-br", type=int, default=1)
    parser.add_argument("--arc-scale-fl", type=float, default=-0.20)
    parser.add_argument("--arc-scale-fr", type=float, default=-0.50)
    parser.add_argument("--arc-scale-bl", type=float, default=-0.40)
    parser.add_argument("--arc-scale-br", type=float, default=-0.40)
    parser.add_argument(
        "--arc-controller",
        type=Path,
        default=None,
        help="Optional JSON controller with per-arc base/yaw residual parameters.",
    )
    parser.add_argument(
        "--scaled-arc-controller",
        action="store_true",
        help="Evaluate/view the deterministic safe-carrier plus scaled yaw-table arc controller.",
    )
    parser.add_argument(
        "--residual-bc-only",
        action="store_true",
        help="Run residual pretraining, evaluate/save, and exit before PPO.",
    )
    return parser.parse_args()


def residual_gate(forward: float, yaw: float) -> float:
    forward_gate = smoothstep(abs(float(forward)) / 0.28)
    yaw_gate = smoothstep(abs(float(yaw)) / 0.22)
    return float(forward_gate * yaw_gate)


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def make_policy_obs(
    obs: np.ndarray,
    base_action: np.ndarray,
    gate: float,
    yaw_action: np.ndarray | None = None,
    command: tuple[float, float] | None = None,
) -> np.ndarray:
    if yaw_action is None:
        yaw_action = np.zeros_like(base_action)
    if command is None:
        command = (0.0, 0.0)
    return np.concatenate(
        [
            obs.astype(np.float32),
            base_action.astype(np.float32),
            yaw_action.astype(np.float32),
            np.array([gate], dtype=np.float32),
            np.array([command[0], command[1]], dtype=np.float32),
        ]
    ).astype(np.float32)


def policy_obs_for_agent(
    agent: ActorCritic,
    obs: np.ndarray,
    base_action: np.ndarray,
    gate: float,
    yaw_action: np.ndarray,
    command: tuple[float, float],
) -> np.ndarray:
    features = make_policy_obs(obs, base_action, gate, yaw_action, command)
    if features.shape[0] == agent.obs_dim:
        return features
    if features.shape[0] > agent.obs_dim:
        return features[: agent.obs_dim]
    padded = np.zeros(agent.obs_dim, dtype=np.float32)
    padded[: features.shape[0]] = features
    return padded


def component_action(
    library: TeacherLibrary,
    env: BramTripodEnv,
    state: ControllerState,
    arc_yaw_scale: float,
) -> np.ndarray:
    t = state.step * env.dt
    if residual_gate(env.forward_command, env.yaw_rate_command) > 1e-6:
        return library.base_action(
            env.forward_command,
            env.yaw_rate_command,
            t,
            heading_error=state.heading_error,
            yaw_rate=state.yaw_rate,
        )
    _, _, action = library.teacher_action(
        env.forward_command,
        env.yaw_rate_command,
        state.step,
        t,
        arc_yaw_scale,
        heading_error=state.heading_error,
        yaw_rate=state.yaw_rate,
    )
    return action


def final_action(base_action: np.ndarray, residual: np.ndarray, gate: float, limit: float) -> np.ndarray:
    return np.clip(base_action + float(limit) * float(gate) * residual, -1.0, 1.0).astype(
        np.float32
    )


def scaled_arc_residual(
    library: TeacherLibrary,
    forward: float,
    yaw: float,
    step: int,
    gate: float,
    args: argparse.Namespace,
    base_action: np.ndarray | None = None,
) -> np.ndarray:
    if gate <= 1e-6:
        return np.zeros(3, dtype=np.float32)
    params = arc_controller_params(forward, yaw, args)
    yaw_action = arc_yaw_action(library, yaw, step, params)
    yaw_scales = np.asarray(params["yaw_scales"], dtype=np.float32)
    if base_action is None:
        base_delta = np.zeros(3, dtype=np.float32)
    else:
        base_delta = (float(params["base_scale"]) - 1.0) * base_action.astype(np.float32)
    residual = (base_delta + yaw_scales * yaw_action) / max(1e-6, args.residual_limit * gate)
    return np.clip(residual, -1.0, 1.0).astype(np.float32)


def arc_yaw_action(
    library: TeacherLibrary,
    yaw: float,
    step: int,
    params: dict[str, Any],
) -> np.ndarray:
    return library.yaw_teacher_action(yaw, int(step) + int(round(float(params["step_offset"]))))


def arc_yaw_feature(
    library: TeacherLibrary,
    forward: float,
    yaw: float,
    step: int,
    args: argparse.Namespace,
) -> np.ndarray:
    if residual_gate(forward, yaw) <= 1e-6:
        return library.yaw_teacher_action(yaw, step)
    return arc_yaw_action(library, yaw, step, arc_controller_params(forward, yaw, args))


def arc_controller_params(
    forward: float,
    yaw: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    key = arc_command_name(forward, yaw)
    data = getattr(args, "arc_controller_data", None)
    if data is not None:
        command_data = data.get("commands", {}).get(key)
        if command_data is not None:
            return select_arc_params(
                command_data,
                abs(float(forward)),
                abs(float(yaw)),
                arc_scale(forward, yaw, args),
            )
    scale = arc_scale(forward, yaw, args)
    return {
        "base_scale": 1.0,
        "yaw_scales": [scale, scale, scale],
        "step_offset": 0,
    }


def select_arc_params(
    command_data: dict[str, Any],
    forward_mag: float,
    yaw_mag: float,
    fallback_scale: float,
) -> dict[str, Any]:
    grid = command_data.get("grid")
    if isinstance(grid, dict) and grid:
        return interpolate_arc_grid_params(
            command_data,
            grid,
            forward_mag,
            yaw_mag,
            fallback_scale,
        )
    return normalize_arc_params(command_data, fallback_scale)


def interpolate_arc_grid_params(
    command_data: dict[str, Any],
    grid: dict[str, Any],
    forward_mag: float,
    yaw_mag: float,
    fallback_scale: float,
) -> dict[str, Any]:
    parsed = parse_arc_grid(grid)
    if not parsed:
        return normalize_arc_params(command_data, fallback_scale)

    forward_values = sorted({key[0] for key in parsed})
    yaw_values = sorted({key[1] for key in parsed})
    f0, f1, ft = bracket_value(forward_values, forward_mag)
    y0, y1, yt = bracket_value(yaw_values, yaw_mag)

    p00 = grid_params(parsed, command_data, f0, y0, forward_mag, yaw_mag, fallback_scale)
    p10 = grid_params(parsed, command_data, f1, y0, forward_mag, yaw_mag, fallback_scale)
    p01 = grid_params(parsed, command_data, f0, y1, forward_mag, yaw_mag, fallback_scale)
    p11 = grid_params(parsed, command_data, f1, y1, forward_mag, yaw_mag, fallback_scale)
    return blend_arc_params(p00, p10, p01, p11, ft, yt)


def parse_arc_grid(grid: dict[str, Any]) -> dict[tuple[float, float], dict[str, Any]]:
    parsed: dict[tuple[float, float], dict[str, Any]] = {}
    for key, value in grid.items():
        parsed_key = parse_arc_grid_key(key)
        if parsed_key is not None and isinstance(value, dict):
            parsed[parsed_key] = value
    return parsed


def parse_arc_grid_key(key: str) -> tuple[float, float] | None:
    try:
        forward_part, yaw_part = key.split("_", maxsplit=1)
        if not forward_part.startswith("f") or not yaw_part.startswith("y"):
            return None
        forward = float(forward_part[1:].replace("p", "."))
        yaw = float(yaw_part[1:].replace("p", "."))
    except ValueError:
        return None
    return round(forward, 2), round(yaw, 2)


def arc_grid_key(forward_mag: float, yaw_mag: float) -> str:
    return f"f{forward_mag:.2f}_y{yaw_mag:.2f}".replace(".", "p")


def bracket_value(values: list[float], target: float) -> tuple[float, float, float]:
    if not values:
        raise ValueError("Cannot bracket an empty value list")
    target = float(target)
    if target <= values[0]:
        return values[0], values[0], 0.0
    if target >= values[-1]:
        return values[-1], values[-1], 0.0
    for index in range(len(values) - 1):
        low = values[index]
        high = values[index + 1]
        if low <= target <= high:
            if high == low:
                return low, high, 0.0
            return low, high, (target - low) / (high - low)
    return values[-1], values[-1], 0.0


def grid_params(
    parsed: dict[tuple[float, float], dict[str, Any]],
    command_data: dict[str, Any],
    forward_mag: float,
    yaw_mag: float,
    target_forward: float,
    target_yaw: float,
    fallback_scale: float,
) -> dict[str, Any]:
    direct = parsed.get((round(forward_mag, 2), round(yaw_mag, 2)))
    if direct is not None:
        return normalize_arc_params(direct, fallback_scale)
    nearest_key = min(
        parsed,
        key=lambda item: (item[0] - target_forward) ** 2 + (item[1] - target_yaw) ** 2,
    )
    nearest = parsed.get(nearest_key)
    if nearest is not None:
        return normalize_arc_params(nearest, fallback_scale)
    return normalize_arc_params(command_data, fallback_scale)


def blend_arc_params(
    p00: dict[str, Any],
    p10: dict[str, Any],
    p01: dict[str, Any],
    p11: dict[str, Any],
    ft: float,
    yt: float,
) -> dict[str, Any]:
    w00 = (1.0 - ft) * (1.0 - yt)
    w10 = ft * (1.0 - yt)
    w01 = (1.0 - ft) * yt
    w11 = ft * yt
    weights = np.asarray([w00, w10, w01, w11], dtype=np.float64)
    params = (p00, p10, p01, p11)
    base_scale = float(sum(weight * param["base_scale"] for weight, param in zip(weights, params)))
    yaw_scales = np.sum(
        [
            weight * np.asarray(param["yaw_scales"], dtype=np.float64)
            for weight, param in zip(weights, params)
        ],
        axis=0,
    )
    step_offset = float(sum(weight * param["step_offset"] for weight, param in zip(weights, params)))
    return {
        "base_scale": base_scale,
        "yaw_scales": [float(value) for value in yaw_scales],
        "step_offset": int(round(step_offset)),
    }


def normalize_arc_params(params: dict[str, Any], fallback_scale: float) -> dict[str, Any]:
    yaw_scales = params.get("yaw_scales", params.get("yaw_scale", fallback_scale))
    if isinstance(yaw_scales, (int, float, np.integer, np.floating)):
        yaw_scales = [float(yaw_scales)] * 3
    if len(yaw_scales) != 3:
        raise ValueError(f"arc controller yaw_scales must have 3 values, got {yaw_scales}")
    return {
        "base_scale": float(params.get("base_scale", 1.0)),
        "yaw_scales": [float(value) for value in yaw_scales],
        "step_offset": int(round(float(params.get("step_offset", 0)))),
    }


def arc_command_name(forward: float, yaw: float) -> str:
    forward_sign = 1 if float(forward) >= 0.0 else -1
    yaw_sign = 1 if float(yaw) >= 0.0 else -1
    return ARC_COMMAND_NAMES[(forward_sign, yaw_sign)]


def arc_scale(forward: float, yaw: float, args: argparse.Namespace) -> float:
    if forward >= 0.0 and yaw >= 0.0:
        return float(args.arc_scale_fl)
    if forward >= 0.0 and yaw < 0.0:
        return float(args.arc_scale_fr)
    if forward < 0.0 and yaw >= 0.0:
        return float(args.arc_scale_bl)
    return float(args.arc_scale_br)


def load_arc_controller(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.open("r") as f:
        data = json.load(f)
    if "commands" not in data or not isinstance(data["commands"], dict):
        raise ValueError(f"{path} does not contain a commands object")
    return data


def prepare_arc_controller(args: argparse.Namespace) -> None:
    args.arc_controller_data = load_arc_controller(args.arc_controller)


def sample_command(rng: np.random.Generator) -> tuple[float, float]:
    mode = float(rng.uniform())
    if mode < 0.03:
        return 0.0, 0.0
    if mode < 0.08:
        return signed_magnitude(rng, 0.35, 1.0), 0.0
    if mode < 0.15:
        return 0.0, signed_magnitude(rng, 0.35, 1.0)
    if mode < 0.76:
        index = int(rng.choice(len(EXACT_ARC_COMMANDS), p=EXACT_ARC_WEIGHTS))
        return EXACT_ARC_COMMANDS[index]
    if mode < 0.92:
        return signed_magnitude(rng, 0.65, 1.0), signed_magnitude(rng, 0.55, 1.0)
    return signed_magnitude(rng, 0.35, 0.80), signed_magnitude(rng, 0.25, 0.70)


def signed_magnitude(rng: np.random.Generator, low: float, high: float) -> float:
    value = float(rng.uniform(low, high))
    return -value if float(rng.uniform()) < 0.5 else value


def shaped_reward(
    env_reward: float,
    info: dict[str, Any],
    residual: np.ndarray,
    gate: float,
    args: argparse.Namespace,
) -> float:
    if gate <= 1e-6:
        return float(env_reward)

    forward = float(info.get("forward_command", 0.0))
    yaw = float(info.get("yaw_rate_command", 0.0))
    forward_progress = forward * float(info.get("line_velocity", 0.0))
    yaw_progress = yaw * float(info.get("yaw_rate", 0.0))
    wrong_forward = max(0.0, -forward_progress)
    wrong_yaw = max(0.0, -yaw_progress)
    tracking = (
        1.25 * float(info.get("linear_tracking_reward", 0.0))
        + 1.25 * float(info.get("yaw_tracking_reward", 0.0))
        + 0.65 * float(info.get("command_quality", 0.0))
    )
    progress = 7.00 * max(0.0, forward_progress) + 3.50 * max(0.0, yaw_progress)
    residual_cost = 0.020 * float(np.mean(np.square(residual)))
    wrong_cost = 10.00 * wrong_forward + 8.00 * wrong_yaw
    drift_cost = 0.08 * min(abs(float(info.get("cross_track_error", 0.0))), 0.8)
    posture_cost = (
        16.0 * float(info.get("body_height_warning_deficit", 0.0))
        + 80.0 * float(info.get("body_height_deficit", 0.0))
        + 0.15 * float(info.get("level_tilt_excess_rad", 0.0))
        + 0.035 * float(info.get("termination_penalty", 0.0))
    )
    survival_bonus = 0.04
    return float(
        gate
        * (
            survival_bonus
            + tracking
            + progress
            - wrong_cost
            - drift_cost
            - posture_cost
            - residual_cost
        )
    )


def update_state(state: ControllerState, info: dict[str, Any]) -> None:
    state.step += 1
    state.heading_error = float(info.get("heading_error", 0.0))
    state.yaw_rate = float(info.get("yaw_rate", 0.0))


def reset_env(
    env: BramTripodEnv,
    seed: int,
    command: tuple[float, float],
) -> tuple[np.ndarray, ControllerState]:
    obs, _ = env.reset(seed=seed, options=command_options(command[0], command[1]))
    return obs, ControllerState()


def pretrain_residual_actor(
    agent: ActorCritic,
    library: TeacherLibrary,
    optimizer: torch.optim.Optimizer,
    obs_dim: int,
    args: argparse.Namespace,
) -> None:
    x, y = build_residual_pretrain_dataset(library, obs_dim, args)
    if x.shape[0] == 0:
        print("residual_pretrain skipped=no_samples", flush=True)
        return

    x, y = cap_residual_dataset(x, y, args.residual_dataset_max_samples)
    train_residual_supervised(
        agent,
        optimizer,
        x,
        y,
        args.residual_pretrain_steps,
        args,
        "residual_pretrain",
    )
    for round_index in range(args.residual_dagger_rounds):
        dagger_x, dagger_y = build_residual_pretrain_dataset(
            library,
            obs_dim,
            args,
            rollout_agent=agent,
            episodes=args.residual_dagger_episodes,
            seed_offset=105_000 + 10_000 * round_index,
        )
        if dagger_x.shape[0] == 0:
            print(f"residual_dagger round={round_index + 1} skipped=no_samples", flush=True)
            continue
        x = torch.cat([x, dagger_x], dim=0)
        y = torch.cat([y, dagger_y], dim=0)
        x, y = cap_residual_dataset(x, y, args.residual_dataset_max_samples)
        train_residual_supervised(
            agent,
            optimizer,
            x,
            y,
            args.residual_dagger_steps,
            args,
            f"residual_dagger round={round_index + 1}",
        )
    agent.eval()


def cap_residual_dataset(
    x: torch.Tensor,
    y: torch.Tensor,
    max_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_samples <= 0 or x.shape[0] <= max_samples:
        return x, y
    keep = torch.randperm(x.shape[0])[:max_samples]
    return x[keep], y[keep]


def train_residual_supervised(
    agent: ActorCritic,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int,
    args: argparse.Namespace,
    label: str,
) -> None:
    if steps <= 0:
        return
    batch_size = min(args.residual_pretrain_batch_size, x.shape[0])
    agent.train()
    for step in range(1, steps + 1):
        indices = torch.randint(0, x.shape[0], (batch_size,))
        obs_batch = x[indices]
        target_batch = y[indices]
        pred = agent.deterministic_action(obs_batch)
        loss = torch.mean((pred - target_batch) ** 2)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.actor.parameters(), args.max_grad_norm)
        optimizer.step()
        if step == 1 or step == steps or step % 250 == 0:
            print(
                f"{label} "
                f"step={step}/{steps} "
                f"samples={x.shape[0]} "
                f"loss={float(loss.detach()):.6f}",
                flush=True,
            )
    agent.eval()


def build_residual_pretrain_dataset(
    library: TeacherLibrary,
    obs_dim: int,
    args: argparse.Namespace,
    rollout_agent: ActorCritic | None = None,
    episodes: int | None = None,
    seed_offset: int = 90_000,
) -> tuple[torch.Tensor, torch.Tensor]:
    obs_samples: list[np.ndarray] = []
    residual_samples: list[np.ndarray] = []
    commands = residual_pretrain_commands(args)
    total_episodes = args.residual_pretrain_episodes if episodes is None else episodes
    for episode in range(total_episodes):
        for command_index, command in enumerate(commands):
            env = BramTripodEnv(
                episode_seconds=args.episode_seconds,
                randomize_reset=False,
                domain_randomization=False,
                randomize_command=False,
                command_forward=command[0],
                command_yaw_rate=command[1],
            )
            obs, state = reset_env(env, args.seed + seed_offset + 1000 * episode + command_index, command)
            for _ in range(env.max_steps):
                base = component_action(library, env, state, args.arc_yaw_scale)
                gate = residual_gate(env.forward_command, env.yaw_rate_command)
                residual = scaled_arc_residual(
                    library,
                    env.forward_command,
                    env.yaw_rate_command,
                    state.step,
                    gate,
                    args,
                    base,
                )
                yaw_action = arc_yaw_feature(
                    library,
                    env.forward_command,
                    env.yaw_rate_command,
                    state.step,
                    args,
                )
                policy_obs = make_policy_obs(
                    obs,
                    base,
                    gate,
                    yaw_action,
                    (env.forward_command, env.yaw_rate_command),
                )
                if policy_obs.shape[0] != obs_dim:
                    raise ValueError(
                        f"pretrain obs dim {policy_obs.shape[0]} != expected {obs_dim}"
                    )
                obs_samples.append(policy_obs)
                residual_samples.append(residual)
                step_residual = residual
                if rollout_agent is not None:
                    with torch.no_grad():
                        step_residual = (
                            rollout_agent.deterministic_action(
                                torch.as_tensor(policy_obs[None, :], dtype=torch.float32)
                            )
                            .cpu()
                            .numpy()[0]
                        )
                action = final_action(base, step_residual, gate, args.residual_limit)
                obs, _, terminated, truncated, info = env.step(action)
                update_state(state, info)
                if terminated or truncated:
                    break
            env.close()

    if not obs_samples:
        return (
            torch.empty((0, obs_dim), dtype=torch.float32),
            torch.empty((0, 3), dtype=torch.float32),
        )
    return (
        torch.as_tensor(np.stack(obs_samples), dtype=torch.float32),
        torch.as_tensor(np.stack(residual_samples), dtype=torch.float32),
    )


def residual_pretrain_commands(args: argparse.Namespace) -> tuple[tuple[float, float], ...]:
    repeats = {
        "arc_fl": max(1, int(args.residual_repeat_arc_fl)),
        "arc_fr": max(1, int(args.residual_repeat_arc_fr)),
        "arc_bl": max(1, int(args.residual_repeat_arc_bl)),
        "arc_br": max(1, int(args.residual_repeat_arc_br)),
    }
    source_commands = BROAD_ARC_COMMANDS if args.residual_train_broad_mixed else EXACT_ARC_COMMANDS
    commands: list[tuple[float, float]] = []
    for command in source_commands:
        commands.extend([command] * repeats[arc_command_name(command[0], command[1])])
    return tuple(commands)


def main() -> None:
    args = parse_args()
    maybe_relaunch_with_mjpython(args)
    prepare_arc_controller(args)
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    library = make_library(args)
    if args.scaled_arc_controller:
        if args.view:
            run_scaled_viewer(library, args)
        else:
            print_eval(evaluate_scaled_controller(library, args))
        return

    if args.checkpoint is not None:
        agent, payload = load_checkpoint(args.checkpoint)
        apply_checkpoint_args(args, payload)
        prepare_arc_controller(args)
        library = make_library(args)
        if args.view:
            run_viewer(agent, library, args)
        else:
            stats = evaluate(agent, library, args)
            print_eval(stats)
        return

    envs = [
        BramTripodEnv(
            episode_seconds=args.episode_seconds,
            randomize_reset=True,
            domain_randomization=args.domain_randomization,
            domain_randomization_strength=args.domain_randomization_strength,
            randomize_command=False,
        )
        for _ in range(args.num_envs)
    ]
    states: list[ControllerState] = []
    obs_list: list[np.ndarray] = []
    for index, env in enumerate(envs):
        obs, state = reset_env(env, args.seed + index, sample_command(rng))
        obs_list.append(obs)
        states.append(state)

    first_base = component_action(library, envs[0], states[0], args.arc_yaw_scale)
    obs_dim = make_policy_obs(
        obs_list[0],
        first_base,
        0.0,
        command=(envs[0].forward_command, envs[0].yaw_rate_command),
    ).shape[0]
    agent = ActorCritic(obs_dim, envs[0].action_space.shape[0], args.hidden_size, args.log_std_init)
    if args.init_checkpoint is not None:
        load_initial_checkpoint(args.init_checkpoint, agent)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    if args.residual_pretrain_steps > 0:
        pretrain_residual_actor(agent, library, optimizer, obs_dim, args)

    run_dir = args.output_dir / time.strftime("residual_ppo_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = run_dir / "snapshots"
    if args.snapshot_interval > 0:
        snapshot_dir.mkdir(exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    checkpoint_path = run_dir / "residual_policy.pt"
    best_checkpoint_path = run_dir / "residual_policy_best.pt"

    if args.residual_bc_only:
        stats = evaluate(agent, library, args)
        print_eval(stats, prefix="bc_only_eval")
        save_checkpoint(checkpoint_path, agent, args, stats)
        save_checkpoint(best_checkpoint_path, agent, args, stats)
        print(f"saved={run_dir}")
        for env in envs:
            env.close()
        return

    total_batch = args.num_envs * args.rollout_steps
    updates = max(1, args.total_steps // total_batch)
    best_score = -float("inf")
    recent_returns = deque(maxlen=50)
    recent_lengths = deque(maxlen=50)
    episode_returns = np.zeros(args.num_envs, dtype=np.float64)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int32)
    global_step = 0

    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "update",
                "global_step",
                "sps",
                "recent_return",
                "recent_length",
                "eval_reward",
                "eval_score",
                "eval_arc_score",
                "eval_arc_command_distance",
                "eval_arc_length",
                "eval_command_distance",
                "eval_length",
                "policy_loss",
                "value_loss",
                "entropy",
                "approx_kl",
            ],
        )
        writer.writeheader()
        start_time = time.time()

        for update in range(1, updates + 1):
            obs_buf = torch.zeros((args.rollout_steps, args.num_envs, obs_dim), dtype=torch.float32)
            action_buf = torch.zeros((args.rollout_steps, args.num_envs, envs[0].action_space.shape[0]), dtype=torch.float32)
            logprob_buf = torch.zeros((args.rollout_steps, args.num_envs), dtype=torch.float32)
            reward_buf = torch.zeros((args.rollout_steps, args.num_envs), dtype=torch.float32)
            done_buf = torch.zeros((args.rollout_steps, args.num_envs), dtype=torch.float32)
            value_buf = torch.zeros((args.rollout_steps, args.num_envs), dtype=torch.float32)

            for step in range(args.rollout_steps):
                policy_obs_rows = []
                bases = []
                gates = []
                for env_index, env in enumerate(envs):
                    base = component_action(library, env, states[env_index], args.arc_yaw_scale)
                    gate = residual_gate(env.forward_command, env.yaw_rate_command)
                    yaw_action = arc_yaw_feature(
                        library,
                        env.forward_command,
                        env.yaw_rate_command,
                        states[env_index].step,
                        args,
                    )
                    bases.append(base)
                    gates.append(gate)
                    policy_obs_rows.append(
                        make_policy_obs(
                            obs_list[env_index],
                            base,
                            gate,
                            yaw_action,
                            (env.forward_command, env.yaw_rate_command),
                        )
                    )

                policy_obs = torch.as_tensor(np.stack(policy_obs_rows), dtype=torch.float32)
                with torch.no_grad():
                    residuals, logprob, _, value = agent.get_action_and_value(policy_obs)

                obs_buf[step] = policy_obs
                action_buf[step] = residuals
                logprob_buf[step] = logprob
                value_buf[step] = value
                global_step += args.num_envs

                residuals_np = residuals.cpu().numpy().astype(np.float32)
                for env_index, env in enumerate(envs):
                    action = final_action(
                        bases[env_index],
                        residuals_np[env_index],
                        gates[env_index],
                        args.residual_limit,
                    )
                    next_obs, env_reward, terminated, truncated, info = env.step(action)
                    reward = shaped_reward(env_reward, info, residuals_np[env_index], gates[env_index], args)
                    done = bool(terminated or truncated)
                    reward_buf[step, env_index] = float(reward)
                    done_buf[step, env_index] = float(done)
                    episode_returns[env_index] += reward
                    episode_lengths[env_index] += 1
                    update_state(states[env_index], info)

                    if done:
                        recent_returns.append(float(episode_returns[env_index]))
                        recent_lengths.append(float(episode_lengths[env_index]))
                        episode_returns[env_index] = 0.0
                        episode_lengths[env_index] = 0
                        next_obs, states[env_index] = reset_env(
                            env,
                            args.seed + global_step + env_index,
                            sample_command(rng),
                        )
                    obs_list[env_index] = next_obs

            next_values = []
            with torch.no_grad():
                for env_index, env in enumerate(envs):
                    base = component_action(library, env, states[env_index], args.arc_yaw_scale)
                    gate = residual_gate(env.forward_command, env.yaw_rate_command)
                    yaw_action = arc_yaw_feature(
                        library,
                        env.forward_command,
                        env.yaw_rate_command,
                        states[env_index].step,
                        args,
                    )
                    policy_obs = torch.as_tensor(
                        make_policy_obs(
                            obs_list[env_index],
                            base,
                            gate,
                            yaw_action,
                            (env.forward_command, env.yaw_rate_command),
                        )[None, :],
                        dtype=torch.float32,
                    )
                    next_values.append(float(agent.get_value(policy_obs)[0]))

            advantages = torch.zeros_like(reward_buf)
            lastgaelam = torch.zeros(args.num_envs, dtype=torch.float32)
            next_value_tensor = torch.as_tensor(next_values, dtype=torch.float32)
            for t in reversed(range(args.rollout_steps)):
                if t == args.rollout_steps - 1:
                    next_nonterminal = 1.0 - done_buf[t]
                    next_value = next_value_tensor
                else:
                    next_nonterminal = 1.0 - done_buf[t + 1]
                    next_value = value_buf[t + 1]
                delta = reward_buf[t] + args.gamma * next_value * next_nonterminal - value_buf[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * next_nonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + value_buf
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            b_obs = obs_buf.reshape((-1, obs_dim))
            b_actions = action_buf.reshape((-1, envs[0].action_space.shape[0]))
            b_logprobs = logprob_buf.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = value_buf.reshape(-1)

            indices = np.arange(total_batch)
            policy_loss_value = 0.0
            value_loss_value = 0.0
            entropy_value = 0.0
            approx_kl_value = 0.0
            for _ in range(args.update_epochs):
                rng.shuffle(indices)
                for start in range(0, total_batch, args.minibatch_size):
                    mb_idx = indices[start : start + args.minibatch_size]
                    _, new_logprob, entropy, new_value = agent.get_action_and_value(
                        b_obs[mb_idx],
                        b_actions[mb_idx],
                    )
                    logratio = new_logprob - b_logprobs[mb_idx]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - logratio).mean()
                    mb_advantages = b_advantages[mb_idx]
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio,
                        1.0 - args.clip_coef,
                        1.0 + args.clip_coef,
                    )
                    policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                    value_loss = 0.5 * ((new_value - b_returns[mb_idx]) ** 2).mean()
                    entropy_loss = entropy.mean()
                    loss = (
                        policy_loss
                        - args.entropy_coef * entropy_loss
                        + args.value_coef * value_loss
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                    policy_loss_value = float(policy_loss.detach())
                    value_loss_value = float(value_loss.detach())
                    entropy_value = float(entropy_loss.detach())
                    approx_kl_value = float(approx_kl.detach())
                if approx_kl_value > args.target_kl:
                    break

            should_eval = update == 1 or update == updates or update % args.eval_interval == 0
            eval_stats = None
            if should_eval:
                eval_stats = evaluate(agent, library, args)
                print_eval(eval_stats, prefix=f"eval update={update}")
                if eval_stats.arc_score > best_score:
                    best_score = eval_stats.arc_score
                    save_checkpoint(best_checkpoint_path, agent, args, eval_stats)

            if update == updates or update % max(1, args.eval_interval) == 0:
                save_checkpoint(checkpoint_path, agent, args, eval_stats)
            if args.snapshot_interval > 0 and update % args.snapshot_interval == 0:
                save_checkpoint(
                    snapshot_dir / f"residual_policy_update_{update:05d}_step_{global_step}.pt",
                    agent,
                    args,
                    eval_stats,
                )

            elapsed = max(1e-6, time.time() - start_time)
            sps = int(global_step / elapsed)
            row = {
                "update": update,
                "global_step": global_step,
                "sps": sps,
                "recent_return": float(np.mean(recent_returns)) if recent_returns else float("nan"),
                "recent_length": float(np.mean(recent_lengths)) if recent_lengths else float("nan"),
                "eval_reward": eval_stats.reward if eval_stats else float("nan"),
                "eval_score": eval_stats.score if eval_stats else float("nan"),
                "eval_arc_score": eval_stats.arc_score if eval_stats else float("nan"),
                "eval_arc_command_distance": (
                    eval_stats.arc_command_distance if eval_stats else float("nan")
                ),
                "eval_arc_length": eval_stats.arc_length if eval_stats else float("nan"),
                "eval_command_distance": eval_stats.command_distance if eval_stats else float("nan"),
                "eval_length": eval_stats.length if eval_stats else float("nan"),
                "policy_loss": policy_loss_value,
                "value_loss": value_loss_value,
                "entropy": entropy_value,
                "approx_kl": approx_kl_value,
            }
            writer.writerow(row)
            f.flush()
            print(
                f"update={update:04d}/{updates:04d} step={global_step} "
                f"recent={row['recent_return']:.2f} eval_score={row['eval_score']:.3f} "
                f"arc_score={row['eval_arc_score']:.3f} "
                f"sps={sps} entropy={entropy_value:.3f}",
                flush=True,
            )

    for env in envs:
        env.close()
    print(f"saved={run_dir}")


def evaluate(agent: ActorCritic, library: TeacherLibrary, args: argparse.Namespace) -> EvalStats:
    results: list[EvalResult] = []
    for name, forward, yaw in eval_command_suite(args):
        episode_results = []
        for episode in range(args.eval_episodes):
            env = BramTripodEnv(
                episode_seconds=args.episode_seconds,
                randomize_reset=False,
                domain_randomization=False,
                randomize_command=False,
                command_forward=forward,
                command_yaw_rate=yaw,
            )
            obs, state = reset_env(env, args.seed + 80_000 + episode, (forward, yaw))
            total_reward = 0.0
            final_info: dict[str, Any] = {}
            terminated = False
            truncated = False
            for step in range(env.max_steps):
                base = component_action(library, env, state, args.arc_yaw_scale)
                gate = residual_gate(forward, yaw)
                yaw_action = arc_yaw_feature(library, forward, yaw, state.step, args)
                policy_obs = torch.as_tensor(
                    policy_obs_for_agent(
                        agent,
                        obs,
                        base,
                        gate,
                        yaw_action,
                        (forward, yaw),
                    )[None, :],
                    dtype=torch.float32,
                )
                with torch.no_grad():
                    residual = agent.deterministic_action(policy_obs).cpu().numpy()[0]
                action = final_action(base, residual, gate, args.residual_limit)
                obs, reward, terminated, truncated, final_info = env.step(action)
                total_reward += shaped_reward(reward, final_info, residual, gate, args)
                update_state(state, final_info)
                if terminated or truncated:
                    break
            env.close()
            episode_results.append(result_from_info(name, total_reward, final_info, step + 1, terminated))
        results.append(mean_result(episode_results))
    return stats_from_results(results)


def evaluate_scaled_controller(library: TeacherLibrary, args: argparse.Namespace) -> EvalStats:
    results: list[EvalResult] = []
    for name, forward, yaw in eval_command_suite(args):
        episode_results = []
        for episode in range(args.eval_episodes):
            env = BramTripodEnv(
                episode_seconds=args.episode_seconds,
                randomize_reset=False,
                domain_randomization=False,
                randomize_command=False,
                command_forward=forward,
                command_yaw_rate=yaw,
            )
            obs, state = reset_env(env, args.seed + 82_000 + episode, (forward, yaw))
            total_reward = 0.0
            final_info: dict[str, Any] = {}
            terminated = False
            truncated = False
            for step in range(env.max_steps):
                base = component_action(library, env, state, args.arc_yaw_scale)
                gate = residual_gate(forward, yaw)
                residual = scaled_arc_residual(library, forward, yaw, state.step, gate, args, base)
                action = final_action(base, residual, gate, args.residual_limit)
                obs, reward, terminated, truncated, final_info = env.step(action)
                total_reward += shaped_reward(reward, final_info, residual, gate, args)
                update_state(state, final_info)
                if terminated or truncated:
                    break
            env.close()
            episode_results.append(
                result_from_info(name, total_reward, final_info, step + 1, terminated)
            )
        results.append(mean_result(episode_results))
    return stats_from_results(results)


def result_from_info(
    command: str,
    reward: float,
    info: dict[str, Any],
    length: int,
    terminated: bool,
) -> EvalResult:
    command_distance = float(info.get("command_distance", 0.0))
    line_distance = float(info.get("line_distance", 0.0))
    yaw_distance = float(info.get("yaw_distance", 0.0))
    cross = abs(float(info.get("cross_track_error", 0.0)))
    score = command_distance - 0.20 * cross - (0.20 if terminated else 0.0)
    return EvalResult(
        command=command,
        reward=float(reward),
        score=float(score),
        command_distance=command_distance,
        line_distance=line_distance,
        yaw_distance=yaw_distance,
        x_distance=float(info.get("x_distance", 0.0)),
        y_distance=float(info.get("y_distance", 0.0)),
        cross_track_error=float(info.get("cross_track_error", 0.0)),
        heading_error=float(info.get("heading_error", 0.0)),
        length=int(length),
        terminated=bool(terminated),
    )


def mean_result(results: list[EvalResult]) -> EvalResult:
    return EvalResult(
        command=results[0].command,
        reward=float(np.mean([result.reward for result in results])),
        score=float(np.mean([result.score for result in results])),
        command_distance=float(np.mean([result.command_distance for result in results])),
        line_distance=float(np.mean([result.line_distance for result in results])),
        yaw_distance=float(np.mean([result.yaw_distance for result in results])),
        x_distance=float(np.mean([result.x_distance for result in results])),
        y_distance=float(np.mean([result.y_distance for result in results])),
        cross_track_error=float(np.mean([result.cross_track_error for result in results])),
        heading_error=float(np.mean([result.heading_error for result in results])),
        length=int(round(float(np.mean([result.length for result in results])))),
        terminated=any(result.terminated for result in results),
    )


def stats_from_results(results: list[EvalResult]) -> EvalStats:
    weights = np.asarray([command_weight(result.command) for result in results], dtype=np.float64)
    scores = np.asarray([result.score for result in results], dtype=np.float64)
    arc_results = [result for result in results if result.command.startswith("arc_")]
    return EvalStats(
        reward=float(np.mean([result.reward for result in results])),
        score=float(np.average(scores, weights=weights)),
        arc_score=float(np.mean([result.score for result in arc_results])),
        arc_command_distance=float(
            np.mean([result.command_distance for result in arc_results])
        ),
        arc_length=float(np.mean([result.length for result in arc_results])),
        command_distance=float(np.mean([result.command_distance for result in results])),
        length=float(np.mean([result.length for result in results])),
        per_command=tuple(results),
    )


def eval_command_suite(args: argparse.Namespace) -> tuple[tuple[str, float, float], ...]:
    return BROAD_COMMAND_SUITE if args.eval_suite == "broad" else COMMAND_SUITE


def command_weight(command: str) -> float:
    if command.startswith("arc_"):
        return 4.0
    if command == "idle":
        return 0.35
    return 0.75


def print_eval(stats: EvalStats, prefix: str = "eval") -> None:
    print(
        f"{prefix} reward={stats.reward:.3f} score={stats.score:.4f} "
        f"arc_score={stats.arc_score:.4f} "
        f"arc_cmd={stats.arc_command_distance:.4f} "
        f"cmd={stats.command_distance:.4f} len={stats.length:.1f}",
        flush=True,
    )
    for result in stats.per_command:
        print(
            f"  {result.command:7s} cmd={result.command_distance: .4f} "
            f"line={result.line_distance: .4f} yaw={result.yaw_distance: .4f} "
            f"xy=({result.x_distance: .3f},{result.y_distance: .3f}) "
            f"cross={result.cross_track_error: .3f} "
            f"len={result.length} term={result.terminated}",
            flush=True,
        )


def save_checkpoint(
    path: Path,
    agent: ActorCritic,
    args: argparse.Namespace,
    stats: EvalStats | None,
) -> None:
    payload = {
        "model_state_dict": agent.state_dict(),
        "obs_dim": agent.obs_dim,
        "action_dim": agent.action_dim,
        "hidden_size": agent.hidden_size,
        "env_command_mode": ENV_COMMAND_MODE,
        "args": serializable_args(args),
        "eval": asdict(stats) if stats is not None else None,
    }
    torch.save(payload, path)


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: value
        for key, value in vars(args).items()
        if key != "arc_controller_data"
    }


def make_library(args: argparse.Namespace) -> TeacherLibrary:
    return TeacherLibrary(
        args.forward_gait,
        args.backward_gait,
        args.yaw_left_table,
        args.yaw_right_table,
        args.base_scaling,
        args.base_speed_min,
        args.base_action_min,
    )


def load_checkpoint(path: Path) -> tuple[ActorCritic, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    agent = ActorCritic(
        int(payload["obs_dim"]),
        int(payload["action_dim"]),
        int(payload.get("hidden_size", payload.get("args", {}).get("hidden_size", 64))),
        float(payload.get("args", {}).get("log_std_init", -1.6)),
    )
    agent.load_state_dict(payload["model_state_dict"])
    agent.eval()
    return agent, payload


def load_initial_checkpoint(path: Path, agent: ActorCritic) -> None:
    payload = torch.load(path, map_location="cpu")
    checkpoint_obs_dim = int(payload["obs_dim"])
    checkpoint_action_dim = int(payload["action_dim"])
    if (checkpoint_obs_dim, checkpoint_action_dim) != (agent.obs_dim, agent.action_dim):
        raise ValueError(
            f"{path} has obs/action dims "
            f"{checkpoint_obs_dim}/{checkpoint_action_dim}; expected "
            f"{agent.obs_dim}/{agent.action_dim}."
        )
    agent.load_state_dict(payload["model_state_dict"])
    print(
        "loaded_init_checkpoint "
        f"path={path} "
        f"arc_score={payload.get('eval', {}).get('arc_score', float('nan'))}",
        flush=True,
    )


def apply_checkpoint_args(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    saved = payload.get("args", {})
    for key in (
        "forward_gait",
        "backward_gait",
        "yaw_left_table",
        "yaw_right_table",
        "residual_limit",
        "arc_yaw_scale",
        "base_scaling",
        "base_speed_min",
        "base_action_min",
        "episode_seconds",
        "arc_controller",
    ):
        if key in saved:
            value = saved[key]
            if value is not None and (
                key.endswith("gait") or key.endswith("table") or key == "arc_controller"
            ):
                value = Path(value)
            setattr(args, key, value)


def maybe_relaunch_with_mjpython(args: argparse.Namespace) -> None:
    if not args.view or platform.system() != "Darwin":
        return
    if Path(sys.executable).name == "mjpython" or os.environ.get("MJPYTHON_BIN"):
        return
    mjpython = Path(sys.executable).with_name("mjpython")
    if mjpython.exists():
        os.execv(str(mjpython), [str(mjpython), *sys.argv])


def run_viewer(agent: ActorCritic, library: TeacherLibrary, args: argparse.Namespace) -> None:
    import mujoco.viewer

    commands = viewer_commands(args.view_command)
    command_index = 0
    name, forward, yaw = commands[command_index]
    env = BramTripodEnv(
        episode_seconds=args.episode_seconds,
        randomize_reset=False,
        domain_randomization=False,
        randomize_command=False,
        command_forward=forward,
        command_yaw_rate=yaw,
    )
    obs, state = reset_env(env, args.seed, (forward, yaw))
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            started = time.perf_counter()
            base = component_action(library, env, state, args.arc_yaw_scale)
            gate = residual_gate(forward, yaw)
            yaw_action = arc_yaw_feature(library, forward, yaw, state.step, args)
            policy_obs = torch.as_tensor(
                policy_obs_for_agent(
                    agent,
                    obs,
                    base,
                    gate,
                    yaw_action,
                    (forward, yaw),
                )[None, :],
                dtype=torch.float32,
            )
            with torch.no_grad():
                residual = agent.deterministic_action(policy_obs).cpu().numpy()[0]
            action = final_action(base, residual, gate, args.residual_limit)
            obs, _, terminated, truncated, info = env.step(action)
            update_state(state, info)
            viewer.sync()
            if terminated or truncated:
                command_index = (command_index + 1) % len(commands)
                name, forward, yaw = commands[command_index]
                print(f"viewer_command={name} forward={forward:.2f} yaw={yaw:.2f}")
                obs, state = reset_env(env, args.seed + command_index, (forward, yaw))
                time.sleep(0.4)
            sleep_time = (env.dt / max(args.speed, 1e-6)) - (time.perf_counter() - started)
            if sleep_time > 0:
                time.sleep(sleep_time)
    env.close()


def run_scaled_viewer(library: TeacherLibrary, args: argparse.Namespace) -> None:
    import mujoco.viewer

    commands = viewer_commands(args.view_command)
    command_index = 0
    name, forward, yaw = commands[command_index]
    env = BramTripodEnv(
        episode_seconds=args.episode_seconds,
        randomize_reset=False,
        domain_randomization=False,
        randomize_command=False,
        command_forward=forward,
        command_yaw_rate=yaw,
    )
    obs, state = reset_env(env, args.seed, (forward, yaw))
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            started = time.perf_counter()
            base = component_action(library, env, state, args.arc_yaw_scale)
            gate = residual_gate(forward, yaw)
            residual = scaled_arc_residual(library, forward, yaw, state.step, gate, args, base)
            action = final_action(base, residual, gate, args.residual_limit)
            obs, _, terminated, truncated, info = env.step(action)
            update_state(state, info)
            viewer.sync()
            if terminated or truncated:
                command_index = (command_index + 1) % len(commands)
                name, forward, yaw = commands[command_index]
                print(f"viewer_command={name} forward={forward:.2f} yaw={yaw:.2f}")
                obs, state = reset_env(env, args.seed + command_index, (forward, yaw))
                time.sleep(0.4)
            sleep_time = (env.dt / max(args.speed, 1e-6)) - (
                time.perf_counter() - started
            )
            if sleep_time > 0:
                time.sleep(sleep_time)
    env.close()


def viewer_commands(view_command: str) -> tuple[tuple[str, float, float], ...]:
    if view_command == "suite":
        return COMMAND_SUITE
    if view_command == "broad":
        return BROAD_COMMAND_SUITE
    if view_command == "primitives":
        return COMMAND_SUITE[:9]
    for command in BROAD_COMMAND_SUITE:
        if command[0] == view_command:
            return (command,)
    raise ValueError(f"Unknown viewer command: {view_command}")


if __name__ == "__main__":
    main()
