from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bram_env import BramTripodEnv
from search_gait import (
    PARAM_HIGH,
    PARAM_LOW,
    PARAM_NAMES,
    gait_action,
    load_params,
)


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_PUSHED_RUN = MODULE_DIR / "pushed_runs" / "current_policy_20260701"
DEFAULT_FORWARD_GAIT = DEFAULT_PUSHED_RUN / "gaits" / "forward_best_params.json"
DEFAULT_BACKWARD_GAIT = DEFAULT_PUSHED_RUN / "gaits" / "backward_best_params.json"
DEFAULT_YAW_LEFT_TABLE = DEFAULT_PUSHED_RUN / "yaw_tables" / "yaw_left_policy_table.json"
DEFAULT_YAW_RIGHT_TABLE = DEFAULT_PUSHED_RUN / "yaw_tables" / "yaw_right_policy_table.json"

COMMAND_SUITE: tuple[tuple[str, float, float], ...] = (
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
PRIMITIVE_SUITE = COMMAND_SUITE[:9]

TRAIN_FORWARD_VALUES = (-1.0, -0.75, -0.5, 0.0, 0.5, 0.75, 1.0)
TRAIN_YAW_VALUES = (-1.0, -0.75, -0.5, 0.0, 0.5, 0.75, 1.0)

PARAM_MOD_LIMITS = torch.tensor(
    [
        0.55,  # frequency Hz
        0.34,
        0.34,
        0.34,  # centers
        0.52,
        0.52,
        0.52,  # amplitudes
        1.55,
        1.55,
        1.55,  # phase offsets
        0.34,
        0.34,
        0.34,  # second harmonic amplitudes
        1.75,
        1.75,
        1.75,  # second harmonic phase offsets
    ],
    dtype=torch.float32,
)
ACTION_RESIDUAL_DIM = 3


@dataclass(frozen=True)
class EvalResult:
    command: str
    reward: float
    command_distance: float
    line_distance: float
    yaw_distance: float
    x_distance: float
    y_distance: float
    cross_track_error: float
    heading_error: float
    length: int
    terminated: bool


class CpgModulator(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, len(PARAM_MOD_LIMITS) + ACTION_RESIDUAL_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))


class TeacherLibrary:
    def __init__(
        self,
        forward_gait: Path,
        backward_gait: Path,
        yaw_left_table: Path,
        yaw_right_table: Path,
        base_scaling: str,
        base_speed_min: float,
        base_action_min: float,
    ) -> None:
        _, self.forward_params = load_params(forward_gait)
        _, self.backward_params = load_params(backward_gait)
        self.yaw_left_table = load_action_table(yaw_left_table)
        self.yaw_right_table = load_action_table(yaw_right_table)
        self.base_scaling = base_scaling
        self.base_speed_min = float(base_speed_min)
        self.base_action_min = float(base_action_min)

    def scaled_params(
        self, forward_command: float, yaw_command: float
    ) -> tuple[np.ndarray, float, float]:
        forward = float(forward_command)
        yaw = float(yaw_command)
        forward_mag = abs(forward)
        yaw_mag = abs(yaw)
        activity = float(np.clip(max(forward_mag, yaw_mag), 0.0, 1.0))
        params = self.forward_params if forward >= 0.0 else self.backward_params
        if forward_mag < 0.05 and yaw_mag >= 0.05:
            params = self.forward_params
        if activity < 0.05:
            speed_scale = 1.0
            action_scale = 0.0
        elif self.base_scaling == "linear":
            speed_scale = 1.0
            action_scale = activity
        else:
            speed_scale = self.base_speed_min + (1.0 - self.base_speed_min) * activity
            action_scale = self.base_action_min + (1.0 - self.base_action_min) * activity

        scaled = params.copy()
        scaled[0] = params[0] * speed_scale
        scaled[1:4] = params[1:4] * action_scale
        scaled[4:7] = params[4:7] * action_scale
        scaled[10:13] = params[10:13] * action_scale
        return scaled.astype(np.float32), speed_scale, action_scale

    def base_action(
        self,
        forward_command: float,
        yaw_command: float,
        t: float,
        heading_error: float = 0.0,
        yaw_rate: float = 0.0,
    ) -> np.ndarray:
        params, _, _ = self.scaled_params(forward_command, yaw_command)
        use_heading_correction = (
            abs(float(forward_command)) >= 0.05 and abs(float(yaw_command)) < 0.05
        )
        return gait_action(
            params,
            t,
            heading_error=heading_error,
            yaw_rate=yaw_rate,
            use_heading_correction=use_heading_correction,
        )

    def yaw_teacher_action(self, yaw_command: float, step: int) -> np.ndarray:
        magnitude = abs(float(yaw_command))
        if magnitude < 1e-6:
            return np.zeros(3, dtype=np.float32)
        table = self.yaw_left_table if yaw_command > 0.0 else self.yaw_right_table
        return np.clip(magnitude * table_action(table, step), -1.0, 1.0).astype(
            np.float32
        )

    def teacher_action(
        self,
        forward_command: float,
        yaw_command: float,
        step: int,
        t: float,
        arc_yaw_scale: float,
        heading_error: float = 0.0,
        yaw_rate: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        base = self.base_action(
            forward_command,
            yaw_command,
            t,
            heading_error=heading_error,
            yaw_rate=yaw_rate,
        )
        yaw = self.yaw_teacher_action(yaw_command, step)
        forward_mag = abs(float(forward_command))
        yaw_mag = abs(float(yaw_command))
        if forward_mag < 0.05 and yaw_mag < 0.05:
            target = np.zeros(3, dtype=np.float32)
        elif yaw_mag < 0.05:
            target = base
        elif forward_mag < 0.05:
            target = yaw
        else:
            target = np.clip(base + arc_yaw_scale * yaw, -1.0, 1.0).astype(np.float32)
        return base, yaw, target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a single CPG-parameter modulator for Bram."
    )
    parser.add_argument("--forward-gait", type=Path, default=DEFAULT_FORWARD_GAIT)
    parser.add_argument("--backward-gait", type=Path, default=DEFAULT_BACKWARD_GAIT)
    parser.add_argument("--yaw-left-table", type=Path, default=DEFAULT_YAW_LEFT_TABLE)
    parser.add_argument("--yaw-right-table", type=Path, default=DEFAULT_YAW_RIGHT_TABLE)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--teacher-controller",
        action="store_true",
        help="Evaluate/view the exact CPG + yaw-table component controller.",
    )
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--episode-seconds", type=float, default=8.0)
    parser.add_argument("--dataset-repeats", type=int, default=1)
    parser.add_argument(
        "--base-scaling",
        choices=("linear", "gait-speed"),
        default="gait-speed",
    )
    parser.add_argument("--base-speed-min", type=float, default=0.35)
    parser.add_argument("--base-action-min", type=float, default=0.60)
    parser.add_argument("--arc-yaw-scale", type=float, default=0.65)
    parser.add_argument("--action-residual-limit", type=float, default=0.55)
    parser.add_argument("--phase-harmonics", type=int, default=4)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--view", action="store_true")
    parser.add_argument(
        "--view-command",
        choices=("suite", "primitives") + tuple(command[0] for command in COMMAND_SUITE),
        default="primitives",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    return parser.parse_args()


def load_action_table(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text())
    if "actions" not in payload:
        raise ValueError(f"{path} does not contain an actions table.")
    return np.asarray(payload["actions"], dtype=np.float32)


def table_action(table: np.ndarray, step: int) -> np.ndarray:
    if len(table) == 0:
        return np.zeros(3, dtype=np.float32)
    return np.asarray(table[step % len(table)], dtype=np.float32)


def command_options(forward: float, yaw: float) -> dict[str, float]:
    return {"forward_command": float(forward), "yaw_rate_command": float(yaw)}


def command_grid() -> tuple[tuple[str, float, float], ...]:
    commands: list[tuple[str, float, float]] = []
    for forward in TRAIN_FORWARD_VALUES:
        for yaw in TRAIN_YAW_VALUES:
            if abs(forward) < 1e-6 and abs(yaw) < 1e-6:
                name = "idle"
            else:
                name = f"f{forward:+.2f}_y{yaw:+.2f}"
            commands.append((name, float(forward), float(yaw)))
    return tuple(commands)


def model_input(
    base_action: np.ndarray,
    previous_action: np.ndarray,
    forward_command: float,
    yaw_command: float,
    t: float,
    base_frequency: float,
    episode_seconds: float,
    phase_harmonics: int,
) -> np.ndarray:
    forward = float(forward_command)
    yaw = float(yaw_command)
    features: list[float] = [
        forward,
        yaw,
        abs(forward),
        abs(yaw),
        forward * yaw,
    ]
    features.extend(float(value) for value in base_action)
    gait_phase = 2.0 * np.pi * base_frequency * t
    episode_phase = 2.0 * np.pi * t / max(episode_seconds, 1e-6)
    for harmonic in range(1, phase_harmonics + 1):
        features.append(float(np.sin(harmonic * gait_phase)))
        features.append(float(np.cos(harmonic * gait_phase)))
    for harmonic in range(1, phase_harmonics + 1):
        features.append(float(np.sin(harmonic * episode_phase)))
        features.append(float(np.cos(harmonic * episode_phase)))
    return np.asarray(features, dtype=np.float32)


def build_dataset(
    library: TeacherLibrary,
    args: argparse.Namespace,
) -> tuple[TensorDataset, dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[float] = []
    base_params: list[np.ndarray] = []
    base_actions: list[np.ndarray] = []
    times: list[float] = []
    yaw_gates: list[float] = []
    command_counts: dict[str, int] = {}
    commands = command_grid()

    for repeat in range(args.dataset_repeats):
        for command_index, (name, forward, yaw) in enumerate(commands):
            env = BramTripodEnv(
                episode_seconds=args.episode_seconds,
                randomize_reset=False,
                domain_randomization=False,
                randomize_command=False,
                command_forward=forward,
                command_yaw_rate=yaw,
            )
            env.reset(
                seed=args.seed + 1000 * repeat + command_index,
                options=command_options(forward, yaw),
            )
            previous_action = np.zeros(3, dtype=np.float32)
            heading_error = 0.0
            yaw_rate = 0.0
            for step in range(env.max_steps):
                t = step * env.dt
                params, _, _ = library.scaled_params(forward, yaw)
                base, _, target = library.teacher_action(
                    forward,
                    yaw,
                    step,
                    t,
                    args.arc_yaw_scale,
                    heading_error=heading_error,
                    yaw_rate=yaw_rate,
                )
                inputs.append(
                    model_input(
                        base,
                        previous_action,
                        forward,
                        yaw,
                        t,
                        float(params[0]),
                        args.episode_seconds,
                        args.phase_harmonics,
                    )
                )
                targets.append(target.astype(np.float32))
                weights.append(command_weight(forward, yaw))
                base_params.append(params.astype(np.float32))
                base_actions.append(base.astype(np.float32))
                times.append(float(t))
                yaw_gates.append(float(np.clip(abs(yaw), 0.0, 1.0)))
                command_counts[name] = command_counts.get(name, 0) + 1
                _, _, terminated, truncated, info = env.step(target)
                heading_error = float(info.get("heading_error", 0.0))
                yaw_rate = float(info.get("yaw_rate", 0.0))
                previous_action = target
                if terminated or truncated:
                    break
            env.close()

    order = rng.permutation(len(inputs))
    x = torch.as_tensor(np.asarray(inputs, dtype=np.float32)[order])
    y = torch.as_tensor(np.asarray(targets, dtype=np.float32)[order])
    w = torch.as_tensor(np.asarray(weights, dtype=np.float32)[order, None])
    params = torch.as_tensor(np.asarray(base_params, dtype=np.float32)[order])
    base = torch.as_tensor(np.asarray(base_actions, dtype=np.float32)[order])
    t_tensor = torch.as_tensor(np.asarray(times, dtype=np.float32)[order, None])
    gate = torch.as_tensor(np.asarray(yaw_gates, dtype=np.float32)[order, None])
    dataset = TensorDataset(x, y, w, params, base, t_tensor, gate)
    meta = {
        "samples": int(len(inputs)),
        "input_dim": int(x.shape[1]),
        "target_dim": int(y.shape[1]),
        "commands": len(commands),
        "command_counts": command_counts,
    }
    return dataset, meta


def command_weight(forward: float, yaw: float) -> float:
    forward_mag = abs(float(forward))
    yaw_mag = abs(float(yaw))
    if forward_mag < 1e-6 and yaw_mag < 1e-6:
        return 3.0
    if yaw_mag < 1e-6:
        return 1.0
    if forward_mag < 1e-6:
        return 4.0
    return 1.4


def cpg_action_torch(
    model: CpgModulator,
    x: torch.Tensor,
    params: torch.Tensor,
    base_action: torch.Tensor,
    t: torch.Tensor,
    yaw_gate: torch.Tensor,
    action_residual_limit: float,
) -> torch.Tensor:
    raw = model(x)
    param_raw = raw[:, : len(PARAM_MOD_LIMITS)]
    action_raw = raw[:, len(PARAM_MOD_LIMITS) :]
    limits = PARAM_MOD_LIMITS.to(device=x.device, dtype=x.dtype)
    delta = param_raw * limits * yaw_gate

    frequency = torch.clamp(params[:, 0:1] + delta[:, 0:1], 0.20, 3.00)
    center = params[:, 1:4] + delta[:, 1:4]
    amplitude = torch.clamp(params[:, 4:7] + delta[:, 4:7], 0.0, 1.20)
    phase = params[:, 7:10] + delta[:, 7:10]
    harmonic = params[:, 10:13] + delta[:, 10:13]
    harmonic_phase = params[:, 18:21] + delta[:, 13:16]

    theta = 2.0 * torch.pi * frequency * t + phase
    unmodulated_theta = 2.0 * torch.pi * params[:, 0:1] * t + params[:, 7:10]
    unmodulated = (
        params[:, 1:4]
        + params[:, 4:7] * torch.sin(unmodulated_theta)
        + params[:, 10:13]
        * torch.sin(2.0 * unmodulated_theta + params[:, 18:21])
    )
    modulated = (
        center
        + amplitude * torch.sin(theta)
        + harmonic * torch.sin(2.0 * theta + harmonic_phase)
    )
    residual = float(action_residual_limit) * yaw_gate * action_raw
    action = base_action + yaw_gate * (modulated - unmodulated) + residual
    return torch.clamp(action, -1.0, 1.0)


def train_model(
    dataset: TensorDataset,
    input_dim: int,
    args: argparse.Namespace,
) -> tuple[CpgModulator, list[dict[str, float]]]:
    torch.manual_seed(args.seed)
    model = CpgModulator(input_dim, args.hidden_size)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        losses = []
        action_errors = []
        for (
            batch_x,
            batch_y,
            batch_w,
            batch_params,
            batch_base,
            batch_t,
            batch_gate,
        ) in loader:
            pred = cpg_action_torch(
                model,
                batch_x,
                batch_params,
                batch_base,
                batch_t,
                batch_gate,
                args.action_residual_limit,
            )
            mse = torch.square(pred - batch_y)
            loss = torch.mean(batch_w * mse)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            action_errors.append(float(torch.mean(torch.abs(pred - batch_y)).detach()))
        if epoch == 1 or epoch % max(1, args.epochs // 13) == 0 or epoch == args.epochs:
            row = {
                "epoch": float(epoch),
                "train_loss": float(np.mean(losses)),
                "mean_abs_action_error": float(np.mean(action_errors)),
            }
            history.append(row)
            print(
                f"epoch={epoch:04d}/{args.epochs:04d} "
                f"loss={row['train_loss']:.6f} "
                f"mae={row['mean_abs_action_error']:.4f}",
                flush=True,
            )
    model.eval()
    return model, history


def cpg_modulated_action(
    model: CpgModulator,
    library: TeacherLibrary,
    base_action: np.ndarray,
    previous_action: np.ndarray,
    forward_command: float,
    yaw_command: float,
    t: float,
    episode_seconds: float,
    phase_harmonics: int,
    action_residual_limit: float,
) -> np.ndarray:
    params, _, _ = library.scaled_params(forward_command, yaw_command)
    features = model_input(
        base_action,
        previous_action,
        forward_command,
        yaw_command,
        t,
        float(params[0]),
        episode_seconds,
        phase_harmonics,
    )
    x = torch.as_tensor(features[None, :], dtype=torch.float32)
    params_tensor = torch.as_tensor(params[None, :], dtype=torch.float32)
    t_tensor = torch.as_tensor([[float(t)]], dtype=torch.float32)
    base_tensor = torch.as_tensor(base_action[None, :], dtype=torch.float32)
    gate = torch.as_tensor(
        [[float(np.clip(abs(yaw_command), 0.0, 1.0))]], dtype=torch.float32
    )
    with torch.no_grad():
        action = cpg_action_torch(
            model,
            x,
            params_tensor,
            base_tensor,
            t_tensor,
            gate,
            action_residual_limit,
        )
    return action.cpu().numpy()[0].astype(np.float32)


def evaluate_model(
    model: CpgModulator,
    library: TeacherLibrary,
    args: argparse.Namespace,
) -> list[EvalResult]:
    results: list[EvalResult] = []
    for name, forward, yaw in COMMAND_SUITE:
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
            env.reset(
                seed=args.seed + 50_000 + 1000 * episode,
                options=command_options(forward, yaw),
            )
            previous_action = np.zeros(3, dtype=np.float32)
            total_reward = 0.0
            final_info: dict[str, Any] = {}
            terminated = False
            truncated = False
            heading_error = 0.0
            yaw_rate = 0.0
            for step in range(env.max_steps):
                t = step * env.dt
                base = library.base_action(
                    forward,
                    yaw,
                    t,
                    heading_error=heading_error,
                    yaw_rate=yaw_rate,
                )
                action = cpg_modulated_action(
                    model,
                    library,
                    base,
                    previous_action,
                    forward,
                    yaw,
                    t,
                    args.episode_seconds,
                    args.phase_harmonics,
                    args.action_residual_limit,
                )
                _, reward, terminated, truncated, final_info = env.step(action)
                heading_error = float(final_info.get("heading_error", 0.0))
                yaw_rate = float(final_info.get("yaw_rate", 0.0))
                previous_action = action
                total_reward += reward
                if terminated or truncated:
                    break
            env.close()
            episode_results.append(
                result_from_info(name, total_reward, final_info, step + 1, terminated)
            )
        results.append(mean_result(episode_results))
    return results


def evaluate_teacher_controller(
    library: TeacherLibrary,
    args: argparse.Namespace,
) -> list[EvalResult]:
    results: list[EvalResult] = []
    for name, forward, yaw in COMMAND_SUITE:
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
            env.reset(
                seed=args.seed + 70_000 + 1000 * episode,
                options=command_options(forward, yaw),
            )
            total_reward = 0.0
            final_info: dict[str, Any] = {}
            terminated = False
            truncated = False
            heading_error = 0.0
            yaw_rate = 0.0
            for step in range(env.max_steps):
                t = step * env.dt
                _, _, action = library.teacher_action(
                    forward,
                    yaw,
                    step,
                    t,
                    args.arc_yaw_scale,
                    heading_error=heading_error,
                    yaw_rate=yaw_rate,
                )
                _, reward, terminated, truncated, final_info = env.step(action)
                heading_error = float(final_info.get("heading_error", 0.0))
                yaw_rate = float(final_info.get("yaw_rate", 0.0))
                total_reward += reward
                if terminated or truncated:
                    break
            env.close()
            episode_results.append(
                result_from_info(name, total_reward, final_info, step + 1, terminated)
            )
        results.append(mean_result(episode_results))
    return results


def result_from_info(
    name: str,
    reward: float,
    info: dict[str, Any],
    length: int,
    terminated: bool,
) -> EvalResult:
    return EvalResult(
        command=name,
        reward=float(reward),
        command_distance=float(info.get("command_distance", 0.0)),
        line_distance=float(info.get("line_distance", 0.0)),
        yaw_distance=float(info.get("yaw_distance", 0.0)),
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


def save_outputs(
    model: CpgModulator,
    history: list[dict[str, float]],
    eval_results: list[EvalResult],
    dataset_meta: dict[str, Any],
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": model.net[0].in_features,
        "hidden_size": args.hidden_size,
        "param_mod_limits": [float(value) for value in PARAM_MOD_LIMITS],
        "action_residual_limit": args.action_residual_limit,
        "param_names": PARAM_NAMES,
        "param_low": [float(value) for value in PARAM_LOW],
        "param_high": [float(value) for value in PARAM_HIGH],
        "args": vars(args),
        "dataset": dataset_meta,
        "history": history,
        "eval": [asdict(result) for result in eval_results],
    }
    torch.save(checkpoint, out_dir / "cpg_modulator.pt")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "checkpoint": str(out_dir / "cpg_modulator.pt"),
                "dataset": dataset_meta,
                "history": history,
                "eval": [asdict(result) for result in eval_results],
            },
            indent=2,
        )
        + "\n"
    )
    with (out_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(eval_results[0]).keys()))
        writer.writeheader()
        for result in eval_results:
            writer.writerow(asdict(result))
    with (out_dir / "training_history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "mean_abs_action_error"]
        )
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def load_checkpoint(path: Path) -> tuple[CpgModulator, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    model = CpgModulator(int(payload["input_dim"]), int(payload["hidden_size"]))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def apply_checkpoint_args(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    saved_args = payload.get("args", {})
    for key in (
        "base_scaling",
        "base_speed_min",
        "base_action_min",
        "arc_yaw_scale",
        "phase_harmonics",
        "episode_seconds",
        "action_residual_limit",
    ):
        if key in saved_args:
            setattr(args, key, saved_args[key])


def maybe_relaunch_with_mjpython(args: argparse.Namespace) -> None:
    if not args.view or args.headless or platform.system() != "Darwin":
        return
    if Path(sys.executable).name == "mjpython" or os.environ.get("MJPYTHON_BIN"):
        return
    mjpython = Path(sys.executable).with_name("mjpython")
    if mjpython.exists():
        os.execv(str(mjpython), [str(mjpython), *sys.argv])


def run_viewer(
    model: CpgModulator,
    library: TeacherLibrary,
    args: argparse.Namespace,
) -> None:
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
    env.reset(seed=args.seed, options=command_options(forward, yaw))
    step = 0
    previous_action = np.zeros(3, dtype=np.float32)
    heading_error = 0.0
    yaw_rate = 0.0
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            started = time.perf_counter()
            t = step * env.dt
            base = library.base_action(
                forward,
                yaw,
                t,
                heading_error=heading_error,
                yaw_rate=yaw_rate,
            )
            action = cpg_modulated_action(
                model,
                library,
                base,
                previous_action,
                forward,
                yaw,
                t,
                args.episode_seconds,
                args.phase_harmonics,
                args.action_residual_limit,
            )
            _, _, terminated, truncated, info = env.step(action)
            heading_error = float(info.get("heading_error", 0.0))
            yaw_rate = float(info.get("yaw_rate", 0.0))
            previous_action = action
            viewer.sync()
            step += 1
            if terminated or truncated:
                command_index = (command_index + 1) % len(commands)
                name, forward, yaw = commands[command_index]
                print(f"viewer_command={name} forward={forward:.2f} yaw={yaw:.2f}")
                env.reset(
                    seed=args.seed + command_index,
                    options=command_options(forward, yaw),
                )
                step = 0
                previous_action = np.zeros(3, dtype=np.float32)
                heading_error = 0.0
                yaw_rate = 0.0
                time.sleep(0.4)
            sleep_time = (env.dt / max(args.speed, 1e-6)) - (
                time.perf_counter() - started
            )
            if sleep_time > 0:
                time.sleep(sleep_time)
    env.close()


def run_teacher_viewer(library: TeacherLibrary, args: argparse.Namespace) -> None:
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
    env.reset(seed=args.seed, options=command_options(forward, yaw))
    step = 0
    heading_error = 0.0
    yaw_rate = 0.0
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            started = time.perf_counter()
            t = step * env.dt
            _, _, action = library.teacher_action(
                forward,
                yaw,
                step,
                t,
                args.arc_yaw_scale,
                heading_error=heading_error,
                yaw_rate=yaw_rate,
            )
            _, _, terminated, truncated, info = env.step(action)
            heading_error = float(info.get("heading_error", 0.0))
            yaw_rate = float(info.get("yaw_rate", 0.0))
            viewer.sync()
            step += 1
            if terminated or truncated:
                command_index = (command_index + 1) % len(commands)
                name, forward, yaw = commands[command_index]
                print(f"viewer_command={name} forward={forward:.2f} yaw={yaw:.2f}")
                env.reset(
                    seed=args.seed + command_index,
                    options=command_options(forward, yaw),
                )
                step = 0
                heading_error = 0.0
                yaw_rate = 0.0
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
    if view_command == "primitives":
        return PRIMITIVE_SUITE
    for command in COMMAND_SUITE:
        if command[0] == view_command:
            return (command,)
    raise ValueError(f"Unknown viewer command: {view_command}")


def print_eval(results: list[EvalResult]) -> None:
    for result in results:
        print(
            f"{result.command:7s} "
            f"cmd={result.command_distance: .4f} "
            f"line={result.line_distance: .4f} "
            f"yaw={result.yaw_distance: .4f} "
            f"xy=({result.x_distance: .3f},{result.y_distance: .3f}) "
            f"cross={result.cross_track_error: .3f} "
            f"heading={result.heading_error: .3f} "
            f"len={result.length} term={result.terminated}"
        )


def main() -> None:
    args = parse_args()
    maybe_relaunch_with_mjpython(args)
    out_dir = args.out_dir or Path("runs") / (
        "cpg_modulator_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    model = None
    payload = None
    if args.checkpoint is not None:
        model, payload = load_checkpoint(args.checkpoint)
        apply_checkpoint_args(args, payload)

    library = TeacherLibrary(
        args.forward_gait,
        args.backward_gait,
        args.yaw_left_table,
        args.yaw_right_table,
        args.base_scaling,
        args.base_speed_min,
        args.base_action_min,
    )

    if args.teacher_controller:
        results = evaluate_teacher_controller(library, args)
        print_eval(results)
        if args.view:
            run_teacher_viewer(library, args)
        return

    if args.checkpoint is not None:
        assert model is not None
        results = evaluate_model(model, library, args)
        print_eval(results)
        if args.view:
            run_viewer(model, library, args)
        return

    dataset, dataset_meta = build_dataset(library, args)
    print(
        f"dataset samples={dataset_meta['samples']} "
        f"commands={dataset_meta['commands']} "
        f"input_dim={dataset_meta['input_dim']}",
        flush=True,
    )
    model, history = train_model(dataset, dataset_meta["input_dim"], args)
    results = evaluate_model(model, library, args)
    print_eval(results)
    save_outputs(model, history, results, dataset_meta, out_dir, args)
    print(f"saved={out_dir}")
    if args.view:
        run_viewer(model, library, args)


if __name__ == "__main__":
    main()
