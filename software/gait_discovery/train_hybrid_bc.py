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
from search_gait import gait_action, load_params


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_PUSHED_RUN = MODULE_DIR / "pushed_runs" / "current_policy_20260701"
DEFAULT_FORWARD_GAIT = DEFAULT_PUSHED_RUN / "gaits" / "forward_best_params.json"
DEFAULT_BACKWARD_GAIT = DEFAULT_PUSHED_RUN / "gaits" / "backward_best_params.json"
DEFAULT_YAW_LEFT_GAIT = None
DEFAULT_YAW_RIGHT_TABLE = DEFAULT_PUSHED_RUN / "yaw_tables" / "yaw_right_policy_table.json"
DEFAULT_YAW_LEFT_TABLE = DEFAULT_PUSHED_RUN / "yaw_tables" / "yaw_left_policy_table.json"

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


class ResidualModulator(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 48) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))


class TeacherLibrary:
    def __init__(
        self,
        forward_gait: Path,
        backward_gait: Path,
        yaw_left_gait: Path | None,
        yaw_left_table: Path | None,
        yaw_right_table: Path | None,
        residual_limit: float,
        base_scaling: str,
        base_speed_min: float,
        base_action_min: float,
    ) -> None:
        _, self.forward_params = load_params(forward_gait)
        _, self.backward_params = load_params(backward_gait)
        self.yaw_left_params = None
        if yaw_left_gait is not None and yaw_left_gait.exists():
            _, self.yaw_left_params = load_params(yaw_left_gait)
        self.yaw_left_table = load_action_table(yaw_left_table) if yaw_left_table else None
        self.yaw_right_table = (
            load_action_table(yaw_right_table) if yaw_right_table else None
        )
        self.residual_limit = float(residual_limit)
        self.base_scaling = base_scaling
        self.base_speed_min = float(base_speed_min)
        self.base_action_min = float(base_action_min)

    def base_action(self, forward_command: float, t: float) -> np.ndarray:
        magnitude = abs(float(forward_command))
        if magnitude < 0.05:
            return np.zeros(3, dtype=np.float32)
        if self.base_scaling == "linear":
            speed_scale = 1.0
            action_scale = magnitude
        else:
            speed_scale = self.base_speed_min + (1.0 - self.base_speed_min) * magnitude
            action_scale = self.base_action_min + (1.0 - self.base_action_min) * magnitude
        if forward_command > 0.0:
            action = gait_action(
                self.forward_params,
                t * speed_scale,
                use_heading_correction=False,
            )
        else:
            action = gait_action(
                self.backward_params,
                t * speed_scale,
                use_heading_correction=False,
            )
        return np.clip(action_scale * action, -1.0, 1.0).astype(np.float32)

    def yaw_teacher_action(self, yaw_command: float, step: int, t: float) -> np.ndarray:
        magnitude = abs(float(yaw_command))
        if magnitude < 1e-6:
            return np.zeros(3, dtype=np.float32)
        if yaw_command < 0.0:
            action = table_action(self.yaw_right_table, step)
        else:
            action = self._yaw_left_action(step, t)
        return np.clip(magnitude * action, -1.0, 1.0).astype(np.float32)

    def _yaw_left_action(self, step: int, t: float) -> np.ndarray:
        if self.yaw_left_table is not None:
            return table_action(self.yaw_left_table, step)
        if self.yaw_left_params is not None:
            return gait_action(self.yaw_left_params, t, use_heading_correction=False)
        return np.zeros(3, dtype=np.float32)

    def teacher_action(
        self,
        forward_command: float,
        yaw_command: float,
        step: int,
        t: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        base = self.base_action(forward_command, t)
        yaw = self.yaw_teacher_action(yaw_command, step, t)
        target = np.clip(base + yaw, -1.0, 1.0).astype(np.float32)
        return base, yaw, target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a conservative hybrid CPG + learned residual by behavior cloning."
    )
    parser.add_argument("--forward-gait", type=Path, default=DEFAULT_FORWARD_GAIT)
    parser.add_argument("--backward-gait", type=Path, default=DEFAULT_BACKWARD_GAIT)
    parser.add_argument("--yaw-left-gait", type=Path, default=DEFAULT_YAW_LEFT_GAIT)
    parser.add_argument("--yaw-left-table", type=Path, default=None)
    parser.add_argument("--yaw-right-table", type=Path, default=DEFAULT_YAW_RIGHT_TABLE)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--episode-seconds", type=float, default=4.0)
    parser.add_argument("--dataset-repeats", type=int, default=2)
    parser.add_argument("--residual-limit", type=float, default=0.70)
    parser.add_argument(
        "--base-scaling",
        choices=("linear", "gait-speed"),
        default="linear",
        help="Scale forward/back CPG by amplitude only, or preserve stride while scaling phase speed.",
    )
    parser.add_argument("--base-speed-min", type=float, default=0.35)
    parser.add_argument("--base-action-min", type=float, default=0.60)
    parser.add_argument(
        "--feature-mode",
        choices=("phase", "obs"),
        default="phase",
        help="Use compact phase/command features or full env obs plus base action.",
    )
    parser.add_argument("--phase-harmonics", type=int, default=4)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument(
        "--direct-pure-yaw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bypass the residual net for pure yaw and use the best direct yaw teacher.",
    )
    parser.add_argument(
        "--direct-pure-yaw-threshold",
        type=float,
        default=0.05,
        help="Maximum absolute forward command considered pure yaw.",
    )
    parser.add_argument("--view", action="store_true")
    parser.add_argument(
        "--view-command",
        choices=("suite",) + tuple(command[0] for command in COMMAND_SUITE),
        default="suite",
        help="Viewer command to run, or cycle the full command suite.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    return parser.parse_args()


def load_action_table(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    payload = json.loads(path.read_text())
    return np.asarray(payload["actions"], dtype=np.float32)


def table_action(table: np.ndarray | None, step: int) -> np.ndarray:
    if table is None or len(table) == 0:
        return np.zeros(3, dtype=np.float32)
    return np.asarray(table[step % len(table)], dtype=np.float32)


def command_options(forward: float, yaw: float) -> dict[str, float]:
    return {"forward_command": float(forward), "yaw_rate_command": float(yaw)}


def model_input(
    obs: np.ndarray,
    base_action: np.ndarray,
    previous_action: np.ndarray,
    forward_command: float,
    yaw_command: float,
    t: float,
    episode_seconds: float,
    args: argparse.Namespace,
) -> np.ndarray:
    if args.feature_mode == "obs":
        return np.concatenate(
            [
                obs.astype(np.float32),
                base_action.astype(np.float32),
                previous_action.astype(np.float32),
            ]
        )

    features: list[float] = [
        float(forward_command),
        float(yaw_command),
        abs(float(forward_command)),
        abs(float(yaw_command)),
        float(forward_command * yaw_command),
    ]
    features.extend(float(value) for value in base_action)
    features.extend(float(value) for value in previous_action)
    gait_phase = 2.0 * np.pi * 1.4 * t
    episode_phase = 2.0 * np.pi * t / max(episode_seconds, 1e-6)
    for harmonic in range(1, args.phase_harmonics + 1):
        features.append(float(np.sin(harmonic * gait_phase)))
        features.append(float(np.cos(harmonic * gait_phase)))
    for harmonic in range(1, args.phase_harmonics + 1):
        features.append(float(np.sin(harmonic * episode_phase)))
        features.append(float(np.cos(harmonic * episode_phase)))
    return np.asarray(features, dtype=np.float32)


def residual_gate(yaw_command: float) -> float:
    return float(np.clip(abs(yaw_command), 0.0, 1.0))


def build_dataset(
    library: TeacherLibrary,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[float] = []
    command_counts: dict[str, int] = {}

    for repeat in range(args.dataset_repeats):
        for command_index, (name, forward, yaw) in enumerate(COMMAND_SUITE):
            env = BramTripodEnv(
                episode_seconds=args.episode_seconds,
                randomize_reset=False,
                domain_randomization=False,
                randomize_command=False,
                command_forward=forward,
                command_yaw_rate=yaw,
            )
            obs, _ = env.reset(
                seed=args.seed + 1000 * repeat + command_index,
                options=command_options(forward, yaw),
            )
            previous_action = np.zeros(3, dtype=np.float32)
            for step in range(env.max_steps):
                t = step * env.dt
                base, _, target = library.teacher_action(forward, yaw, step, t)
                gate = residual_gate(yaw)
                if gate > 1e-6:
                    residual_target = (target - base) / (args.residual_limit * gate)
                    residual_target = np.clip(residual_target, -1.0, 1.0)
                else:
                    residual_target = np.zeros(3, dtype=np.float32)
                inputs.append(
                    model_input(
                        obs,
                        base,
                        previous_action,
                        forward,
                        yaw,
                        t,
                        args.episode_seconds,
                        args,
                    )
                )
                targets.append(residual_target.astype(np.float32))
                weights.append(command_weight(forward, yaw))
                command_counts[name] = command_counts.get(name, 0) + 1
                obs, _, terminated, truncated, _ = env.step(target)
                previous_action = target
                if terminated or truncated:
                    break
            env.close()

    order = rng.permutation(len(inputs))
    x = torch.as_tensor(np.asarray(inputs, dtype=np.float32)[order])
    y = torch.as_tensor(np.asarray(targets, dtype=np.float32)[order])
    w = torch.as_tensor(np.asarray(weights, dtype=np.float32)[order, None])
    meta = {
        "samples": int(len(inputs)),
        "input_dim": int(x.shape[1]),
        "target_dim": int(y.shape[1]),
        "command_counts": command_counts,
    }
    return x, y, {"weights": w, **meta}


def command_weight(forward: float, yaw: float) -> float:
    if abs(forward) < 1e-6 and abs(yaw) < 1e-6:
        return 1.7
    if abs(yaw) < 1e-6:
        return 1.5
    if abs(forward) < 1e-6:
        if yaw < 0.0:
            return 1.9
        return 1.2
    if yaw < 0.0:
        return 1.3
    return 1.0


def train_model(
    x: torch.Tensor,
    y: torch.Tensor,
    weights: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[ResidualModulator, list[dict[str, float]]]:
    torch.manual_seed(args.seed)
    model = ResidualModulator(x.shape[1], args.hidden_size)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    dataset = TensorDataset(x, y, weights)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        losses = []
        for batch_x, batch_y, batch_w in loader:
            pred = model(batch_x)
            mse = torch.square(pred - batch_y)
            loss = torch.mean(batch_w * mse)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0:
            with torch.no_grad():
                full_pred = model(x)
                full_loss = torch.mean(weights * torch.square(full_pred - y)).item()
                max_err = torch.max(torch.abs(full_pred - y)).item()
            row = {
                "epoch": float(epoch),
                "train_loss": float(np.mean(losses)),
                "full_loss": float(full_loss),
                "max_abs_error": float(max_err),
            }
            history.append(row)
            print(
                f"epoch={epoch:04d}/{args.epochs:04d} "
                f"loss={row['full_loss']:.6f} max_err={row['max_abs_error']:.3f}"
            )
    return model, history


def hybrid_action(
    model: ResidualModulator,
    library: TeacherLibrary,
    obs: np.ndarray,
    base_action: np.ndarray,
    previous_action: np.ndarray,
    forward_command: float,
    yaw_command: float,
    step: int,
    t: float,
    episode_seconds: float,
    residual_limit: float,
    args: argparse.Namespace,
) -> np.ndarray:
    if (
        args.direct_pure_yaw
        and abs(forward_command) <= args.direct_pure_yaw_threshold
        and abs(yaw_command) > 1e-6
    ):
        return library.yaw_teacher_action(yaw_command, step, t)
    if abs(yaw_command) < 1e-6:
        return base_action.astype(np.float32)
    features = model_input(
        obs,
        base_action,
        previous_action,
        forward_command,
        yaw_command,
        t,
        episode_seconds,
        args,
    )
    x = torch.as_tensor(features[None, :], dtype=torch.float32)
    with torch.no_grad():
        residual = model(x).cpu().numpy()[0].astype(np.float32)
    action = base_action + residual_limit * residual_gate(yaw_command) * residual
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def evaluate_model(
    model: ResidualModulator,
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
            obs, _ = env.reset(
                seed=args.seed + 50_000 + 1000 * episode,
                options=command_options(forward, yaw),
            )
            previous_action = np.zeros(3, dtype=np.float32)
            total_reward = 0.0
            final_info: dict[str, Any] = {}
            terminated = False
            truncated = False
            for step in range(env.max_steps):
                t = step * env.dt
                base = library.base_action(forward, t)
                action = hybrid_action(
                    model,
                    library,
                    obs,
                    base,
                    previous_action,
                    forward,
                    yaw,
                    step,
                    t,
                    args.episode_seconds,
                    args.residual_limit,
                    args,
                )
                obs, reward, terminated, truncated, final_info = env.step(action)
                previous_action = action
                total_reward += reward
                if terminated or truncated:
                    break
            env.close()
            episode_results.append(
                result_from_info(
                    name,
                    total_reward,
                    final_info,
                    step + 1,
                    terminated,
                )
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
    model: ResidualModulator,
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
        "residual_limit": args.residual_limit,
        "args": vars(args),
        "feature_mode": args.feature_mode,
        "phase_harmonics": args.phase_harmonics,
        "dataset": dataset_meta,
        "eval": [asdict(result) for result in eval_results],
    }
    torch.save(checkpoint, out_dir / "hybrid_bc.pt")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "checkpoint": str(out_dir / "hybrid_bc.pt"),
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
            f, fieldnames=["epoch", "train_loss", "full_loss", "max_abs_error"]
        )
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def load_checkpoint(path: Path) -> tuple[ResidualModulator, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    model = ResidualModulator(int(payload["input_dim"]), int(payload["hidden_size"]))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def apply_checkpoint_args(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    saved_args = payload.get("args", {})
    for key in (
        "residual_limit",
        "feature_mode",
        "phase_harmonics",
        "base_scaling",
        "base_speed_min",
        "base_action_min",
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
    model: ResidualModulator,
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
    obs, _ = env.reset(seed=args.seed, options=command_options(forward, yaw))
    step = 0
    previous_action = np.zeros(3, dtype=np.float32)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            started = time.perf_counter()
            t = step * env.dt
            base = library.base_action(forward, t)
            action = hybrid_action(
                model,
                library,
                obs,
                base,
                previous_action,
                forward,
                yaw,
                step,
                t,
                args.episode_seconds,
                args.residual_limit,
                args,
            )
            obs, _, terminated, truncated, _ = env.step(action)
            previous_action = action
            viewer.sync()
            step += 1
            if terminated or truncated:
                command_index = (command_index + 1) % len(commands)
                name, forward, yaw = commands[command_index]
                print(f"viewer_command={name} forward={forward:.2f} yaw={yaw:.2f}")
                obs, _ = env.reset(
                    seed=args.seed + command_index,
                    options=command_options(forward, yaw),
                )
                step = 0
                previous_action = np.zeros(3, dtype=np.float32)
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
        "hybrid_bc_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    model = None
    payload = None
    if args.checkpoint is not None:
        model, payload = load_checkpoint(args.checkpoint)
        apply_checkpoint_args(args, payload)
    library = TeacherLibrary(
        args.forward_gait,
        args.backward_gait,
        args.yaw_left_gait,
        args.yaw_left_table,
        args.yaw_right_table,
        args.residual_limit,
        args.base_scaling,
        args.base_speed_min,
        args.base_action_min,
    )

    if args.checkpoint is not None:
        assert model is not None
        results = evaluate_model(model, library, args)
        print_eval(results)
        if args.view:
            run_viewer(model, library, args)
        return

    x, y, dataset_meta = build_dataset(library, args)
    weights = dataset_meta.pop("weights")
    print(
        f"dataset samples={dataset_meta['samples']} "
        f"input_dim={dataset_meta['input_dim']} "
        f"residual_limit={args.residual_limit:.2f}"
    )
    model, history = train_model(x, y, weights, args)
    results = evaluate_model(model, library, args)
    print_eval(results)
    save_outputs(model, history, results, dataset_meta, out_dir, args)
    print(f"saved={out_dir}")
    if args.view:
        run_viewer(model, library, args)


if __name__ == "__main__":
    main()
