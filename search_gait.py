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

from bram_env import BramTripodEnv


PARAM_NAMES = (
    "frequency_hz",
    "center_front",
    "center_back_left",
    "center_back_right",
    "amplitude_front",
    "amplitude_back_left",
    "amplitude_back_right",
    "phase_front",
    "phase_back_left",
    "phase_back_right",
    "harmonic_front",
    "harmonic_back_left",
    "harmonic_back_right",
    "heading_kp",
    "yaw_kd",
    "turn_front",
    "turn_back_left",
    "turn_back_right",
    "harmonic_phase_front",
    "harmonic_phase_back_left",
    "harmonic_phase_back_right",
)

PARAM_LOW = np.array(
    [
        0.35,
        -0.45,
        -0.45,
        -0.45,
        0.02,
        0.02,
        0.02,
        -np.pi,
        -np.pi,
        -np.pi,
        -0.38,
        -0.38,
        -0.38,
        0.00,
        0.00,
        -1.00,
        -1.00,
        -1.00,
        -np.pi,
        -np.pi,
        -np.pi,
    ],
    dtype=np.float64,
)

PARAM_HIGH = np.array(
    [
        2.60,
        0.45,
        0.45,
        0.45,
        0.98,
        0.98,
        0.98,
        np.pi,
        np.pi,
        np.pi,
        0.38,
        0.38,
        0.38,
        0.95,
        0.45,
        1.00,
        1.00,
        1.00,
        np.pi,
        np.pi,
        np.pi,
    ],
    dtype=np.float64,
)

PHASE_SLICE = slice(7, 10)
HARMONIC_PHASE_SLICE = slice(18, 21)
HEADING_TRIM_LIMIT = 0.35


@dataclass(frozen=True)
class RolloutResult:
    score: float
    progress: float
    x_distance: float
    y_distance: float
    cross_track_error: float
    heading_error: float
    yaw_distance: float
    planar_drift: float
    mean_abs_cross_velocity: float
    mean_abs_yaw_rate: float
    mean_action_delta: float
    mean_action_accel: float
    mean_abs_action: float
    length: int
    terminated: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Black-box search for smooth sinusoidal Bram tripod gaits."
    )
    parser.add_argument(
        "--primitive",
        choices=("forward", "backward", "yaw-left", "yaw-right", "idle"),
        default="forward",
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--elite-frac", type=float, default=0.18)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--episode-seconds", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--params", type=Path, default=None)
    parser.add_argument("--init-params", type=Path, default=None)
    parser.add_argument(
        "--init-inverse",
        action="store_true",
        help="Allow initializing backward from a forward gait, or forward from backward, by phase-inverting the waveform.",
    )
    parser.add_argument("--init-std-scale", type=float, default=0.42)
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--domain-randomization-strength", type=float, default=0.25)
    parser.add_argument("--randomize-reset", action="store_true")
    parser.add_argument("--momentum", type=float, default=0.72)
    parser.add_argument("--min-std-frac", type=float, default=0.025)
    return parser.parse_args()


def maybe_relaunch_with_mjpython(args: argparse.Namespace) -> None:
    if not args.view or args.headless or platform.system() != "Darwin":
        return
    if Path(sys.executable).name == "mjpython" or os.environ.get("MJPYTHON_BIN"):
        return
    candidate = Path(sys.executable).with_name("mjpython")
    if not candidate.exists():
        return
    os.execv(str(candidate), [str(candidate), *sys.argv])


def command_for_primitive(primitive: str) -> tuple[float, float]:
    if primitive == "forward":
        return 1.0, 0.0
    if primitive == "backward":
        return -1.0, 0.0
    if primitive == "yaw-left":
        return 0.0, 1.0
    if primitive == "yaw-right":
        return 0.0, -1.0
    if primitive == "idle":
        return 0.0, 0.0
    raise ValueError(f"Unknown primitive: {primitive}")


def initial_mean() -> np.ndarray:
    mean = (PARAM_LOW + PARAM_HIGH) * 0.5
    mean[0] = 1.25
    mean[1:4] = 0.0
    mean[4:7] = 0.34
    mean[7:10] = np.array([0.0, 2.0 * np.pi / 3.0, -2.0 * np.pi / 3.0])
    mean[10:13] = 0.0
    mean[13] = 0.20
    mean[14] = 0.06
    mean[15:18] = np.array([0.0, 0.65, -0.65])
    mean[18:21] = 0.0
    return mean


def initial_std() -> np.ndarray:
    std = (PARAM_HIGH - PARAM_LOW) * 0.30
    std[0] = 0.45
    std[1:4] = 0.22
    std[4:7] = 0.28
    std[7:10] = 1.30
    std[10:13] = 0.18
    std[13] = 0.22
    std[14] = 0.12
    std[15:18] = 0.50
    std[18:21] = 1.20
    return std


def clip_params(params: np.ndarray) -> np.ndarray:
    params = normalize_params(params)
    clipped = np.clip(params, PARAM_LOW, PARAM_HIGH)
    clipped[PHASE_SLICE] = wrap_pi(clipped[PHASE_SLICE])
    clipped[HARMONIC_PHASE_SLICE] = wrap_pi(clipped[HARMONIC_PHASE_SLICE])
    return clipped


def inverse_translation_params(params: np.ndarray) -> np.ndarray:
    inverted = normalize_params(params).copy()
    inverted[PHASE_SLICE] = wrap_pi(inverted[PHASE_SLICE] + np.pi)
    inverted[10:13] *= -1.0
    inverted[13] = abs(inverted[13])
    inverted[14] = abs(inverted[14])
    return clip_params(inverted)


def normalize_params(params: np.ndarray) -> np.ndarray:
    params = np.asarray(params, dtype=np.float64)
    if params.shape[0] == len(PARAM_NAMES):
        return params
    if params.shape[0] > len(PARAM_NAMES):
        return params[: len(PARAM_NAMES)]
    padded = np.zeros(len(PARAM_NAMES), dtype=np.float64)
    padded[: params.shape[0]] = params
    return padded


def wrap_pi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def gait_action(
    params: np.ndarray,
    t: float,
    heading_error: float = 0.0,
    yaw_rate: float = 0.0,
    use_heading_correction: bool = True,
) -> np.ndarray:
    frequency = params[0]
    center = params[1:4]
    amplitude = params[4:7]
    phase = params[7:10]
    harmonic = params[10:13]
    harmonic_phase = params[18:21]
    theta = 2.0 * np.pi * frequency * t + phase
    action = (
        center
        + amplitude * np.sin(theta)
        + harmonic * np.sin(2.0 * theta + harmonic_phase)
    )
    if use_heading_correction:
        trim = -params[13] * heading_error - params[14] * yaw_rate
        trim = float(np.clip(trim, -HEADING_TRIM_LIMIT, HEADING_TRIM_LIMIT))
        action = action + trim * params[15:18]
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def make_env(args: argparse.Namespace) -> BramTripodEnv:
    forward_command, yaw_command = command_for_primitive(args.primitive)
    return BramTripodEnv(
        frame_skip=10,
        episode_seconds=args.episode_seconds,
        randomize_reset=args.randomize_reset,
        domain_randomization=args.domain_randomization,
        domain_randomization_strength=args.domain_randomization_strength,
        randomize_command=False,
        command_forward=forward_command,
        command_yaw_rate=yaw_command,
    )


def uses_heading_correction(primitive: str) -> bool:
    return primitive in ("forward", "backward")


def rollout(
    env: BramTripodEnv,
    params: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> RolloutResult:
    forward_command, yaw_command = command_for_primitive(args.primitive)
    obs, info = env.reset(
        seed=seed,
        options={
            "forward_command": forward_command,
            "yaw_rate_command": yaw_command,
            "randomize": args.randomize_reset,
        },
    )
    del obs, info

    action_prev = np.zeros(env.action_space.shape[0], dtype=np.float32)
    delta_prev = np.zeros_like(action_prev)
    abs_cross_velocity = []
    abs_yaw_rate = []
    action_deltas = []
    action_accels = []
    abs_actions = []
    final_info: dict[str, Any] = {}
    terminated = False
    truncated = False
    heading_error = 0.0
    yaw_rate = 0.0
    use_heading_correction = uses_heading_correction(args.primitive)

    for step in range(env.max_steps):
        t = step * env.dt
        action = gait_action(
            params,
            t,
            heading_error=heading_error,
            yaw_rate=yaw_rate,
            use_heading_correction=use_heading_correction,
        )
        delta = action - action_prev
        accel = delta - delta_prev
        _, _, terminated, truncated, final_info = env.step(action)
        heading_error = float(final_info.get("heading_error", 0.0))
        yaw_rate = float(final_info.get("yaw_rate", 0.0))

        abs_cross_velocity.append(abs(float(final_info.get("cross_track_velocity", 0.0))))
        abs_yaw_rate.append(abs(float(final_info.get("yaw_rate", 0.0))))
        action_deltas.append(float(np.mean(np.abs(delta))))
        action_accels.append(float(np.mean(np.abs(accel))))
        abs_actions.append(float(np.mean(np.abs(action))))

        action_prev = action
        delta_prev = delta
        if terminated or truncated:
            break

    result = result_from_info(
        primitive=args.primitive,
        info=final_info,
        length=step + 1,
        max_steps=env.max_steps,
        terminated=terminated,
        mean_abs_cross_velocity=float(np.mean(abs_cross_velocity)),
        mean_abs_yaw_rate=float(np.mean(abs_yaw_rate)),
        mean_action_delta=float(np.mean(action_deltas)),
        mean_action_accel=float(np.mean(action_accels)),
        mean_abs_action=float(np.mean(abs_actions)),
    )
    return result


def result_from_info(
    primitive: str,
    info: dict[str, Any],
    length: int,
    max_steps: int,
    terminated: bool,
    mean_abs_cross_velocity: float,
    mean_abs_yaw_rate: float,
    mean_action_delta: float,
    mean_action_accel: float,
    mean_abs_action: float,
) -> RolloutResult:
    x_distance = float(info.get("x_distance", 0.0))
    y_distance = float(info.get("y_distance", 0.0))
    line_distance = float(info.get("line_distance", 0.0))
    yaw_distance = float(info.get("yaw_distance", 0.0))
    cross_track_error = float(info.get("cross_track_error", 0.0))
    heading_error = float(info.get("heading_error", 0.0))
    planar_drift = float(np.hypot(x_distance, y_distance))

    if primitive in ("forward", "backward"):
        progress = line_distance
        score = (
            900.0 * progress
            - 2400.0 * abs(cross_track_error)
            - 620.0 * min(abs(heading_error), np.pi * 0.5)
            - 125.0 * mean_abs_cross_velocity
            - 80.0 * mean_abs_yaw_rate
            - 28.0 * mean_action_delta
            - 42.0 * mean_action_accel
            - 5.0 * mean_abs_action
        )
    elif primitive in ("yaw-left", "yaw-right"):
        progress = yaw_distance
        score = (
            155.0 * progress
            - 260.0 * planar_drift
            - 35.0 * abs(cross_track_error)
            - 22.0 * mean_action_delta
            - 34.0 * mean_action_accel
            - 4.0 * mean_abs_action
        )
    else:
        progress = -planar_drift - abs(yaw_distance)
        score = (
            -900.0 * planar_drift
            -120.0 * abs(yaw_distance)
            -30.0 * mean_action_delta
            -45.0 * mean_action_accel
            -10.0 * mean_abs_action
        )

    if terminated:
        remaining_frac = max(0.0, (max_steps - length) / max_steps)
        score -= 240.0 + 180.0 * remaining_frac

    return RolloutResult(
        score=float(score),
        progress=float(progress),
        x_distance=x_distance,
        y_distance=y_distance,
        cross_track_error=cross_track_error,
        heading_error=heading_error,
        yaw_distance=yaw_distance,
        planar_drift=planar_drift,
        mean_abs_cross_velocity=mean_abs_cross_velocity,
        mean_abs_yaw_rate=mean_abs_yaw_rate,
        mean_action_delta=mean_action_delta,
        mean_action_accel=mean_action_accel,
        mean_abs_action=mean_abs_action,
        length=int(length),
        terminated=bool(terminated),
    )


def evaluate_candidate(
    params: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> RolloutResult:
    episode_results = []
    env = make_env(args)
    for episode in range(args.episodes):
        episode_results.append(rollout(env, params, args, seed + episode))
    env.close()
    return average_results(episode_results)


def average_results(results: list[RolloutResult]) -> RolloutResult:
    keys = asdict(results[0]).keys()
    values: dict[str, Any] = {}
    for key in keys:
        series = [getattr(result, key) for result in results]
        if key == "terminated":
            values[key] = any(bool(value) for value in series)
        elif key == "length":
            values[key] = int(round(float(np.mean(series))))
        else:
            values[key] = float(np.mean(series))
    return RolloutResult(**values)


def run_search(args: argparse.Namespace) -> Path:
    rng = np.random.default_rng(args.seed)
    out_dir = args.out_dir or Path("runs") / (
        "gait_search_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.csv"
    best_path = out_dir / "best_params.json"

    mean = initial_mean()
    std = initial_std()
    if args.init_params is not None:
        init_primitive, init_params = load_params(args.init_params)
        if init_primitive != args.primitive and not args.init_inverse:
            raise ValueError(
                f"{args.init_params} is for primitive {init_primitive!r}, "
                f"not {args.primitive!r}."
            )
        if init_primitive != args.primitive:
            valid_inverse = {init_primitive, args.primitive} == {"forward", "backward"}
            if not valid_inverse:
                raise ValueError(
                    "--init-inverse only supports forward/backward translation gaits."
                )
            mean = inverse_translation_params(init_params)
        else:
            mean = init_params
        std = initial_std() * args.init_std_scale
    min_std = (PARAM_HIGH - PARAM_LOW) * args.min_std_frac
    elite_count = max(2, int(round(args.population * args.elite_frac)))
    best_params = mean.copy()
    best_result = evaluate_candidate(best_params, args, args.seed + 1_000_000)

    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "iteration",
                "candidate",
                "elite",
                *PARAM_NAMES,
                *asdict(best_result).keys(),
            ],
        )
        writer.writeheader()

        for iteration in range(args.iterations):
            candidates = []
            for candidate_index in range(args.population):
                if iteration == 0 and candidate_index == 0:
                    params = mean.copy()
                else:
                    params = clip_params(rng.normal(mean, std))
                result = evaluate_candidate(
                    params,
                    args,
                    args.seed + iteration * 100_000 + candidate_index * 1_000,
                )
                candidates.append((result.score, params, result))

            candidates.sort(key=lambda item: item[0], reverse=True)
            elites = candidates[:elite_count]
            elite_params = np.stack([params for _, params, _ in elites])
            elite_mean = np.mean(elite_params, axis=0)
            elite_std = np.std(elite_params, axis=0)
            mean = clip_params(args.momentum * elite_mean + (1.0 - args.momentum) * mean)
            std = np.maximum(
                args.momentum * elite_std + (1.0 - args.momentum) * std,
                min_std,
            )

            if candidates[0][0] > best_result.score:
                best_params = candidates[0][1].copy()
                best_result = candidates[0][2]
                save_params(best_path, best_params, args, best_result)

            for rank, (_, params, result) in enumerate(candidates):
                row = {
                    "iteration": iteration,
                    "candidate": rank,
                    "elite": rank < elite_count,
                }
                row.update(dict(zip(PARAM_NAMES, params, strict=True)))
                row.update(asdict(result))
                writer.writerow(row)
            f.flush()

            print(
                f"iter={iteration + 1:03d}/{args.iterations:03d} "
                f"best_score={best_result.score:.3f} "
                f"iter_best={candidates[0][2].score:.3f} "
                f"progress={best_result.progress:.4f} "
                f"cross={best_result.cross_track_error:.4f} "
                f"heading={best_result.heading_error:.3f} "
                f"len={best_result.length} "
                f"term={best_result.terminated}"
            )

    save_params(best_path, best_params, args, best_result)
    print(f"best_params={best_path}")
    return best_path


def save_params(
    path: Path,
    params: np.ndarray,
    args: argparse.Namespace,
    result: RolloutResult,
) -> None:
    payload = {
        "primitive": args.primitive,
        "param_names": PARAM_NAMES,
        "params": {name: float(value) for name, value in zip(PARAM_NAMES, params, strict=True)},
        "vector": [float(value) for value in params],
        "result": asdict(result),
        "episode_seconds": args.episode_seconds,
        "domain_randomization": args.domain_randomization,
        "domain_randomization_strength": args.domain_randomization_strength,
        "randomize_reset": args.randomize_reset,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_params(path: Path) -> tuple[str, np.ndarray]:
    payload = json.loads(path.read_text())
    if "vector" in payload:
        vector = np.array(payload["vector"], dtype=np.float64)
    else:
        vector = np.array([payload["params"][name] for name in PARAM_NAMES], dtype=np.float64)
    primitive = str(payload.get("primitive", "forward"))
    return primitive, clip_params(vector)


def view_params(params: np.ndarray, args: argparse.Namespace) -> None:
    if args.headless:
        result = evaluate_candidate(params, args, args.seed + 2_000_000)
        print(json.dumps(asdict(result), indent=2))
        return

    import mujoco.viewer

    env = make_env(args)
    forward_command, yaw_command = command_for_primitive(args.primitive)
    env.reset(
        seed=args.seed,
        options={
            "forward_command": forward_command,
            "yaw_rate_command": yaw_command,
            "randomize": args.randomize_reset,
        },
    )
    step = 0
    heading_error = 0.0
    yaw_rate = 0.0
    use_heading_correction = uses_heading_correction(args.primitive)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            start = time.perf_counter()
            action = gait_action(
                params,
                step * env.dt,
                heading_error=heading_error,
                yaw_rate=yaw_rate,
                use_heading_correction=use_heading_correction,
            )
            _, _, terminated, truncated, info = env.step(action)
            heading_error = float(info.get("heading_error", 0.0))
            yaw_rate = float(info.get("yaw_rate", 0.0))
            viewer.sync()
            step += 1
            if terminated or truncated:
                env.reset(
                    seed=args.seed + step,
                    options={
                        "forward_command": forward_command,
                        "yaw_rate_command": yaw_command,
                        "randomize": args.randomize_reset,
                    },
                )
                step = 0
                heading_error = 0.0
                yaw_rate = 0.0
            elapsed = time.perf_counter() - start
            if elapsed < env.dt:
                time.sleep(env.dt - elapsed)


def main() -> None:
    args = parse_args()
    maybe_relaunch_with_mjpython(args)

    if args.params is not None:
        primitive, params = load_params(args.params)
        args.primitive = primitive
        if args.view or args.headless:
            view_params(params, args)
            return
        result = evaluate_candidate(params, args, args.seed + 3_000_000)
        print(json.dumps(asdict(result), indent=2))
        return

    best_path = run_search(args)
    if args.view or args.headless:
        primitive, params = load_params(best_path)
        args.primitive = primitive
        view_params(params, args)


if __name__ == "__main__":
    main()
