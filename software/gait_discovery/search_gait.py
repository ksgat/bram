from __future__ import annotations

import argparse
import copy
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
YAW_LOW_DRIFT_FREQ_RANGE = (0.70, 1.70)
YAW_LOW_DRIFT_CENTER_RANGE = (-0.38, 0.38)
YAW_LOW_DRIFT_AMPLITUDE_RANGE = (0.12, 0.88)
YAW_LOW_DRIFT_HARMONIC_RANGE = (-0.22, 0.22)


@dataclass(frozen=True)
class RolloutResult:
    score: float
    progress: float
    x_distance: float
    y_distance: float
    cross_track_error: float
    heading_error: float
    yaw_distance: float
    target_yaw_distance: float
    yaw_distance_error: float
    planar_drift: float
    min_height: float
    mean_height_warning_deficit: float
    max_height_warning_deficit: float
    mean_height_deficit: float
    max_height_deficit: float
    mean_planar_drift: float
    max_planar_drift: float
    rms_planar_drift: float
    yaw_target_fraction: float
    yaw_gate_pass: bool
    mean_abs_planar_speed: float
    max_abs_planar_speed: float
    mean_abs_yaw_error: float
    mean_abs_roll_pitch_rate: float
    mean_support_deficit: float
    mean_contact_foot_speed: float
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
    parser.add_argument(
        "--search-space",
        choices=("default", "yaw_low_drift"),
        default="default",
        help="Narrow yaw_low_drift locks unused yaw params and searches a smoother gait space.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=50,
        help="MuJoCo sim steps per controller action. 50 matches movement_v2 yaw at 10 Hz.",
    )
    parser.add_argument(
        "--yaw-target-rate-per-command",
        type=float,
        default=0.36,
        help="Yaw scoring target in rad/s for abs(command)=1.0.",
    )
    parser.add_argument("--yaw-final-drift-limit-m", type=float, default=0.040)
    parser.add_argument("--yaw-mean-drift-limit-m", type=float, default=0.025)
    parser.add_argument("--yaw-max-drift-limit-m", type=float, default=0.040)
    parser.add_argument("--yaw-min-target-frac", type=float, default=0.65)
    parser.add_argument(
        "--episode-seconds-suite",
        type=str,
        default=None,
        help="Comma-separated horizons to average for each candidate, e.g. 4,8.",
    )
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


def initial_mean(search_space: str = "default") -> np.ndarray:
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
    if search_space == "yaw_low_drift":
        mean[0] = 1.30
        mean[1:4] = np.array([0.00, -0.06, 0.06])
        mean[4:7] = np.array([0.58, 0.62, 0.62])
        mean[7:10] = np.array([0.0, 2.0 * np.pi / 3.0, -2.0 * np.pi / 3.0])
        mean[10:13] = 0.0
        mean[13:18] = 0.0
        mean[18:21] = 0.0
    return mean


def initial_std(search_space: str = "default") -> np.ndarray:
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
    if search_space == "yaw_low_drift":
        std[0] = 0.26
        std[1:4] = 0.16
        std[4:7] = 0.18
        std[7:10] = 1.05
        std[10:13] = 0.07
        std[13:18] = 0.0
        std[18:21] = 0.85
    return std


def parameter_bounds(search_space: str = "default") -> tuple[np.ndarray, np.ndarray]:
    low = PARAM_LOW.copy()
    high = PARAM_HIGH.copy()
    if search_space == "yaw_low_drift":
        low[0], high[0] = YAW_LOW_DRIFT_FREQ_RANGE
        low[1:4] = YAW_LOW_DRIFT_CENTER_RANGE[0]
        high[1:4] = YAW_LOW_DRIFT_CENTER_RANGE[1]
        low[4:7] = YAW_LOW_DRIFT_AMPLITUDE_RANGE[0]
        high[4:7] = YAW_LOW_DRIFT_AMPLITUDE_RANGE[1]
        low[10:13] = YAW_LOW_DRIFT_HARMONIC_RANGE[0]
        high[10:13] = YAW_LOW_DRIFT_HARMONIC_RANGE[1]
        low[13:18] = 0.0
        high[13:18] = 0.0
    return low, high


def clip_params(
    params: np.ndarray,
    low: np.ndarray | None = None,
    high: np.ndarray | None = None,
) -> np.ndarray:
    params = normalize_params(params)
    low = PARAM_LOW if low is None else low
    high = PARAM_HIGH if high is None else high
    clipped = np.clip(params, low, high)
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
        frame_skip=args.frame_skip,
        episode_seconds=args.episode_seconds,
        randomize_reset=args.randomize_reset,
        domain_randomization=args.domain_randomization,
        domain_randomization_strength=args.domain_randomization_strength,
        randomize_command=False,
        command_forward=forward_command,
        command_yaw_rate=yaw_command,
    )


def episode_seconds_values(args: argparse.Namespace) -> list[float]:
    if args.episode_seconds_suite is None:
        return [float(args.episode_seconds)]
    values = [
        float(value.strip())
        for value in args.episode_seconds_suite.split(",")
        if value.strip()
    ]
    if not values:
        raise ValueError("--episode-seconds-suite did not contain any values.")
    if any(value <= 0.0 for value in values):
        raise ValueError("--episode-seconds-suite values must be positive.")
    return values


def uses_heading_correction(primitive: str) -> bool:
    return primitive in ("forward", "backward")


def yaw_quality_score(
    *,
    progress: float,
    planar_drift: float,
    mean_planar_drift: float,
    max_planar_drift: float,
    mean_abs_planar_speed: float,
    max_abs_planar_speed: float,
    mean_abs_cross_velocity: float,
    mean_abs_yaw_error: float,
    mean_abs_roll_pitch_rate: float,
    mean_support_deficit: float,
    mean_contact_foot_speed: float,
    mean_height_warning_deficit: float,
    max_height_warning_deficit: float,
    mean_height_deficit: float,
    max_height_deficit: float,
    mean_action_delta: float,
    mean_action_accel: float,
    mean_abs_action: float,
    target_progress: float,
    final_drift_limit_m: float,
    mean_drift_limit_m: float,
    max_drift_limit_m: float,
    min_target_frac: float,
    terminated: bool,
    length: int,
    max_steps: int,
) -> float:
    """Strict yaw-in-place score aligned with visual usefulness.

    The old yaw score mostly cared about final yaw and final drift. This one
    punishes wandering during the rollout, high translation speed while
    spinning, rough commands, and body thrash.
    """

    target_progress = max(1e-6, target_progress)
    useful_progress = max(0.0, progress)
    rewarded_progress = min(useful_progress, target_progress)
    target_error = abs(progress - target_progress)
    wrong_way_penalty = 520.0 * max(0.0, -progress)
    underspin_penalty = 460.0 * max(
        0.0, min_target_frac * target_progress - useful_progress
    )
    no_turn_penalty = 720.0 * max(0.0, 0.35 * target_progress - useful_progress)
    overspin_penalty = 125.0 * max(0.0, useful_progress - 1.08 * target_progress)
    final_excess = max(0.0, planar_drift - final_drift_limit_m)
    mean_excess = max(0.0, mean_planar_drift - mean_drift_limit_m)
    max_excess = max(0.0, max_planar_drift - max_drift_limit_m)
    gate_pass = (
        useful_progress >= min_target_frac * target_progress
        and planar_drift <= final_drift_limit_m
        and mean_planar_drift <= mean_drift_limit_m
        and max_planar_drift <= max_drift_limit_m
        and not terminated
    )
    gate_bonus = 240.0 + 80.0 * min(useful_progress / target_progress, 1.25) if gate_pass else 0.0
    score = (
        720.0 * rewarded_progress
        - 180.0 * target_error
        - 1600.0 * planar_drift
        - 3600.0 * mean_planar_drift
        - 7000.0 * max_planar_drift
        - 240.0 * mean_abs_planar_speed
        - 110.0 * max_abs_planar_speed
        - 75.0 * mean_abs_cross_velocity
        - 70.0 * mean_abs_yaw_error
        - 22.0 * mean_abs_roll_pitch_rate
        - 100.0 * mean_support_deficit
        - 320.0 * mean_contact_foot_speed
        - 3500.0 * mean_height_warning_deficit
        - 6000.0 * max_height_warning_deficit
        - 1400.0 * mean_height_deficit
        - 2200.0 * max_height_deficit
        - 24.0 * mean_action_delta
        - 42.0 * mean_action_accel
        - 4.0 * mean_abs_action
        - 9000.0 * final_excess
        - 11000.0 * mean_excess
        - 14000.0 * max_excess
        + gate_bonus
        - wrong_way_penalty
        - underspin_penalty
        - no_turn_penalty
        - overspin_penalty
    )
    if terminated:
        remaining_frac = max(0.0, (max_steps - length) / max_steps)
        score -= 6500.0 + 5000.0 * remaining_frac
    return float(score)


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
    planar_drifts = []
    planar_speeds = []
    abs_yaw_errors = []
    abs_roll_pitch_rates = []
    support_deficits = []
    contact_foot_speeds = []
    heights = []
    height_warning_deficits = []
    height_deficits = []
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
        planar_drifts.append(
            float(
                np.hypot(
                    float(final_info.get("x_distance", 0.0)),
                    float(final_info.get("y_distance", 0.0)),
                )
            )
        )
        planar_speeds.append(abs(float(final_info.get("planar_speed", 0.0))))
        abs_yaw_errors.append(abs(float(final_info.get("yaw_error", 0.0))))
        abs_roll_pitch_rates.append(abs(float(final_info.get("roll_pitch_rate", 0.0))))
        support_deficits.append(float(final_info.get("support_deficit", 0.0)))
        contact_foot_speeds.append(float(final_info.get("mean_contact_foot_speed", 0.0)))
        heights.append(float(final_info.get("height", 0.0)))
        height_warning_deficits.append(
            float(final_info.get("body_height_warning_deficit", 0.0))
        )
        height_deficits.append(float(final_info.get("body_height_deficit", 0.0)))
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
        elapsed_time=(step + 1) * env.dt,
        min_height=float(np.min(heights)),
        mean_height_warning_deficit=float(np.mean(height_warning_deficits)),
        max_height_warning_deficit=float(np.max(height_warning_deficits)),
        mean_height_deficit=float(np.mean(height_deficits)),
        max_height_deficit=float(np.max(height_deficits)),
        mean_planar_drift=float(np.mean(planar_drifts)),
        max_planar_drift=float(np.max(planar_drifts)),
        rms_planar_drift=float(np.sqrt(np.mean(np.square(planar_drifts)))),
        mean_abs_planar_speed=float(np.mean(planar_speeds)),
        max_abs_planar_speed=float(np.max(planar_speeds)),
        mean_abs_yaw_error=float(np.mean(abs_yaw_errors)),
        mean_abs_roll_pitch_rate=float(np.mean(abs_roll_pitch_rates)),
        mean_support_deficit=float(np.mean(support_deficits)),
        mean_contact_foot_speed=float(np.mean(contact_foot_speeds)),
        mean_abs_cross_velocity=float(np.mean(abs_cross_velocity)),
        mean_abs_yaw_rate=float(np.mean(abs_yaw_rate)),
        mean_action_delta=float(np.mean(action_deltas)),
        mean_action_accel=float(np.mean(action_accels)),
        mean_abs_action=float(np.mean(abs_actions)),
        yaw_target_rate_per_command=args.yaw_target_rate_per_command,
        final_drift_limit_m=args.yaw_final_drift_limit_m,
        mean_drift_limit_m=args.yaw_mean_drift_limit_m,
        max_drift_limit_m=args.yaw_max_drift_limit_m,
        min_target_frac=args.yaw_min_target_frac,
    )
    return result


def result_from_info(
    primitive: str,
    info: dict[str, Any],
    length: int,
    max_steps: int,
    terminated: bool,
    elapsed_time: float,
    min_height: float,
    mean_height_warning_deficit: float,
    max_height_warning_deficit: float,
    mean_height_deficit: float,
    max_height_deficit: float,
    mean_planar_drift: float,
    max_planar_drift: float,
    rms_planar_drift: float,
    mean_abs_planar_speed: float,
    max_abs_planar_speed: float,
    mean_abs_yaw_error: float,
    mean_abs_roll_pitch_rate: float,
    mean_support_deficit: float,
    mean_contact_foot_speed: float,
    mean_abs_cross_velocity: float,
    mean_abs_yaw_rate: float,
    mean_action_delta: float,
    mean_action_accel: float,
    mean_abs_action: float,
    yaw_target_rate_per_command: float,
    final_drift_limit_m: float,
    mean_drift_limit_m: float,
    max_drift_limit_m: float,
    min_target_frac: float,
) -> RolloutResult:
    x_distance = float(info.get("x_distance", 0.0))
    y_distance = float(info.get("y_distance", 0.0))
    line_distance = float(info.get("line_distance", 0.0))
    yaw_distance = float(info.get("yaw_distance", 0.0))
    target_yaw_distance = (
        abs(float(info.get("yaw_rate_command", 0.0)))
        * float(yaw_target_rate_per_command)
        * elapsed_time
    )
    yaw_distance_error = yaw_distance - target_yaw_distance
    yaw_target_fraction = yaw_distance / max(1.0e-6, target_yaw_distance)
    cross_track_error = float(info.get("cross_track_error", 0.0))
    heading_error = float(info.get("heading_error", 0.0))
    planar_drift = float(np.hypot(x_distance, y_distance))
    useful_yaw = max(0.0, yaw_distance)
    yaw_gate_pass = (
        primitive in ("yaw-left", "yaw-right")
        and useful_yaw >= min_target_frac * max(1.0e-6, target_yaw_distance)
        and planar_drift <= final_drift_limit_m
        and mean_planar_drift <= mean_drift_limit_m
        and max_planar_drift <= max_drift_limit_m
        and not terminated
    )

    if primitive in ("forward", "backward"):
        progress = line_distance
        score = (
            900.0 * progress
            - 2400.0 * abs(cross_track_error)
            - 620.0 * min(abs(heading_error), np.pi * 0.5)
            - 125.0 * mean_abs_cross_velocity
            - 80.0 * mean_abs_yaw_rate
            - 3500.0 * mean_height_warning_deficit
            - 6000.0 * max_height_warning_deficit
            - 1400.0 * mean_height_deficit
            - 2200.0 * max_height_deficit
            - 28.0 * mean_action_delta
            - 42.0 * mean_action_accel
            - 5.0 * mean_abs_action
        )
    elif primitive in ("yaw-left", "yaw-right"):
        progress = yaw_distance
        score = yaw_quality_score(
            progress=progress,
            planar_drift=planar_drift,
            mean_planar_drift=mean_planar_drift,
            max_planar_drift=max_planar_drift,
            mean_abs_planar_speed=mean_abs_planar_speed,
            max_abs_planar_speed=max_abs_planar_speed,
            mean_abs_cross_velocity=mean_abs_cross_velocity,
            mean_abs_yaw_error=mean_abs_yaw_error,
            mean_abs_roll_pitch_rate=mean_abs_roll_pitch_rate,
            mean_support_deficit=mean_support_deficit,
            mean_contact_foot_speed=mean_contact_foot_speed,
            mean_height_warning_deficit=mean_height_warning_deficit,
            max_height_warning_deficit=max_height_warning_deficit,
            mean_height_deficit=mean_height_deficit,
            max_height_deficit=max_height_deficit,
            mean_action_delta=mean_action_delta,
            mean_action_accel=mean_action_accel,
            mean_abs_action=mean_abs_action,
            target_progress=target_yaw_distance,
            final_drift_limit_m=final_drift_limit_m,
            mean_drift_limit_m=mean_drift_limit_m,
            max_drift_limit_m=max_drift_limit_m,
            min_target_frac=min_target_frac,
            terminated=terminated,
            length=length,
            max_steps=max_steps,
        )
    else:
        progress = -planar_drift - abs(yaw_distance)
        score = (
            -900.0 * planar_drift
            -120.0 * abs(yaw_distance)
            -3500.0 * mean_height_warning_deficit
            -6000.0 * max_height_warning_deficit
            -1400.0 * mean_height_deficit
            -2200.0 * max_height_deficit
            -30.0 * mean_action_delta
            -45.0 * mean_action_accel
            -10.0 * mean_abs_action
        )

    if terminated and primitive not in ("yaw-left", "yaw-right"):
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
        target_yaw_distance=target_yaw_distance,
        yaw_distance_error=yaw_distance_error,
        planar_drift=planar_drift,
        min_height=min_height,
        mean_height_warning_deficit=mean_height_warning_deficit,
        max_height_warning_deficit=max_height_warning_deficit,
        mean_height_deficit=mean_height_deficit,
        max_height_deficit=max_height_deficit,
        mean_planar_drift=mean_planar_drift,
        max_planar_drift=max_planar_drift,
        rms_planar_drift=rms_planar_drift,
        yaw_target_fraction=yaw_target_fraction,
        yaw_gate_pass=bool(yaw_gate_pass),
        mean_abs_planar_speed=mean_abs_planar_speed,
        max_abs_planar_speed=max_abs_planar_speed,
        mean_abs_yaw_error=mean_abs_yaw_error,
        mean_abs_roll_pitch_rate=mean_abs_roll_pitch_rate,
        mean_support_deficit=mean_support_deficit,
        mean_contact_foot_speed=mean_contact_foot_speed,
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
    for horizon_index, episode_seconds in enumerate(episode_seconds_values(args)):
        env_args = copy.copy(args)
        env_args.episode_seconds = episode_seconds
        env = make_env(env_args)
        for episode in range(args.episodes):
            episode_results.append(
                rollout(
                    env,
                    params,
                    env_args,
                    seed + horizon_index * 10_000 + episode,
                )
            )
        env.close()
    return average_results(episode_results)


def average_results(results: list[RolloutResult]) -> RolloutResult:
    keys = asdict(results[0]).keys()
    values: dict[str, Any] = {}
    for key in keys:
        series = [getattr(result, key) for result in results]
        if key == "terminated":
            values[key] = any(bool(value) for value in series)
        elif key == "yaw_gate_pass":
            values[key] = all(bool(value) for value in series)
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

    low, high = parameter_bounds(args.search_space)
    mean = clip_params(initial_mean(args.search_space), low, high)
    std = initial_std(args.search_space)
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
        mean = clip_params(mean, low, high)
        std = initial_std(args.search_space) * args.init_std_scale
    min_std = (high - low) * args.min_std_frac
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
                    params = clip_params(rng.normal(mean, std), low, high)
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
            mean = clip_params(
                args.momentum * elite_mean + (1.0 - args.momentum) * mean,
                low,
                high,
            )
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
                f"target_frac={best_result.yaw_target_fraction:.2f} "
                f"max_drift={best_result.max_planar_drift:.4f} "
                f"gate={int(best_result.yaw_gate_pass)} "
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
        "episode_seconds_suite": episode_seconds_values(args),
        "domain_randomization": args.domain_randomization,
        "domain_randomization_strength": args.domain_randomization_strength,
        "randomize_reset": args.randomize_reset,
        "search_space": args.search_space,
        "yaw_drift_gate": {
            "final_m": float(args.yaw_final_drift_limit_m),
            "mean_m": float(args.yaw_mean_drift_limit_m),
            "max_m": float(args.yaw_max_drift_limit_m),
            "min_target_frac": float(args.yaw_min_target_frac),
        },
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
