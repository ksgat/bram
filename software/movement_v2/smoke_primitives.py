from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
GAIT_DISCOVERY = REPO_ROOT / "software" / "gait_discovery"
if str(GAIT_DISCOVERY) not in sys.path:
    sys.path.insert(0, str(GAIT_DISCOVERY))

from bram_controller import BramGridController  # noqa: E402
from bram_env import BramTripodEnv  # noqa: E402


DEFAULT_EXPORT = GAIT_DISCOVERY / "exports" / "bram_grid_controller_export.json"
DEFAULT_V2_EXPORT = REPO_ROOT / "software" / "movement_v2" / "exports" / "bram_v2_primitives.json"


@dataclass(frozen=True)
class SmokeResult:
    name: str
    forward_command: float
    yaw_command: float
    x_distance: float
    y_distance: float
    line_distance: float
    yaw_distance: float
    planar_drift: float
    cross_track_error: float
    max_tilt_rad: float
    min_height_m: float
    action_delta_rms: float
    length: int
    terminated: bool


ActionFn = Callable[[BramTripodEnv, int], np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Headless V2 primitive smoke checks for BRAM in MuJoCo."
    )
    parser.add_argument("--controller-export", type=Path, default=default_export())
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=25,
        help="MuJoCo sim steps per controller action. 25 matches movement_v2's 20 Hz primitive rate.",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--randomize-reset", action="store_true")
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--domain-randomization-strength", type=float, default=0.15)
    parser.add_argument(
        "--composition",
        choices=("translation-priority", "blend"),
        default="translation-priority",
        help="How mixed joystick commands are arbitrated. Smoke commands are pure primitives.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    controller = BramGridController.from_export(args.controller_export)
    results = run_suite(controller, args)
    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
        return
    print_results(results)


def default_export() -> Path:
    return DEFAULT_V2_EXPORT if DEFAULT_V2_EXPORT.exists() else DEFAULT_EXPORT


def run_suite(
    controller: BramGridController,
    args: argparse.Namespace,
) -> list[SmokeResult]:
    cases: tuple[tuple[str, float, float, ActionFn], ...] = (
        ("idle_zero", 0.0, 0.0, zero_action),
        ("random_small", 0.0, 0.0, random_action(args.seed)),
        ("forward", 1.0, 0.0, controller_action(controller, 1.0, 0.0, args.composition)),
        ("backward", -1.0, 0.0, controller_action(controller, -1.0, 0.0, args.composition)),
        ("yaw_pos", 0.0, 1.0, controller_action(controller, 0.0, 1.0, args.composition)),
        ("yaw_neg", 0.0, -1.0, controller_action(controller, 0.0, -1.0, args.composition)),
        ("yaw_pos_half", 0.0, 0.5, controller_action(controller, 0.0, 0.5, args.composition)),
        ("yaw_neg_half", 0.0, -0.5, controller_action(controller, 0.0, -0.5, args.composition)),
    )
    results = []
    for index, (name, forward, yaw, action_fn) in enumerate(cases):
        env = BramTripodEnv(
            episode_seconds=args.seconds,
            frame_skip=args.frame_skip,
            randomize_reset=args.randomize_reset,
            domain_randomization=args.domain_randomization,
            domain_randomization_strength=args.domain_randomization_strength,
            randomize_command=False,
            command_forward=forward,
            command_yaw_rate=yaw,
        )
        try:
            results.append(
                rollout(
                    env,
                    name,
                    forward,
                    yaw,
                    action_fn,
                    seed=args.seed + index * 1000,
                    randomize=args.randomize_reset,
                )
            )
        finally:
            env.close()
    return results


def controller_action(
    controller: BramGridController,
    forward: float,
    yaw: float,
    composition: str,
) -> ActionFn:
    def action(env: BramTripodEnv, step: int) -> np.ndarray:
        command_forward, command_yaw = arbitrate_command(forward, yaw, composition)
        controller_step = int(round(step * env.dt / controller.dt))
        return controller.action(
            command_forward,
            command_yaw,
            controller_step,
            heading_error=float(env.heading_error),
            yaw_rate=float(env.measured_gyro[2]),
        )

    return action


def arbitrate_command(
    forward: float,
    yaw: float,
    composition: str,
    *,
    deadband: float = 0.08,
) -> tuple[float, float]:
    forward = apply_deadband(forward, deadband)
    yaw = apply_deadband(yaw, deadband)
    if abs(forward) < 1e-6 or abs(yaw) < 1e-6:
        return forward, yaw
    if composition == "blend":
        return forward, yaw
    return forward, 0.0


def apply_deadband(value: float, deadband: float) -> float:
    value = float(np.clip(value, -1.0, 1.0))
    if abs(value) < deadband:
        return 0.0
    sign = -1.0 if value < 0.0 else 1.0
    return sign * ((abs(value) - deadband) / (1.0 - deadband))


def zero_action(env: BramTripodEnv, step: int) -> np.ndarray:
    del env, step
    return np.zeros(3, dtype=np.float32)


def random_action(seed: int) -> ActionFn:
    rng = np.random.default_rng(seed)

    def action(env: BramTripodEnv, step: int) -> np.ndarray:
        del env, step
        return rng.uniform(-0.20, 0.20, 3).astype(np.float32)

    return action


def rollout(
    env: BramTripodEnv,
    name: str,
    forward: float,
    yaw: float,
    action_fn: ActionFn,
    *,
    seed: int,
    randomize: bool,
) -> SmokeResult:
    env.reset(
        seed=seed,
        options={
            "forward_command": forward,
            "yaw_rate_command": yaw,
            "randomize": randomize,
        },
    )
    previous_action = np.zeros(env.action_space.shape[0], dtype=np.float32)
    action_delta_squares: list[float] = []
    max_tilt = 0.0
    min_height = float("inf")
    final_info: dict[str, Any] = {}
    terminated = False
    truncated = False
    length = 0
    for step in range(env.max_steps):
        action = np.clip(action_fn(env, step), -1.0, 1.0).astype(np.float32)
        delta = action - previous_action
        action_delta_squares.append(float(np.mean(np.square(delta))))
        previous_action = action
        _, _, terminated, truncated, final_info = env.step(action)
        max_tilt = max(max_tilt, float(final_info.get("level_tilt_rad", 0.0)))
        min_height = min(min_height, float(final_info.get("height", float("inf"))))
        length = step + 1
        if terminated or truncated:
            break

    x_distance = float(final_info.get("x_distance", 0.0))
    y_distance = float(final_info.get("y_distance", 0.0))
    return SmokeResult(
        name=name,
        forward_command=forward,
        yaw_command=yaw,
        x_distance=x_distance,
        y_distance=y_distance,
        line_distance=float(final_info.get("line_distance", 0.0)),
        yaw_distance=float(final_info.get("yaw_distance", 0.0)),
        planar_drift=float(np.hypot(x_distance, y_distance)),
        cross_track_error=float(final_info.get("cross_track_error", 0.0)),
        max_tilt_rad=max_tilt,
        min_height_m=min_height,
        action_delta_rms=float(np.sqrt(np.mean(action_delta_squares))),
        length=length,
        terminated=bool(terminated),
    )


def print_results(results: list[SmokeResult]) -> None:
    for result in results:
        print(
            f"{result.name:13s} "
            f"cmd=({result.forward_command:+.2f},{result.yaw_command:+.2f}) "
            f"line={result.line_distance:+.4f} "
            f"yaw={result.yaw_distance:+.4f} "
            f"xy=({result.x_distance:+.3f},{result.y_distance:+.3f}) "
            f"drift={result.planar_drift:.3f} "
            f"tilt={result.max_tilt_rad:.3f} "
            f"min_h={result.min_height_m:.3f} "
            f"dact={result.action_delta_rms:.4f} "
            f"len={result.length} "
            f"term={result.terminated}"
        )


if __name__ == "__main__":
    main()
