from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
GAIT_DISCOVERY_DIR = REPO_ROOT / "software" / "gait_discovery"
if str(GAIT_DISCOVERY_DIR) not in sys.path:
    sys.path.insert(0, str(GAIT_DISCOVERY_DIR))

from bram_env import BramTripodEnv  # noqa: E402


DEFAULT_LEFT_TABLE = (
    GAIT_DISCOVERY_DIR
    / "runs"
    / "policy_table_yaw_left_planar_300k_8s"
    / "yaw-left_policy_table.json"
)
DEFAULT_RIGHT_TABLE = (
    GAIT_DISCOVERY_DIR
    / "runs"
    / "policy_table_yaw_right_planar_300k_8s_scaled_0p4"
    / "yaw-right_policy_table.json"
)


@dataclass(frozen=True)
class RateResult:
    direction: str
    yaw_command: float
    action_scale: float
    generator_hz: float
    output_hz: float
    mode: str
    yaw_distance: float
    raw_heading_delta: float
    planar_drift: float
    mean_planar_drift: float
    max_planar_drift: float
    x_distance: float
    y_distance: float
    max_tilt_rad: float
    min_height_m: float
    action_delta_rms: float
    action_accel_rms: float
    mean_abs_action: float
    max_abs_action_delta: float
    length: int
    terminated: bool
    pass_gate: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay strong gait_discovery yaw tables on the current BRAM model "
            "while sweeping yaw generator rate. Servo output stays at 50 Hz by default."
        )
    )
    parser.add_argument("--left-table", type=Path, default=DEFAULT_LEFT_TABLE)
    parser.add_argument("--right-table", type=Path, default=DEFAULT_RIGHT_TABLE)
    parser.add_argument("--rates", type=float, nargs="+", default=[20.0, 30.0, 40.0, 50.0])
    parser.add_argument("--left-scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--right-scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--output-hz", type=float, default=50.0)
    parser.add_argument(
        "--mode",
        choices=("hold", "linear"),
        default="linear",
        help="How lower-rate generator samples are converted to the output tick.",
    )
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--final-drift-limit-m", type=float, default=0.040)
    parser.add_argument("--mean-drift-limit-m", type=float, default=0.025)
    parser.add_argument("--max-drift-limit-m", type=float, default=0.040)
    parser.add_argument("--target-yaw-rate", type=float, default=0.36)
    parser.add_argument("--min-target-frac", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--randomize-reset", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = {
        "left": load_table(args.left_table),
        "right": load_table(args.right_table),
    }
    results: list[RateResult] = []
    for direction, table in tables.items():
        yaw_command = 1.0 if direction == "left" else -1.0
        scales = args.left_scales if direction == "left" else args.right_scales
        for scale in scales:
            for rate in args.rates:
                results.append(
                    rollout_table(
                        table=table,
                        direction=direction,
                        yaw_command=yaw_command,
                        action_scale=float(scale),
                        generator_hz=float(rate),
                        output_hz=float(args.output_hz),
                        mode=args.mode,
                        seconds=float(args.seconds),
                        final_drift_limit_m=float(args.final_drift_limit_m),
                        mean_drift_limit_m=float(args.mean_drift_limit_m),
                        max_drift_limit_m=float(args.max_drift_limit_m),
                        target_yaw_rate=float(args.target_yaw_rate),
                        min_target_frac=float(args.min_target_frac),
                        seed=int(args.seed + len(results) * 1000),
                        randomize_reset=bool(args.randomize_reset),
                    )
                )
    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
    else:
        print_results(results)


def load_table(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    actions = np.asarray(payload.get("actions", []), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 3 or actions.shape[0] == 0:
        raise ValueError(f"{path} does not contain a non-empty Nx3 actions table.")
    control_hz = float(payload.get("control_hz", 0.0))
    if not control_hz > 0.0:
        dt = float(payload.get("dt", 0.0))
        if not dt > 0.0:
            raise ValueError(f"{path} does not define control_hz or dt.")
        control_hz = 1.0 / dt
    return {
        "path": str(path),
        "actions": np.clip(actions, -1.0, 1.0),
        "source_hz": control_hz,
    }


def rollout_table(
    *,
    table: dict[str, Any],
    direction: str,
    yaw_command: float,
    action_scale: float,
    generator_hz: float,
    output_hz: float,
    mode: str,
    seconds: float,
    final_drift_limit_m: float,
    mean_drift_limit_m: float,
    max_drift_limit_m: float,
    target_yaw_rate: float,
    min_target_frac: float,
    seed: int,
    randomize_reset: bool,
) -> RateResult:
    frame_skip = frame_skip_for_output_hz(output_hz)
    env = BramTripodEnv(
        frame_skip=frame_skip,
        episode_seconds=seconds,
        randomize_reset=randomize_reset,
        domain_randomization=False,
        randomize_command=False,
        command_forward=0.0,
        command_yaw_rate=yaw_command,
    )
    previous_action = np.zeros(env.action_space.shape[0], dtype=np.float32)
    previous_delta = np.zeros(env.action_space.shape[0], dtype=np.float32)
    action_delta_squares: list[float] = []
    action_accel_squares: list[float] = []
    abs_actions: list[float] = []
    max_abs_action_delta = 0.0
    max_tilt = 0.0
    min_height = float("inf")
    drift_values: list[float] = []
    final_info: dict[str, Any] = {}
    terminated = False
    truncated = False
    length = 0
    try:
        env.reset(
            seed=seed,
            options={
                "randomize": randomize_reset,
                "forward_command": 0.0,
                "yaw_rate_command": yaw_command,
            },
        )
        for step in range(env.max_steps):
            t = step * env.dt
            action = generator_action(
                table,
                t,
                generator_hz=generator_hz,
                action_scale=action_scale,
                mode=mode,
            )
            delta = action - previous_action
            accel = delta - previous_delta
            action_delta_squares.append(float(np.mean(np.square(delta))))
            action_accel_squares.append(float(np.mean(np.square(accel))))
            abs_actions.append(float(np.mean(np.abs(action))))
            max_abs_action_delta = max(max_abs_action_delta, float(np.max(np.abs(delta))))
            previous_action = action
            previous_delta = delta
            _, _, terminated, truncated, final_info = env.step(action)
            drift_values.append(
                float(
                    np.hypot(
                        float(final_info.get("x_distance", 0.0)),
                        float(final_info.get("y_distance", 0.0)),
                    )
                )
            )
            max_tilt = max(max_tilt, float(final_info.get("level_tilt_rad", 0.0)))
            min_height = min(min_height, float(final_info.get("height", float("inf"))))
            length = step + 1
            if terminated or truncated:
                break
    finally:
        env.close()

    x_distance = float(final_info.get("x_distance", 0.0))
    y_distance = float(final_info.get("y_distance", 0.0))
    yaw_distance = float(final_info.get("yaw_distance", 0.0))
    planar_drift = float(np.hypot(x_distance, y_distance))
    mean_drift = float(np.mean(drift_values)) if drift_values else 0.0
    max_drift = float(np.max(drift_values)) if drift_values else 0.0
    target_yaw_distance = abs(float(yaw_command)) * target_yaw_rate * length * env.dt
    pass_gate = (
        planar_drift <= final_drift_limit_m
        and mean_drift <= mean_drift_limit_m
        and max_drift <= max_drift_limit_m
        and yaw_distance >= min_target_frac * target_yaw_distance
        and not terminated
    )
    return RateResult(
        direction=direction,
        yaw_command=float(yaw_command),
        action_scale=float(action_scale),
        generator_hz=float(generator_hz),
        output_hz=float(1.0 / env.dt),
        mode=mode,
        yaw_distance=yaw_distance,
        raw_heading_delta=float(np.sign(yaw_command) * yaw_distance),
        planar_drift=planar_drift,
        mean_planar_drift=mean_drift,
        max_planar_drift=max_drift,
        x_distance=x_distance,
        y_distance=y_distance,
        max_tilt_rad=max_tilt,
        min_height_m=min_height,
        action_delta_rms=float(np.sqrt(np.mean(action_delta_squares))),
        action_accel_rms=float(np.sqrt(np.mean(action_accel_squares))),
        mean_abs_action=float(np.mean(abs_actions)),
        max_abs_action_delta=max_abs_action_delta,
        length=length,
        terminated=bool(terminated),
        pass_gate=pass_gate,
    )


def frame_skip_for_output_hz(output_hz: float) -> int:
    if output_hz <= 0.0:
        raise ValueError("--output-hz must be positive.")
    # bram.xml currently uses a 2 ms MuJoCo timestep.
    return max(1, int(round(1.0 / (0.002 * output_hz))))


def generator_action(
    table: dict[str, Any],
    t: float,
    *,
    generator_hz: float,
    action_scale: float,
    mode: str,
) -> np.ndarray:
    if generator_hz <= 0.0:
        raise ValueError("generator_hz must be positive.")
    low_t = np.floor(t * generator_hz + 1.0e-9) / generator_hz
    low_action = source_action(table, low_t)
    if mode == "hold":
        return np.clip(action_scale * low_action, -1.0, 1.0).astype(np.float32)
    high_action = source_action(table, low_t + 1.0 / generator_hz)
    alpha = float(np.clip((t - low_t) * generator_hz, 0.0, 1.0))
    action = (1.0 - alpha) * low_action + alpha * high_action
    return np.clip(action_scale * action, -1.0, 1.0).astype(np.float32)


def source_action(table: dict[str, Any], t: float) -> np.ndarray:
    actions = table["actions"]
    source_hz = float(table["source_hz"])
    duration = actions.shape[0] / source_hz
    phase = float(t) % duration
    index = phase * source_hz
    low_index = int(np.floor(index + 1.0e-9))
    low = low_index % actions.shape[0]
    high = (low + 1) % actions.shape[0]
    alpha = float(np.clip(index - low_index, 0.0, 1.0))
    return np.clip((1.0 - alpha) * actions[low] + alpha * actions[high], -1.0, 1.0).astype(
        np.float32
    )


def print_results(results: list[RateResult]) -> None:
    for result in results:
        print(
            f"{result.direction:5s} "
            f"scale={result.action_scale:.2f} "
            f"gen={result.generator_hz:>4.0f}Hz out={result.output_hz:>4.0f}Hz "
            f"{result.mode:6s} yaw={result.yaw_distance:+.3f} "
            f"raw={result.raw_heading_delta:+.3f} "
            f"drift={result.planar_drift:.3f} "
            f"mean={result.mean_planar_drift:.3f} "
            f"max={result.max_planar_drift:.3f} "
            f"tilt={result.max_tilt_rad:.3f} min_h={result.min_height_m:.3f} "
            f"dact={result.action_delta_rms:.4f} "
            f"ddact={result.action_accel_rms:.4f} "
            f"maxjump={result.max_abs_action_delta:.3f} "
            f"len={result.length} pass={int(result.pass_gate)} term={result.terminated}"
        )


if __name__ == "__main__":
    main()
