from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


GAIT_DISCOVERY_DIR = Path(__file__).resolve().parents[1] / "gait_discovery"
if str(GAIT_DISCOVERY_DIR) not in sys.path:
    sys.path.insert(0, str(GAIT_DISCOVERY_DIR))

from search_gait import gait_action, load_params  # noqa: E402


DEFAULT_OUT_DIR = Path("software/movement_v2/exports/gait_tables")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a gait_discovery sinusoidal primitive as a movement_v2 "
            "JSON action table."
        )
    )
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--primitive",
        choices=("auto", "yaw-left", "yaw-right", "forward", "backward", "idle"),
        default="auto",
    )
    parser.add_argument(
        "--policy-hz",
        type=float,
        default=10.0,
        help="Rate used to sample the searched gait before interpolation.",
    )
    parser.add_argument(
        "--output-hz",
        type=float,
        default=50.0,
        help="Rate written into the JSON table for existing table loaders.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=4,
        help="Number of gait cycles to export. Use an integer so table looping is smooth.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    primitive, params = load_params(args.params)
    if args.primitive != "auto":
        primitive = args.primitive
    if args.policy_hz <= 0.0 or args.output_hz <= 0.0:
        raise ValueError("policy/output Hz must be positive.")
    if args.cycles <= 0:
        raise ValueError("--cycles must be positive.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    policy_actions, policy_times, duration = sample_policy_actions(
        params,
        policy_hz=float(args.policy_hz),
        cycles=int(args.cycles),
    )
    output_actions = interpolate_actions(
        policy_actions,
        policy_times=policy_times,
        output_hz=float(args.output_hz),
        duration=duration,
    )
    payload = {
        "kind": "movement_v2_gait_params_action_table",
        "primitive": primitive,
        "source_params": str(args.params),
        "policy_hz": float(args.policy_hz),
        "control_hz": float(args.output_hz),
        "cycles": int(args.cycles),
        "duration_seconds": float(duration),
        "servo_order": ["front", "back_left", "back_right"],
        "actions": output_actions.astype(float).tolist(),
        "metadata": {
            "frequency_hz": float(params[0]),
            "rows": int(output_actions.shape[0]),
        },
    }
    out_path = args.out_dir / f"{primitive}_gait_table.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"gait_table={out_path}")
    print(
        f"rows={output_actions.shape[0]} policy_hz={args.policy_hz:g} "
        f"output_hz={args.output_hz:g} duration={duration:.3f}s"
    )


def sample_policy_actions(
    params: np.ndarray,
    *,
    policy_hz: float,
    cycles: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    frequency = max(1.0e-6, float(params[0]))
    duration = float(cycles) / frequency
    dt = 1.0 / policy_hz
    count = max(2, int(np.ceil(duration / dt)) + 1)
    times = np.arange(count, dtype=np.float64) * dt
    times = np.minimum(times, duration)
    actions = np.stack(
        [
            gait_action(
                params,
                float(t),
                use_heading_correction=False,
            )
            for t in times
        ],
        axis=0,
    ).astype(np.float32)
    return actions, times, duration


def interpolate_actions(
    actions: np.ndarray,
    *,
    policy_times: np.ndarray,
    output_hz: float,
    duration: float,
) -> np.ndarray:
    output_dt = 1.0 / output_hz
    output_count = max(1, int(round(duration / output_dt)))
    output_times = np.arange(output_count, dtype=np.float64) * output_dt
    out = np.zeros((output_count, actions.shape[1]), dtype=np.float32)
    for servo in range(actions.shape[1]):
        out[:, servo] = np.interp(
            output_times,
            policy_times,
            actions[:, servo],
            left=actions[0, servo],
            right=actions[-1, servo],
        )
    return np.clip(out, -1.0, 1.0).astype(np.float32)


if __name__ == "__main__":
    main()
