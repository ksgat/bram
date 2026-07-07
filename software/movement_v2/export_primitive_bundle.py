from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MOVEMENT_V2_DIR = REPO_ROOT / "software" / "movement_v2"
GAIT_DISCOVERY_DIR = REPO_ROOT / "software" / "gait_discovery"
FIRMWARE_TOOLS_DIR = REPO_ROOT / "software" / "firmware" / "tools"

for path in (GAIT_DISCOVERY_DIR, FIRMWARE_TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bram_controller import PARAM_NAMES, gait_action, load_action_table, load_gait_params  # noqa: E402
from export_bram_firmware import render_header  # noqa: E402


DEFAULT_PUSHED_RUN = GAIT_DISCOVERY_DIR / "pushed_runs" / "current_policy_20260701"
DEFAULT_FORWARD_GAIT = DEFAULT_PUSHED_RUN / "gaits" / "forward_best_params.json"
DEFAULT_BACKWARD_GAIT = DEFAULT_PUSHED_RUN / "gaits" / "backward_best_params.json"
DEFAULT_YAW_LEFT_TABLE = DEFAULT_PUSHED_RUN / "yaw_tables" / "yaw_left_policy_table.json"
DEFAULT_YAW_RIGHT_TABLE = DEFAULT_PUSHED_RUN / "yaw_tables" / "yaw_right_policy_table.json"
DEFAULT_OUT = MOVEMENT_V2_DIR / "exports" / "bram_v2_primitives.json"
DEFAULT_FIRMWARE_HEADER = (
    REPO_ROOT
    / "software"
    / "firmware"
    / "bram_esp32_controller"
    / "bram_controller_data.hpp"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export movement_v2 primitives as an ESP32-bindable controller bundle."
    )
    parser.add_argument("--forward-gait", type=Path, default=None)
    parser.add_argument("--backward-gait", type=Path, default=None)
    parser.add_argument("--forward-table", type=Path, default=None)
    parser.add_argument("--backward-table", type=Path, default=None)
    parser.add_argument("--yaw-left-table", type=Path, default=DEFAULT_YAW_LEFT_TABLE)
    parser.add_argument("--yaw-right-table", type=Path, default=DEFAULT_YAW_RIGHT_TABLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--firmware-header", type=Path, default=None)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--table-seconds", type=float, default=8.0)
    parser.add_argument(
        "--derive-translation-tables",
        action="store_true",
        help="Derive forward/back tables from legacy CPG params when RL tables are absent.",
    )
    parser.add_argument("--residual-limit", type=float, default=0.80)
    parser.add_argument("--arc-yaw-scale", type=float, default=0.0)
    parser.add_argument("--base-scaling", choices=("linear", "gait-speed"), default="gait-speed")
    parser.add_argument("--base-speed-min", type=float, default=0.35)
    parser.add_argument("--base-action-min", type=float, default=0.60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    forward_gait = args.forward_gait or DEFAULT_FORWARD_GAIT
    backward_gait = args.backward_gait or DEFAULT_BACKWARD_GAIT
    payload = build_payload(args, forward_gait, backward_gait)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"movement_v2_primitive_bundle={args.output}")
    if args.firmware_header is not None:
        args.firmware_header.parent.mkdir(parents=True, exist_ok=True)
        args.firmware_header.write_text(render_header(payload["controller"]), encoding="utf-8")
        print(f"firmware_header={args.firmware_header}")


def build_payload(
    args: argparse.Namespace,
    forward_gait: Path,
    backward_gait: Path,
) -> dict[str, Any]:
    forward_params = load_gait_params(forward_gait)
    backward_params = load_gait_params(backward_gait)
    yaw_left_table = load_action_table(args.yaw_left_table)
    yaw_right_table = load_action_table(args.yaw_right_table)
    forward_table = load_or_derive_translation_table(
        args.forward_table,
        forward_params,
        args.dt,
        args.table_seconds,
        derive=args.derive_translation_tables,
    )
    backward_table = load_or_derive_translation_table(
        args.backward_table,
        backward_params,
        args.dt,
        args.table_seconds,
        derive=args.derive_translation_tables,
    )
    controller = {
        "param_names": list(PARAM_NAMES),
        "forward_params": forward_params.astype(float).tolist(),
        "backward_params": backward_params.astype(float).tolist(),
        "forward_table": forward_table.astype(float).tolist(),
        "backward_table": backward_table.astype(float).tolist(),
        "yaw_left_table": yaw_left_table.astype(float).tolist(),
        "yaw_right_table": yaw_right_table.astype(float).tolist(),
        "arc_controller": primitive_only_arc_controller(),
        "dt": float(args.dt),
        "residual_limit": float(args.residual_limit),
        "arc_yaw_scale": float(args.arc_yaw_scale),
        "base_scaling": args.base_scaling,
        "base_speed_min": float(args.base_speed_min),
        "base_action_min": float(args.base_action_min),
    }
    return {
        "kind": "bram_movement_v2_primitive_bundle",
        "servo_order": ["front", "back_left", "back_right"],
        "command_range": {
            "forward": [-1.0, 1.0],
            "yaw": [-1.0, 1.0],
            "action": [-1.0, 1.0],
        },
        "source": {
            "forward_gait": str(forward_gait),
            "backward_gait": str(backward_gait),
            "forward_table": str(args.forward_table) if args.forward_table else None,
            "backward_table": str(args.backward_table) if args.backward_table else None,
            "yaw_left_table": str(args.yaw_left_table),
            "yaw_right_table": str(args.yaw_right_table),
        },
        "controller": controller,
    }


def primitive_only_arc_controller() -> dict[str, Any]:
    zero = {"base_scale": 1.0, "yaw_scales": [0.0, 0.0, 0.0], "step_offset": 0}
    return {
        "version": 2,
        "kind": "bram_movement_v2_primitive_only_arc_controller",
        "description": "Mixed arcs disabled; firmware V2 arbitration should select one primitive.",
        "commands": {
            "arc_fl": dict(zero),
            "arc_fr": dict(zero),
            "arc_bl": dict(zero),
            "arc_br": dict(zero),
        },
    }


def load_or_derive_translation_table(
    path: Path | None,
    params: np.ndarray,
    dt: float,
    table_seconds: float,
    *,
    derive: bool,
) -> np.ndarray:
    if path is not None:
        return load_action_table(path)
    rows = max(1, int(round(table_seconds / dt)))
    if not derive:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(
        [gait_action(params, step * dt, use_heading_correction=False) for step in range(rows)]
    ).astype(np.float32)


if __name__ == "__main__":
    main()
