from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resample and scale a cyclic normalized servo action table."
    )
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-hz", type=float, required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--duration-seconds", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.table.read_text(encoding="utf-8"))
    actions = np.asarray(payload.get("actions", []), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 3 or actions.shape[0] == 0:
        raise ValueError(f"{args.table} does not contain a non-empty Nx3 actions table.")
    source_hz = table_hz(payload)
    output_hz = float(args.output_hz)
    if output_hz <= 0.0:
        raise ValueError("--output-hz must be positive.")

    duration = (
        float(args.duration_seconds)
        if args.duration_seconds is not None
        else actions.shape[0] / source_hz
    )
    rows = max(1, int(round(duration * output_hz)))
    output_times = np.arange(rows, dtype=np.float64) / output_hz
    output_actions = np.stack(
        [
            np.clip(float(args.scale) * source_action(actions, source_hz, t), -1.0, 1.0)
            for t in output_times
        ],
        axis=0,
    ).astype(np.float32)

    out_payload: dict[str, Any] = {
        "kind": "movement_v2_resampled_action_table",
        "primitive": payload.get("primitive", "unknown"),
        "source_table": str(args.table),
        "source_hz": float(source_hz),
        "control_hz": output_hz,
        "dt": 1.0 / output_hz,
        "duration_seconds": float(rows / output_hz),
        "action_scale": float(args.scale),
        "servo_order": payload.get("servo_order", ["front", "back_left", "back_right"]),
        "actions": output_actions.astype(float).tolist(),
        "source_metadata": {
            "checkpoint": payload.get("checkpoint"),
            "source_kind": payload.get("kind"),
            "source_dt": payload.get("dt"),
            "source_control_hz": payload.get("control_hz"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out_payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote={args.output} rows={rows} source_hz={source_hz:g} "
        f"output_hz={output_hz:g} scale={args.scale:g}"
    )


def table_hz(payload: dict[str, Any]) -> float:
    control_hz = float(payload.get("control_hz", 0.0))
    if control_hz > 0.0:
        return control_hz
    dt = float(payload.get("dt", 0.0))
    if dt > 0.0:
        return 1.0 / dt
    raise ValueError("table must define control_hz or dt.")


def source_action(actions: np.ndarray, source_hz: float, t: float) -> np.ndarray:
    duration = actions.shape[0] / source_hz
    phase = float(t) % duration
    index = phase * source_hz
    low_index = int(np.floor(index + 1.0e-9))
    low = low_index % actions.shape[0]
    high = (low + 1) % actions.shape[0]
    alpha = float(np.clip(index - low_index, 0.0, 1.0))
    return ((1.0 - alpha) * actions[low] + alpha * actions[high]).astype(np.float32)


if __name__ == "__main__":
    main()
