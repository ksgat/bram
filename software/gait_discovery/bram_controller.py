from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


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

DEFAULT_FORWARD_GAIT = Path("runs/gait_search_forward_heading/best_params.json")
DEFAULT_BACKWARD_GAIT = Path("runs/gait_search_backward_rough/best_params.json")
DEFAULT_YAW_LEFT_TABLE = Path(
    "runs/policy_table_yaw_left_planar_300k_8s/yaw-left_policy_table.json"
)
DEFAULT_YAW_RIGHT_TABLE = Path(
    "runs/policy_table_yaw_right_planar_300k_8s_scaled_0p4/yaw-right_policy_table.json"
)
DEFAULT_ARC_CONTROLLER = Path(
    "runs/arc_controller_grid_right_yaw_0p40/arc_controller_20260701_155840/best_arc_controller.json"
)

HEADING_TRIM_LIMIT = 0.35


@dataclass(frozen=True)
class ControllerExport:
    forward_params: list[float]
    backward_params: list[float]
    yaw_left_table: list[list[float]]
    yaw_right_table: list[list[float]]
    arc_controller: dict[str, Any]
    dt: float
    residual_limit: float
    arc_yaw_scale: float
    base_scaling: str
    base_speed_min: float
    base_action_min: float


class BramGridController:
    def __init__(
        self,
        *,
        forward_params: np.ndarray,
        backward_params: np.ndarray,
        yaw_left_table: np.ndarray,
        yaw_right_table: np.ndarray,
        arc_controller: dict[str, Any],
        dt: float = 0.02,
        residual_limit: float = 0.80,
        arc_yaw_scale: float = 0.65,
        base_scaling: str = "gait-speed",
        base_speed_min: float = 0.35,
        base_action_min: float = 0.60,
    ) -> None:
        self.forward_params = normalize_params(forward_params)
        self.backward_params = normalize_params(backward_params)
        self.yaw_left_table = np.asarray(yaw_left_table, dtype=np.float32)
        self.yaw_right_table = np.asarray(yaw_right_table, dtype=np.float32)
        self.arc_controller = arc_controller
        self.dt = float(dt)
        self.residual_limit = float(residual_limit)
        self.arc_yaw_scale = float(arc_yaw_scale)
        self.base_scaling = str(base_scaling)
        self.base_speed_min = float(base_speed_min)
        self.base_action_min = float(base_action_min)

    @classmethod
    def from_files(
        cls,
        *,
        forward_gait: Path = DEFAULT_FORWARD_GAIT,
        backward_gait: Path = DEFAULT_BACKWARD_GAIT,
        yaw_left_table: Path = DEFAULT_YAW_LEFT_TABLE,
        yaw_right_table: Path = DEFAULT_YAW_RIGHT_TABLE,
        arc_controller: Path = DEFAULT_ARC_CONTROLLER,
        dt: float = 0.02,
        residual_limit: float = 0.80,
        arc_yaw_scale: float = 0.65,
        base_scaling: str = "gait-speed",
        base_speed_min: float = 0.35,
        base_action_min: float = 0.60,
    ) -> "BramGridController":
        return cls(
            forward_params=load_gait_params(forward_gait),
            backward_params=load_gait_params(backward_gait),
            yaw_left_table=load_action_table(yaw_left_table),
            yaw_right_table=load_action_table(yaw_right_table),
            arc_controller=load_json(arc_controller),
            dt=dt,
            residual_limit=residual_limit,
            arc_yaw_scale=arc_yaw_scale,
            base_scaling=base_scaling,
            base_speed_min=base_speed_min,
            base_action_min=base_action_min,
        )

    @classmethod
    def from_export(cls, path: Path) -> "BramGridController":
        payload = load_json(path)
        if "controller" in payload:
            payload = payload["controller"]
        return cls(
            forward_params=np.asarray(payload["forward_params"], dtype=np.float64),
            backward_params=np.asarray(payload["backward_params"], dtype=np.float64),
            yaw_left_table=np.asarray(payload["yaw_left_table"], dtype=np.float32),
            yaw_right_table=np.asarray(payload["yaw_right_table"], dtype=np.float32),
            arc_controller=payload["arc_controller"],
            dt=float(payload.get("dt", 0.02)),
            residual_limit=float(payload.get("residual_limit", 0.80)),
            arc_yaw_scale=float(payload.get("arc_yaw_scale", 0.65)),
            base_scaling=str(payload.get("base_scaling", "gait-speed")),
            base_speed_min=float(payload.get("base_speed_min", 0.35)),
            base_action_min=float(payload.get("base_action_min", 0.60)),
        )

    def export_payload(self) -> dict[str, Any]:
        return {
            "kind": "bram_grid_controller_export",
            "servo_order": ["front", "back_left", "back_right"],
            "command_range": {
                "forward": [-1.0, 1.0],
                "yaw": [-1.0, 1.0],
                "action": [-1.0, 1.0],
            },
            "controller": {
                "param_names": list(PARAM_NAMES),
                "forward_params": self.forward_params.astype(float).tolist(),
                "backward_params": self.backward_params.astype(float).tolist(),
                "yaw_left_table": self.yaw_left_table.astype(float).tolist(),
                "yaw_right_table": self.yaw_right_table.astype(float).tolist(),
                "arc_controller": self.arc_controller,
                "dt": self.dt,
                "residual_limit": self.residual_limit,
                "arc_yaw_scale": self.arc_yaw_scale,
                "base_scaling": self.base_scaling,
                "base_speed_min": self.base_speed_min,
                "base_action_min": self.base_action_min,
            },
        }

    def action(
        self,
        forward_command: float,
        yaw_command: float,
        step: int,
        *,
        heading_error: float = 0.0,
        yaw_rate: float = 0.0,
    ) -> np.ndarray:
        forward = float(np.clip(forward_command, -1.0, 1.0))
        yaw = float(np.clip(yaw_command, -1.0, 1.0))
        t = int(step) * self.dt
        base = self.base_action(
            forward,
            yaw,
            t,
            heading_error=heading_error,
            yaw_rate=yaw_rate,
        )
        gate = residual_gate(forward, yaw)
        if gate <= 1e-6:
            return self.teacher_action(
                forward,
                yaw,
                int(step),
                base,
            )

        params = arc_controller_params(
            self.arc_controller,
            forward,
            yaw,
            fallback_scale=default_arc_scale(forward, yaw),
        )
        yaw_action = self.yaw_action(yaw, int(step) + int(params["step_offset"]))
        yaw_scales = np.asarray(params["yaw_scales"], dtype=np.float32)
        base_delta = (float(params["base_scale"]) - 1.0) * base
        raw_residual = (base_delta + yaw_scales * yaw_action) / max(
            1e-6,
            self.residual_limit * gate,
        )
        residual = np.clip(raw_residual, -1.0, 1.0)
        return np.clip(base + self.residual_limit * gate * residual, -1.0, 1.0).astype(
            np.float32
        )

    def base_action(
        self,
        forward_command: float,
        yaw_command: float,
        t: float,
        *,
        heading_error: float = 0.0,
        yaw_rate: float = 0.0,
    ) -> np.ndarray:
        params = self.scaled_params(forward_command, yaw_command)
        use_heading_correction = abs(forward_command) >= 0.05 and abs(yaw_command) < 0.05
        return gait_action(
            params,
            t,
            heading_error=heading_error,
            yaw_rate=yaw_rate,
            use_heading_correction=use_heading_correction,
        )

    def scaled_params(self, forward_command: float, yaw_command: float) -> np.ndarray:
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
        return scaled.astype(np.float32)

    def yaw_action(self, yaw_command: float, step: int) -> np.ndarray:
        magnitude = abs(float(yaw_command))
        if magnitude < 1e-6:
            return np.zeros(3, dtype=np.float32)
        table = self.yaw_left_table if yaw_command > 0.0 else self.yaw_right_table
        if len(table) == 0:
            return np.zeros(3, dtype=np.float32)
        return np.clip(magnitude * table[int(step) % len(table)], -1.0, 1.0).astype(
            np.float32
        )

    def teacher_action(
        self,
        forward_command: float,
        yaw_command: float,
        step: int,
        base: np.ndarray,
    ) -> np.ndarray:
        forward_mag = abs(float(forward_command))
        yaw_mag = abs(float(yaw_command))
        if forward_mag < 0.05 and yaw_mag < 0.05:
            return np.zeros(3, dtype=np.float32)
        if yaw_mag < 0.05:
            return base.astype(np.float32)
        yaw = self.yaw_action(yaw_command, step)
        if forward_mag < 0.05:
            return yaw.astype(np.float32)
        return np.clip(base + self.arc_yaw_scale * yaw, -1.0, 1.0).astype(np.float32)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def load_gait_params(path: Path) -> np.ndarray:
    payload = load_json(path)
    if "vector" in payload:
        return normalize_params(np.asarray(payload["vector"], dtype=np.float64))
    params = payload.get("params", {})
    return normalize_params(np.asarray([float(params.get(name, 0.0)) for name in PARAM_NAMES]))


def load_action_table(path: Path) -> np.ndarray:
    payload = load_json(path)
    if "actions" not in payload:
        raise ValueError(f"{path} does not contain an actions table.")
    return np.asarray(payload["actions"], dtype=np.float32)


def normalize_params(params: np.ndarray) -> np.ndarray:
    params = np.asarray(params, dtype=np.float64)
    if params.shape[0] == len(PARAM_NAMES):
        return params
    if params.shape[0] > len(PARAM_NAMES):
        return params[: len(PARAM_NAMES)]
    padded = np.zeros(len(PARAM_NAMES), dtype=np.float64)
    padded[: params.shape[0]] = params
    return padded


def gait_action(
    params: np.ndarray,
    t: float,
    *,
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
    action = center + amplitude * np.sin(theta) + harmonic * np.sin(
        2.0 * theta + harmonic_phase
    )
    if use_heading_correction:
        trim = -params[13] * heading_error - params[14] * yaw_rate
        trim = float(np.clip(trim, -HEADING_TRIM_LIMIT, HEADING_TRIM_LIMIT))
        action = action + trim * params[15:18]
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def residual_gate(forward: float, yaw: float) -> float:
    return smoothstep(abs(float(forward)) / 0.28) * smoothstep(abs(float(yaw)) / 0.22)


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def default_arc_scale(forward: float, yaw: float) -> float:
    if forward >= 0.0 and yaw >= 0.0:
        return -0.20
    if forward >= 0.0 and yaw < 0.0:
        return -0.50
    if forward < 0.0 and yaw >= 0.0:
        return -0.40
    return -0.40


def arc_controller_params(
    arc_controller: dict[str, Any],
    forward: float,
    yaw: float,
    *,
    fallback_scale: float,
) -> dict[str, Any]:
    command_data = arc_controller.get("commands", {}).get(arc_command_name(forward, yaw))
    if command_data is None:
        return {
            "base_scale": 1.0,
            "yaw_scales": [fallback_scale, fallback_scale, fallback_scale],
            "step_offset": 0,
        }
    return select_arc_params(command_data, abs(float(forward)), abs(float(yaw)), fallback_scale)


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
    return normalize_arc_param_dict(command_data, fallback_scale)


def interpolate_arc_grid_params(
    command_data: dict[str, Any],
    grid: dict[str, Any],
    forward_mag: float,
    yaw_mag: float,
    fallback_scale: float,
) -> dict[str, Any]:
    parsed = parse_arc_grid(grid)
    if not parsed:
        return normalize_arc_param_dict(command_data, fallback_scale)
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
        return normalize_arc_param_dict(direct, fallback_scale)
    nearest_key = min(
        parsed,
        key=lambda item: (item[0] - target_forward) ** 2 + (item[1] - target_yaw) ** 2,
    )
    nearest = parsed.get(nearest_key)
    if nearest is not None:
        return normalize_arc_param_dict(nearest, fallback_scale)
    return normalize_arc_param_dict(command_data, fallback_scale)


def blend_arc_params(
    p00: dict[str, Any],
    p10: dict[str, Any],
    p01: dict[str, Any],
    p11: dict[str, Any],
    ft: float,
    yt: float,
) -> dict[str, Any]:
    weights = np.asarray(
        [
            (1.0 - ft) * (1.0 - yt),
            ft * (1.0 - yt),
            (1.0 - ft) * yt,
            ft * yt,
        ],
        dtype=np.float64,
    )
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


def normalize_arc_param_dict(params: dict[str, Any], fallback_scale: float) -> dict[str, Any]:
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
    names = {
        (1, 1): "arc_fl",
        (1, -1): "arc_fr",
        (-1, 1): "arc_bl",
        (-1, -1): "arc_br",
    }
    return names[(forward_sign, yaw_sign)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Bram deterministic grid controller.")
    parser.add_argument("--forward-gait", type=Path, default=DEFAULT_FORWARD_GAIT)
    parser.add_argument("--backward-gait", type=Path, default=DEFAULT_BACKWARD_GAIT)
    parser.add_argument("--yaw-left-table", type=Path, default=DEFAULT_YAW_LEFT_TABLE)
    parser.add_argument("--yaw-right-table", type=Path, default=DEFAULT_YAW_RIGHT_TABLE)
    parser.add_argument("--arc-controller", type=Path, default=DEFAULT_ARC_CONTROLLER)
    parser.add_argument("--from-export", type=Path, default=None)
    parser.add_argument("--export", type=Path, default=None)
    parser.add_argument("--forward", type=float, default=0.7)
    parser.add_argument("--yaw", type=float, default=0.7)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--heading-error", type=float, default=0.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--residual-limit", type=float, default=0.80)
    parser.add_argument("--arc-yaw-scale", type=float, default=0.65)
    parser.add_argument("--base-scaling", choices=("linear", "gait-speed"), default="gait-speed")
    parser.add_argument("--base-speed-min", type=float, default=0.35)
    parser.add_argument("--base-action-min", type=float, default=0.60)
    return parser.parse_args()


def make_controller(args: argparse.Namespace) -> BramGridController:
    if args.from_export is not None:
        return BramGridController.from_export(args.from_export)
    return BramGridController.from_files(
        forward_gait=args.forward_gait,
        backward_gait=args.backward_gait,
        yaw_left_table=args.yaw_left_table,
        yaw_right_table=args.yaw_right_table,
        arc_controller=args.arc_controller,
        dt=args.dt,
        residual_limit=args.residual_limit,
        arc_yaw_scale=args.arc_yaw_scale,
        base_scaling=args.base_scaling,
        base_speed_min=args.base_speed_min,
        base_action_min=args.base_action_min,
    )


def main() -> None:
    args = parse_args()
    controller = make_controller(args)
    if args.export is not None:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text(json.dumps(controller.export_payload(), indent=2) + "\n")
        print(f"exported={args.export}")

    for offset in range(max(0, args.steps)):
        step = args.start_step + offset
        action = controller.action(
            args.forward,
            args.yaw,
            step,
            heading_error=args.heading_error,
            yaw_rate=args.yaw_rate,
        )
        print(
            f"step={step:04d} "
            f"front={action[0]: .6f} "
            f"back_left={action[1]: .6f} "
            f"back_right={action[2]: .6f}"
        )


if __name__ == "__main__":
    main()
