from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_FORWARD = Path(
    "software/movement_v2/runs/rl_primitives/forward_primitive_20260706_173603/policy_best.pt"
)
DEFAULT_BACKWARD = Path(
    "software/movement_v2/runs/rl_primitives/backward_primitive_20260706_173609/policy_best.pt"
)
DEFAULT_YAW = Path(
    "software/movement_v2/runs/rl_primitives/yaw_primitive_20260706_174051/policy_best.pt"
)
DEFAULT_OUTPUT = Path(
    "software/firmware/bram_esp32_controller/bram_policy_data.hpp"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export movement_v2 actor weights as an ESP32 C++ header."
    )
    parser.add_argument("--forward-checkpoint", type=Path, default=DEFAULT_FORWARD)
    parser.add_argument("--backward-checkpoint", type=Path, default=DEFAULT_BACKWARD)
    parser.add_argument("--yaw-checkpoint", type=Path, default=DEFAULT_YAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policies = {
        "Forward": load_policy(args.forward_checkpoint, "forward"),
        "Backward": load_policy(args.backward_checkpoint, "backward"),
        "Yaw": load_policy(args.yaw_checkpoint, "yaw"),
    }
    validate_common_shapes(policies)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_header(policies), encoding="utf-8")
    print(f"policy_header={args.output}")


def load_policy(path: Path, primitive: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    checkpoint_primitive = str(payload.get("args", {}).get("primitive", ""))
    if checkpoint_primitive != primitive:
        raise ValueError(
            f"{path} is primitive={checkpoint_primitive!r}, expected {primitive!r}"
        )
    state = payload["model_state_dict"]
    arrays = {
        "w0": tensor_array(state["actor.0.weight"]),
        "b0": tensor_array(state["actor.0.bias"]),
        "w1": tensor_array(state["actor.2.weight"]),
        "b1": tensor_array(state["actor.2.bias"]),
        "w2": tensor_array(state["actor.4.weight"]),
        "b2": tensor_array(state["actor.4.bias"]),
    }
    return {
        "path": str(path),
        "primitive": primitive,
        "obs_dim": int(payload["obs_dim"]),
        "action_dim": int(payload["action_dim"]),
        "hidden_dim": int(payload.get("args", {}).get("hidden_size", arrays["b0"].shape[0])),
        "eval_score": float(payload.get("eval_score", float("nan"))),
        "eval_primary_distance": float(payload.get("eval_primary_distance", float("nan"))),
        "eval_planar_drift": float(payload.get("eval_planar_drift", float("nan"))),
        "arrays": arrays,
    }


def tensor_array(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float32)


def validate_common_shapes(policies: dict[str, dict[str, Any]]) -> None:
    first = next(iter(policies.values()))
    expected = (first["obs_dim"], first["hidden_dim"], first["action_dim"])
    for name, policy in policies.items():
        shape = (policy["obs_dim"], policy["hidden_dim"], policy["action_dim"])
        if shape != expected:
            raise ValueError(f"{name} shape {shape} does not match {expected}")
        arrays = policy["arrays"]
        obs_dim, hidden_dim, action_dim = shape
        expected_shapes = {
            "w0": (hidden_dim, obs_dim),
            "b0": (hidden_dim,),
            "w1": (hidden_dim, hidden_dim),
            "b1": (hidden_dim,),
            "w2": (action_dim, hidden_dim),
            "b2": (action_dim,),
        }
        for key, expected_shape in expected_shapes.items():
            if arrays[key].shape != expected_shape:
                raise ValueError(
                    f"{name} {key} shape {arrays[key].shape} != {expected_shape}"
                )


def render_header(policies: dict[str, dict[str, Any]]) -> str:
    first = next(iter(policies.values()))
    obs_dim = first["obs_dim"]
    hidden_dim = first["hidden_dim"]
    action_dim = first["action_dim"]
    lines = [
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "namespace bram_policy_data {",
        "",
        "// Generated from movement_v2 PPO actor checkpoints.",
        "// Do not edit values by hand; rerun software/movement_v2/export_policy_header.py.",
        "",
        f"static constexpr std::size_t kObsDim = {obs_dim};",
        f"static constexpr std::size_t kHiddenDim = {hidden_dim};",
        f"static constexpr std::size_t kActionDim = {action_dim};",
        "static constexpr std::size_t kImuHistoryFrames = 4;",
        "static constexpr std::size_t kActionHistoryFrames = 6;",
        "static constexpr std::size_t kImuFrameDim = 4;",
        "static constexpr std::size_t kServoCount = 3;",
        "",
    ]
    for name, policy in policies.items():
        lines.extend(policy_metadata(name, policy))
        arrays = policy["arrays"]
        lines.extend(array_2d(f"k{name}W0", arrays["w0"]))
        lines.extend(array_1d(f"k{name}B0", arrays["b0"]))
        lines.extend(array_2d(f"k{name}W1", arrays["w1"]))
        lines.extend(array_1d(f"k{name}B1", arrays["b1"]))
        lines.extend(array_2d(f"k{name}W2", arrays["w2"]))
        lines.extend(array_1d(f"k{name}B2", arrays["b2"]))
    lines.extend(
        [
            "struct ActorWeights {",
            "  const float (*w0)[kObsDim];",
            "  const float* b0;",
            "  const float (*w1)[kHiddenDim];",
            "  const float* b1;",
            "  const float (*w2)[kHiddenDim];",
            "  const float* b2;",
            "};",
            "",
            "static constexpr ActorWeights kForwardPolicy = {",
            "  kForwardW0, kForwardB0, kForwardW1, kForwardB1, kForwardW2, kForwardB2,",
            "};",
            "static constexpr ActorWeights kBackwardPolicy = {",
            "  kBackwardW0, kBackwardB0, kBackwardW1, kBackwardB1, kBackwardW2, kBackwardB2,",
            "};",
            "static constexpr ActorWeights kYawPolicy = {",
            "  kYawW0, kYawB0, kYawW1, kYawB1, kYawW2, kYawB2,",
            "};",
            "",
            "}  // namespace bram_policy_data",
            "",
        ]
    )
    return "\n".join(lines)


def policy_metadata(name: str, policy: dict[str, Any]) -> list[str]:
    return [
        f"// {name} checkpoint: {policy['path']}",
        f"// eval_score={fmt(policy['eval_score'])} "
        f"primary_distance={fmt(policy['eval_primary_distance'])} "
        f"planar_drift={fmt(policy['eval_planar_drift'])}",
        "",
    ]


def array_1d(name: str, values: np.ndarray) -> list[str]:
    if values.shape[0] == 128:
        dim_name = "kHiddenDim"
    elif values.shape[0] == 3:
        dim_name = "kActionDim"
    else:
        dim_name = str(values.shape[0])
    rendered = ", ".join(f"{fmt(value)}f" for value in values.astype(float))
    return [f"static constexpr float {name}[{dim_name}] = {{", f"  {rendered}", "};", ""]


def array_2d(name: str, values: np.ndarray) -> list[str]:
    rows, cols = values.shape
    if cols == 35:
        col_name = "kObsDim"
    elif cols == 128:
        col_name = "kHiddenDim"
    else:
        col_name = str(cols)
    lines = [f"static constexpr float {name}[{rows}][{col_name}] = {{"]
    for row in values.astype(float):
        rendered = ", ".join(f"{fmt(value)}f" for value in row)
        lines.append(f"  {{{rendered}}},")
    lines.extend(["};", ""])
    return lines


def fmt(value: float) -> str:
    return f"{float(value):.9g}"


if __name__ == "__main__":
    main()
