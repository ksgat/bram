#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


COMMANDS = ("arc_fl", "arc_fr", "arc_bl", "arc_br")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an Arduino-friendly Bram controller data header."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("software/movement_v2/exports/bram_v2_primitives.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("software/firmware/bram_esp32_controller/bram_controller_data.hpp"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text())
    controller = payload["controller"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_header(controller), encoding="utf-8")
    print(f"wrote={args.output}")


def render_header(controller: dict[str, Any]) -> str:
    lines: list[str] = [
        "#pragma once",
        "",
        "#include <cstddef>",
        "",
        "namespace bram_data {",
        "",
        "// Generated from a movement_v2 primitive bundle.",
        "// Do not edit values by hand; rerun software/movement_v2/export_primitive_bundle.py.",
        "",
        f"static constexpr float kDt = {fmt(controller['dt'])}f;",
        f"static constexpr float kResidualLimit = {fmt(controller['residual_limit'])}f;",
        f"static constexpr float kArcYawScale = {fmt(controller['arc_yaw_scale'])}f;",
        f"static constexpr float kBaseSpeedMin = {fmt(controller['base_speed_min'])}f;",
        f"static constexpr float kBaseActionMin = {fmt(controller['base_action_min'])}f;",
        "static constexpr std::size_t kParamCount = 21;",
        "static constexpr std::size_t kServoCount = 3;",
        "",
    ]
    lines.extend(array_1d("kForwardParams", controller["forward_params"]))
    lines.extend(array_1d("kBackwardParams", controller["backward_params"]))
    lines.extend(array_2d("kForwardTable", controller["forward_table"]))
    lines.extend(array_2d("kBackwardTable", controller["backward_table"]))
    lines.extend(array_2d("kYawLeftTable", controller["yaw_left_table"]))
    lines.extend(array_2d("kYawRightTable", controller["yaw_right_table"]))
    lines.extend(
        [
            "struct ArcParams {",
            "  float baseScale;",
            "  float yawScales[3];",
            "  int stepOffset;",
            "};",
            "",
            "struct ArcGridPoint {",
            "  float forward;",
            "  float yaw;",
            "  ArcParams params;",
            "};",
            "",
        ]
    )
    commands = controller["arc_controller"]["commands"]
    for command in COMMANDS:
        command_data = commands[command]
        lines.extend(arc_default(command, command_data))
        grid = command_data.get("grid", {})
        lines.extend(arc_grid(command, grid))
    lines.append("}  // namespace bram_data")
    lines.append("")
    return "\n".join(lines)


def array_1d(name: str, values: list[float]) -> list[str]:
    rendered = ", ".join(f"{fmt(value)}f" for value in values)
    return [f"static constexpr float {name}[kParamCount] = {{", f"  {rendered}", "};", ""]


def array_2d(name: str, rows: list[list[float]]) -> list[str]:
    if not rows:
        return [
            f"static constexpr std::size_t {name}Rows = 0;",
            f"static constexpr float {name}[1][kServoCount] = {{",
            "  {0.0f, 0.0f, 0.0f},",
            "};",
            "",
        ]
    lines = [
        f"static constexpr std::size_t {name}Rows = {len(rows)};",
        f"static constexpr float {name}[{name}Rows][kServoCount] = {{",
    ]
    for row in rows:
        rendered = ", ".join(f"{fmt(value)}f" for value in row)
        lines.append(f"  {{{rendered}}},")
    lines.extend(["};", ""])
    return lines


def arc_default(command: str, command_data: dict[str, Any]) -> list[str]:
    params = normalize_params(command_data)
    return [
        f"static constexpr ArcParams k{camel(command)}Default = {{",
        f"  {fmt(params['base_scale'])}f,",
        "  {"
        + ", ".join(f"{fmt(value)}f" for value in params["yaw_scales"])
        + "},",
        f"  {int(params['step_offset'])},",
        "};",
        "",
    ]


def arc_grid(command: str, grid: dict[str, Any]) -> list[str]:
    name = camel(command)
    items = []
    for key, value in sorted(grid.items(), key=lambda item: parse_grid_key(item[0])):
        forward, yaw = parse_grid_key(key)
        items.append((forward, yaw, normalize_params(value)))
    lines = [
        f"static constexpr std::size_t k{name}GridCount = {len(items)};",
        f"static constexpr ArcGridPoint k{name}Grid[k{name}GridCount] = {{",
    ]
    for forward, yaw, params in items:
        yaw_scales = ", ".join(f"{fmt(value)}f" for value in params["yaw_scales"])
        lines.append(
            "  {"
            f"{fmt(forward)}f, {fmt(yaw)}f, "
            "{"
            f"{fmt(params['base_scale'])}f, "
            "{"
            f"{yaw_scales}"
            "}, "
            f"{int(params['step_offset'])}"
            "}},"
        )
    lines.extend(["};", ""])
    return lines


def normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    yaw_scales = params.get("yaw_scales", params.get("yaw_scale", 0.0))
    if isinstance(yaw_scales, int | float):
        yaw_scales = [float(yaw_scales)] * 3
    return {
        "base_scale": float(params.get("base_scale", 1.0)),
        "yaw_scales": [float(value) for value in yaw_scales],
        "step_offset": int(round(float(params.get("step_offset", 0)))),
    }


def parse_grid_key(key: str) -> tuple[float, float]:
    forward_part, yaw_part = key.split("_", maxsplit=1)
    return (
        round(float(forward_part[1:].replace("p", ".")), 2),
        round(float(yaw_part[1:].replace("p", ".")), 2),
    )


def camel(command: str) -> str:
    return "".join(part.capitalize() for part in command.split("_"))


def fmt(value: float) -> str:
    rendered = f"{float(value):.9g}"
    if "e" not in rendered and "." not in rendered:
        rendered += ".0"
    return rendered


if __name__ == "__main__":
    main()
