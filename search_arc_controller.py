from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from bram_env import BramTripodEnv
from train_cpg_modulator import (
    DEFAULT_BACKWARD_GAIT,
    DEFAULT_FORWARD_GAIT,
    DEFAULT_YAW_LEFT_TABLE,
    DEFAULT_YAW_RIGHT_TABLE,
)
from train_residual_ppo import (
    BROAD_ARC_COMMANDS,
    EXACT_ARC_COMMANDS,
    arc_command_name,
    arc_grid_key,
    component_action,
    evaluate_scaled_controller,
    final_action,
    make_library,
    prepare_arc_controller,
    print_eval,
    reset_env,
    residual_gate,
    result_from_info,
    scaled_arc_residual,
    update_state,
)


ARC_VECTOR_LOW = np.array([0.65, -1.20, -1.20, -1.20, -80.0], dtype=np.float64)
ARC_VECTOR_HIGH = np.array([1.25, 0.65, 0.65, 0.65, 80.0], dtype=np.float64)
ARC_VECTOR_STD = np.array([0.12, 0.26, 0.26, 0.26, 28.0], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search a tiny mixed-command teacher over Bram's CPG + yaw-table controller."
    )
    parser.add_argument("--forward-gait", type=Path, default=DEFAULT_FORWARD_GAIT)
    parser.add_argument("--backward-gait", type=Path, default=DEFAULT_BACKWARD_GAIT)
    parser.add_argument("--yaw-left-table", type=Path, default=DEFAULT_YAW_LEFT_TABLE)
    parser.add_argument("--yaw-right-table", type=Path, default=DEFAULT_YAW_RIGHT_TABLE)
    parser.add_argument("--init-controller", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/arc_controller_search"))
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--population", type=int, default=28)
    parser.add_argument("--elite-frac", type=float, default=0.22)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--episode-seconds", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--residual-limit", type=float, default=0.80)
    parser.add_argument("--arc-yaw-scale", type=float, default=0.65)
    parser.add_argument("--base-scaling", choices=("linear", "gait-speed"), default="gait-speed")
    parser.add_argument("--base-speed-min", type=float, default=0.35)
    parser.add_argument("--base-action-min", type=float, default=0.60)
    parser.add_argument("--arc-scale-fl", type=float, default=-0.20)
    parser.add_argument("--arc-scale-fr", type=float, default=-0.50)
    parser.add_argument("--arc-scale-bl", type=float, default=-0.40)
    parser.add_argument("--arc-scale-br", type=float, default=-0.40)
    parser.add_argument("--cross-penalty", type=float, default=0.55)
    parser.add_argument("--cross-limit", type=float, default=0.20)
    parser.add_argument("--cross-excess-penalty", type=float, default=0.85)
    parser.add_argument(
        "--command",
        choices=(
            "all",
            "arc_fl",
            "arc_fr",
            "arc_bl",
            "arc_br",
            "broad_all",
            "broad_fl",
            "broad_fr",
            "broad_bl",
            "broad_br",
            "grid_all",
            "grid_fl",
            "grid_fr",
            "grid_bl",
            "grid_br",
        ),
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    run_dir = args.out_dir / time.strftime("arc_controller_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    best_path = run_dir / "best_arc_controller.json"

    controller_data = initial_controller_data(args)
    base_args = controller_args(args, controller_data)
    library = make_library(base_args)

    command_groups = arc_command_groups(args.command)
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "command",
                "iteration",
                "candidate",
                "objective",
                "command_distance",
                "line_distance",
                "yaw_distance",
                "cross_track_error",
                "length",
                "terminated",
                "base_scale",
                "yaw_scale_front",
                "yaw_scale_back_left",
                "yaw_scale_back_right",
                "step_offset",
            ],
        )
        writer.writeheader()

        for target_name, command_name, grid_key, rollout_items in command_groups:
            print(
                f"search_command={target_name} evals={len(rollout_items)}",
                flush=True,
            )
            best_vector = params_to_vector(
                get_controller_params(controller_data, command_name, grid_key)
            )
            best_eval = evaluate_vector(
                library,
                base_args,
                controller_data,
                command_name,
                grid_key,
                rollout_items,
                best_vector,
                args,
            )
            mean = best_vector.copy()
            std = ARC_VECTOR_STD.copy()
            for iteration in range(1, args.iterations + 1):
                vectors = sample_population(rng, mean, std, best_vector, args.population)
                evaluated = []
                for candidate_index, vector in enumerate(vectors):
                    result = evaluate_vector(
                        library,
                        base_args,
                        controller_data,
                        command_name,
                        grid_key,
                        rollout_items,
                        vector,
                        args,
                    )
                    evaluated.append((result["objective"], vector, result))
                    writer.writerow(
                        {
                            "command": target_name,
                            "iteration": iteration,
                            "candidate": candidate_index,
                            **result,
                            **vector_row(vector),
                        }
                    )
                f.flush()

                evaluated.sort(key=lambda item: item[0], reverse=True)
                if evaluated[0][0] > best_eval["objective"]:
                    best_eval = evaluated[0][2]
                    best_vector = evaluated[0][1].copy()
                    set_controller_params(
                        controller_data,
                        command_name,
                        grid_key,
                        vector_to_params(best_vector),
                    )
                    save_controller(best_path, controller_data, args)

                elite_count = max(2, int(round(args.population * args.elite_frac)))
                elites = np.stack([item[1] for item in evaluated[:elite_count]])
                mean = 0.55 * mean + 0.45 * elites.mean(axis=0)
                std = np.maximum(ARC_VECTOR_STD * 0.08, 0.55 * std + 0.45 * elites.std(axis=0))
                print(
                    "arc_iter "
                    f"cmd={target_name} iter={iteration}/{args.iterations} "
                    f"best_obj={best_eval['objective']:.4f} "
                    f"cmd_dist={best_eval['command_distance']:.4f} "
                    f"line={best_eval['line_distance']:.4f} "
                    f"yaw={best_eval['yaw_distance']:.4f} "
                    f"cross={best_eval['cross_track_error']:.3f} "
                    f"params={vector_to_params(best_vector)}",
                    flush=True,
                )

    save_controller(best_path, controller_data, args)
    base_args.arc_controller = best_path
    prepare_arc_controller(base_args)
    base_args.eval_suite = (
        "broad" if args.command.startswith("broad") or args.command.startswith("grid") else "core"
    )
    print(f"saved={best_path}", flush=True)
    print_eval(evaluate_scaled_controller(library, base_args), prefix="arc_controller_eval")


def arc_command_groups(
    command: str,
) -> tuple[tuple[str, str, str | None, tuple[tuple[str, float, float], ...]], ...]:
    if command.startswith("grid"):
        groups = []
        for forward, yaw in BROAD_ARC_COMMANDS:
            key = arc_command_name(forward, yaw)
            target_name = f"{key}_{abs(forward):.2f}_{abs(yaw):.2f}".replace(".", "p")
            groups.append(
                (
                    target_name,
                    key,
                    arc_grid_key(abs(forward), abs(yaw)),
                    ((target_name, forward, yaw),),
                )
            )
        if command == "grid_all":
            return tuple(groups)
        key = command.replace("grid_", "arc_")
        return tuple(group for group in groups if group[1] == key)

    if command.startswith("broad"):
        groups = []
        for key in ("arc_fl", "arc_fr", "arc_bl", "arc_br"):
            rollout_items = tuple(
                (
                    f"{key}_{abs(forward):.2f}_{abs(yaw):.2f}".replace(".", "p"),
                    forward,
                    yaw,
                )
                for forward, yaw in BROAD_ARC_COMMANDS
                if arc_command_name(forward, yaw) == key
            )
            groups.append((key, key, None, rollout_items))
        if command == "broad_all":
            return tuple(groups)
        key = command.replace("broad_", "arc_")
        return tuple(group for group in groups if group[1] == key)

    groups = tuple(
        (
            arc_command_name(forward, yaw),
            arc_command_name(forward, yaw),
            None,
            ((arc_command_name(forward, yaw), forward, yaw),),
        )
        for forward, yaw in EXACT_ARC_COMMANDS
    )
    if command == "all":
        return groups
    return tuple(group for group in groups if group[0] == command)


def initial_controller_data(args: argparse.Namespace) -> dict[str, Any]:
    if args.init_controller is not None:
        with args.init_controller.open("r") as f:
            return json.load(f)
    return {
        "version": 1,
        "kind": "bram_arc_controller",
        "description": "Per-quadrant mixed-command CPG carrier plus yaw-table residual controller.",
        "commands": {
            "arc_fl": default_params(args.arc_scale_fl),
            "arc_fr": default_params(args.arc_scale_fr),
            "arc_bl": default_params(args.arc_scale_bl),
            "arc_br": default_params(args.arc_scale_br),
        },
    }


def default_params(scale: float) -> dict[str, Any]:
    return {
        "base_scale": 1.0,
        "yaw_scales": [float(scale), float(scale), float(scale)],
        "step_offset": 0,
    }


def controller_args(args: argparse.Namespace, controller_data: dict[str, Any]) -> argparse.Namespace:
    namespace = argparse.Namespace(
        forward_gait=args.forward_gait,
        backward_gait=args.backward_gait,
        yaw_left_table=args.yaw_left_table,
        yaw_right_table=args.yaw_right_table,
        residual_limit=args.residual_limit,
        arc_yaw_scale=args.arc_yaw_scale,
        base_scaling=args.base_scaling,
        base_speed_min=args.base_speed_min,
        base_action_min=args.base_action_min,
        episode_seconds=args.episode_seconds,
        eval_episodes=args.episodes,
        seed=args.seed,
        arc_scale_fl=args.arc_scale_fl,
        arc_scale_fr=args.arc_scale_fr,
        arc_scale_bl=args.arc_scale_bl,
        arc_scale_br=args.arc_scale_br,
        arc_controller=None,
        arc_controller_data=controller_data,
        eval_suite="core",
    )
    return namespace


def sample_population(
    rng: np.random.Generator,
    mean: np.ndarray,
    std: np.ndarray,
    best: np.ndarray,
    population: int,
) -> list[np.ndarray]:
    vectors = [clip_vector(best), clip_vector(mean)]
    while len(vectors) < population:
        vectors.append(clip_vector(rng.normal(mean, std)))
    return vectors


def evaluate_vector(
    library: Any,
    base_args: argparse.Namespace,
    controller_data: dict[str, Any],
    command_name: str,
    grid_key: str | None,
    rollout_items: tuple[tuple[str, float, float], ...],
    vector: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    candidate_data = copy.deepcopy(controller_data)
    set_controller_params(candidate_data, command_name, grid_key, vector_to_params(vector))
    base_args.arc_controller_data = candidate_data
    results = []
    for episode in range(args.episodes):
        for item_index, (rollout_name, forward, yaw) in enumerate(rollout_items):
            results.append(
                rollout_command(
                    library,
                    base_args,
                    rollout_name,
                    forward,
                    yaw,
                    args.seed + 120_000 + 1000 * episode + item_index,
                )
            )
    command_distance = float(np.mean([result.command_distance for result in results]))
    line_distance = float(np.mean([result.line_distance for result in results]))
    yaw_distance = float(np.mean([result.yaw_distance for result in results]))
    cross_track_error = float(np.mean([abs(result.cross_track_error) for result in results]))
    length = float(np.mean([result.length for result in results]))
    terminated = any(result.terminated for result in results)
    line_part = max(0.0, line_distance)
    yaw_part = max(0.0, yaw_distance / 10.0)
    balanced_part = min(line_part, yaw_part)
    objective = (
        command_distance
        + 0.65 * balanced_part
        + 0.14 * line_part
        + 0.08 * yaw_part
        - args.cross_penalty * cross_track_error
        - args.cross_excess_penalty * max(0.0, cross_track_error - args.cross_limit)
        - (0.35 if terminated else 0.0)
    )
    return {
        "objective": float(objective),
        "command_distance": command_distance,
        "line_distance": line_distance,
        "yaw_distance": yaw_distance,
        "cross_track_error": cross_track_error,
        "length": length,
        "terminated": terminated,
    }


def get_controller_params(
    controller_data: dict[str, Any],
    command_name: str,
    grid_key: str | None,
) -> dict[str, Any]:
    command_data = controller_data["commands"][command_name]
    if grid_key is None:
        return command_data
    grid = command_data.get("grid", {})
    if isinstance(grid, dict) and grid_key in grid:
        return grid[grid_key]
    return command_data


def set_controller_params(
    controller_data: dict[str, Any],
    command_name: str,
    grid_key: str | None,
    params: dict[str, Any],
) -> None:
    if grid_key is None:
        existing = controller_data["commands"].get(command_name, {})
        if isinstance(existing, dict) and isinstance(existing.get("grid"), dict):
            params = {**params, "grid": existing["grid"]}
        controller_data["commands"][command_name] = params
        return
    command_data = controller_data["commands"].setdefault(command_name, {})
    if not isinstance(command_data, dict):
        command_data = {}
        controller_data["commands"][command_name] = command_data
    grid = command_data.setdefault("grid", {})
    grid[grid_key] = params


def rollout_command(
    library: Any,
    args: argparse.Namespace,
    command_name: str,
    forward: float,
    yaw: float,
    seed: int,
) -> Any:
    env = BramTripodEnv(
        episode_seconds=args.episode_seconds,
        randomize_reset=False,
        domain_randomization=False,
        randomize_command=False,
        command_forward=forward,
        command_yaw_rate=yaw,
    )
    obs, state = reset_env(env, seed, (forward, yaw))
    total_reward = 0.0
    final_info: dict[str, Any] = {}
    terminated = False
    truncated = False
    for step in range(env.max_steps):
        base = component_action(library, env, state, args.arc_yaw_scale)
        gate = residual_gate(forward, yaw)
        residual = scaled_arc_residual(library, forward, yaw, state.step, gate, args, base)
        action = final_action(base, residual, gate, args.residual_limit)
        obs, reward, terminated, truncated, final_info = env.step(action)
        total_reward += float(reward)
        update_state(state, final_info)
        if terminated or truncated:
            break
    env.close()
    return result_from_info(command_name, total_reward, final_info, step + 1, terminated)


def params_to_vector(params: dict[str, Any]) -> np.ndarray:
    yaw_scales = params.get("yaw_scales", params.get("yaw_scale", 0.0))
    if isinstance(yaw_scales, (int, float, np.integer, np.floating)):
        yaw_scales = [float(yaw_scales)] * 3
    return clip_vector(
        np.array(
            [
                float(params.get("base_scale", 1.0)),
                float(yaw_scales[0]),
                float(yaw_scales[1]),
                float(yaw_scales[2]),
                float(params.get("step_offset", 0)),
            ],
            dtype=np.float64,
        )
    )


def vector_to_params(vector: np.ndarray) -> dict[str, Any]:
    vector = clip_vector(vector)
    return {
        "base_scale": round(float(vector[0]), 5),
        "yaw_scales": [round(float(value), 5) for value in vector[1:4]],
        "step_offset": int(round(float(vector[4]))),
    }


def vector_row(vector: np.ndarray) -> dict[str, Any]:
    params = vector_to_params(vector)
    return {
        "base_scale": params["base_scale"],
        "yaw_scale_front": params["yaw_scales"][0],
        "yaw_scale_back_left": params["yaw_scales"][1],
        "yaw_scale_back_right": params["yaw_scales"][2],
        "step_offset": params["step_offset"],
    }


def clip_vector(vector: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(vector, dtype=np.float64), ARC_VECTOR_LOW, ARC_VECTOR_HIGH)
    return clipped


def save_controller(path: Path, controller_data: dict[str, Any], args: argparse.Namespace) -> None:
    payload = copy.deepcopy(controller_data)
    payload["source"] = {
        "script": "search_arc_controller.py",
        "seed": args.seed,
        "iterations": args.iterations,
        "population": args.population,
        "episodes": args.episodes,
        "episode_seconds": args.episode_seconds,
        "residual_limit": args.residual_limit,
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
