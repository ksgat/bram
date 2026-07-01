from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bram_env import BramTripodEnv
from search_gait import command_for_primitive


@dataclass(frozen=True)
class CandidateResult:
    score: float
    progress: float
    yaw_distance: float
    x_distance: float
    y_distance: float
    planar_drift: float
    cross_track_error: float
    heading_error: float
    length: int
    terminated: bool
    permutation: str
    swap_rear: bool
    reverse_time: bool
    shift: int
    sign_front: int
    sign_back_left: int
    sign_back_right: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mirror a policy action table and brute-force servo sign conventions."
    )
    parser.add_argument("--source-table", type=Path, required=True)
    parser.add_argument(
        "--target-primitive",
        choices=("yaw-left", "yaw-right", "forward", "backward"),
        default="yaw-left",
    )
    parser.add_argument("--episode-seconds", type=float, default=4.0)
    parser.add_argument("--phase-shifts", type=int, default=16)
    parser.add_argument("--all-permutations", action="store_true")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_table(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if "actions" not in payload:
        raise ValueError(f"{path} does not contain an actions table.")
    return payload


def transform_actions(
    actions: np.ndarray,
    *,
    permutation: tuple[int, int, int],
    reverse_time: bool,
    shift: int,
    signs: tuple[int, int, int],
) -> np.ndarray:
    transformed = actions.copy()
    if reverse_time:
        transformed = transformed[::-1]
    transformed = transformed[:, permutation]
    transformed = np.roll(transformed, shift=shift, axis=0)
    transformed = transformed * np.asarray(signs, dtype=np.float32)
    return np.clip(transformed, -1.0, 1.0)


def make_env(primitive: str, episode_seconds: float) -> BramTripodEnv:
    forward_command, yaw_command = command_for_primitive(primitive)
    return BramTripodEnv(
        frame_skip=10,
        episode_seconds=episode_seconds,
        randomize_reset=False,
        domain_randomization=False,
        randomize_command=False,
        command_forward=forward_command,
        command_yaw_rate=yaw_command,
    )


def command_options(primitive: str) -> dict[str, float]:
    forward_command, yaw_command = command_for_primitive(primitive)
    return {"forward_command": forward_command, "yaw_rate_command": yaw_command}


def evaluate_actions(
    env: BramTripodEnv,
    actions: np.ndarray,
    args: argparse.Namespace,
    *,
    permutation_name: str,
    swap_rear: bool,
    reverse_time: bool,
    shift: int,
    signs: tuple[int, int, int],
) -> CandidateResult:
    results = []
    for episode in range(args.episodes):
        env.reset(
            seed=args.seed + episode,
            options=command_options(args.target_primitive),
        )
        final_info: dict[str, Any] = {}
        terminated = False
        truncated = False
        for step in range(env.max_steps):
            action = actions[step % len(actions)]
            _, _, terminated, truncated, final_info = env.step(action)
            if terminated or truncated:
                break
        results.append((final_info, step + 1, terminated))

    yaw_distance = float(np.mean([info.get("yaw_distance", 0.0) for info, _, _ in results]))
    x_distance = float(np.mean([info.get("x_distance", 0.0) for info, _, _ in results]))
    y_distance = float(np.mean([info.get("y_distance", 0.0) for info, _, _ in results]))
    cross_track_error = float(
        np.mean([info.get("cross_track_error", 0.0) for info, _, _ in results])
    )
    heading_error = float(np.mean([info.get("heading_error", 0.0) for info, _, _ in results]))
    length = int(round(float(np.mean([length for _, length, _ in results]))))
    terminated = any(term for _, _, term in results)
    planar_drift = float(np.hypot(x_distance, y_distance))
    progress = yaw_distance
    score = (
        165.0 * progress
        - 520.0 * planar_drift
        - 55.0 * abs(cross_track_error)
        - (240.0 if terminated else 0.0)
    )
    return CandidateResult(
        score=score,
        progress=progress,
        yaw_distance=yaw_distance,
        x_distance=x_distance,
        y_distance=y_distance,
        planar_drift=planar_drift,
        cross_track_error=cross_track_error,
        heading_error=heading_error,
        length=length,
        terminated=terminated,
        permutation=permutation_name,
        swap_rear=swap_rear,
        reverse_time=reverse_time,
        shift=shift,
        sign_front=signs[0],
        sign_back_left=signs[1],
        sign_back_right=signs[2],
    )


def candidate_rows(result: CandidateResult) -> dict[str, float | int | bool]:
    return result.__dict__.copy()


def save_table(
    path: Path,
    source: dict[str, Any],
    actions: np.ndarray,
    best: CandidateResult,
    args: argparse.Namespace,
) -> None:
    payload = {
        "primitive": args.target_primitive,
        "source_table": str(args.source_table),
        "source_primitive": source.get("primitive"),
        "dt": source.get("dt", 0.02),
        "control_hz": source.get("control_hz", 50.0),
        "episode_seconds": args.episode_seconds,
        "command": command_options(args.target_primitive),
        "mirror": candidate_rows(best),
        "actions": actions.astype(float).tolist(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source = load_table(args.source_table)
    source_actions = np.asarray(source["actions"], dtype=np.float32)
    env = make_env(args.target_primitive, args.episode_seconds)
    shifts = sorted(
        set(
            int(round(index * len(source_actions) / max(1, args.phase_shifts)))
            for index in range(max(1, args.phase_shifts))
        )
    )
    signs_options = [
        (front, back_left, back_right)
        for front in (-1, 1)
        for back_left in (-1, 1)
        for back_right in (-1, 1)
    ]
    permutations = {
        "front_left_right": (0, 1, 2),
        "front_right_left": (0, 2, 1),
    }
    if args.all_permutations:
        permutations = {
            "front_left_right": (0, 1, 2),
            "front_right_left": (0, 2, 1),
            "left_front_right": (1, 0, 2),
            "left_right_front": (1, 2, 0),
            "right_front_left": (2, 0, 1),
            "right_left_front": (2, 1, 0),
        }

    candidates: list[tuple[CandidateResult, np.ndarray]] = []
    for permutation_name, permutation in permutations.items():
        swap_rear = permutation == (0, 2, 1)
        for reverse_time in (False, True):
            for shift in shifts:
                for signs in signs_options:
                    actions = transform_actions(
                        source_actions,
                        permutation=permutation,
                        reverse_time=reverse_time,
                        shift=shift,
                        signs=signs,
                    )
                    result = evaluate_actions(
                        env,
                        actions,
                        args,
                        permutation_name=permutation_name,
                        swap_rear=swap_rear,
                        reverse_time=reverse_time,
                        shift=shift,
                        signs=signs,
                    )
                    candidates.append((result, actions))
    env.close()

    candidates.sort(key=lambda item: item[0].score, reverse=True)
    best, best_actions = candidates[0]
    metrics_path = args.out_dir / "mirror_candidates.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(candidate_rows(best).keys()))
        writer.writeheader()
        for result, _ in candidates:
            writer.writerow(candidate_rows(result))
    best_path = args.out_dir / f"{args.target_primitive}_mirrored_table.json"
    save_table(best_path, source, best_actions, best, args)
    print(f"best_table={best_path}")
    print(json.dumps(candidate_rows(best), indent=2))
    print("top5")
    for result, _ in candidates[:5]:
        print(json.dumps(candidate_rows(result), sort_keys=True))


if __name__ == "__main__":
    main()
