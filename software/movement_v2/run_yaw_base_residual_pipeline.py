from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MOVEMENT_V2_DIR = REPO_ROOT / "software" / "movement_v2"
GAIT_DISCOVERY_DIR = REPO_ROOT / "software" / "gait_discovery"
DEFAULT_OUTPUT_DIR = MOVEMENT_V2_DIR / "runs" / "yaw_base_residual"


@dataclass(frozen=True)
class SearchArtifact:
    direction: str
    seed: int
    params_path: Path
    score: float
    yaw_distance: float
    target_fraction: float
    final_drift: float
    mean_drift: float
    max_drift: float
    gate_pass: bool
    terminated: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search low-drift deterministic yaw bases, export movement_v2 tables, "
            "then optionally train residual yaw PPO on top."
        )
    )
    parser.add_argument(
        "--directions",
        choices=("both", "yaw-left", "yaw-right"),
        default="both",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 41, 73])
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--elite-frac", type=float, default=0.16)
    parser.add_argument("--episode-seconds-suite", default="4,8")
    parser.add_argument("--frame-skip", type=int, default=50)
    parser.add_argument("--yaw-target-rate", type=float, default=0.36)
    parser.add_argument("--final-drift-limit-m", type=float, default=0.040)
    parser.add_argument("--mean-drift-limit-m", type=float, default=0.025)
    parser.add_argument("--max-drift-limit-m", type=float, default=0.040)
    parser.add_argument("--yaw-min-target-frac", type=float, default=0.65)
    parser.add_argument("--policy-hz", type=float, default=10.0)
    parser.add_argument("--output-hz", type=float, default=40.0)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--skip-search", action="store_true")
    parser.add_argument("--left-params", type=Path, default=None)
    parser.add_argument("--right-params", type=Path, default=None)
    parser.add_argument(
        "--init-std-scale",
        type=float,
        default=0.18,
        help="Search std scale when warm-starting from --left-params/--right-params.",
    )
    parser.add_argument("--skip-residual", action="store_true")
    parser.add_argument("--residual-steps", type=int, default=50_000)
    parser.add_argument("--residual-num-envs", type=int, default=4)
    parser.add_argument("--residual-rollout-steps", type=int, default=128)
    parser.add_argument("--residual-limit", type=float, default=0.18)
    parser.add_argument("--residual-train-command", default="all")
    parser.add_argument("--residual-eval-suite", default="all")
    parser.add_argument("--no-randomize-reset", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.output_dir / time.strftime("yaw_base_residual_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars_for_json(args), indent=2) + "\n")

    selected: dict[str, Path] = {}
    if args.left_params is not None:
        selected["yaw-left"] = args.left_params
    if args.right_params is not None:
        selected["yaw-right"] = args.right_params

    if not args.skip_search:
        for direction in requested_directions(args):
            artifacts = [
                run_base_search(args, run_dir, direction, seed, selected.get(direction))
                for seed in args.seeds
            ]
            best = select_best(artifacts)
            selected[direction] = best.params_path
            write_artifact_summary(run_dir, direction, artifacts, best)
            print(
                f"selected_{direction}={best.params_path} "
                f"score={best.score:.3f} yaw={best.yaw_distance:.3f} "
                f"frac={best.target_fraction:.2f} drift={best.final_drift:.4f} "
                f"mean={best.mean_drift:.4f} max={best.max_drift:.4f} "
                f"gate={int(best.gate_pass)} term={int(best.terminated)}"
            )

    table_dir = run_dir / "tables"
    exported_tables = export_selected_tables(args, selected, table_dir)
    if args.skip_residual:
        print(f"tables_dir={table_dir}")
        return

    run_residual(args, run_dir, exported_tables)


def requested_directions(args: argparse.Namespace) -> tuple[str, ...]:
    if args.directions == "both":
        return ("yaw-left", "yaw-right")
    return (args.directions,)


def run_base_search(
    args: argparse.Namespace,
    run_dir: Path,
    direction: str,
    seed: int,
    init_params: Path | None = None,
) -> SearchArtifact:
    out_dir = run_dir / "search" / f"{direction}_seed{seed}"
    cmd = [
        sys.executable,
        str(GAIT_DISCOVERY_DIR / "search_gait.py"),
        "--primitive",
        direction,
        "--search-space",
        "yaw_low_drift",
        "--iterations",
        str(args.iterations),
        "--population",
        str(args.population),
        "--elite-frac",
        str(args.elite_frac),
        "--episode-seconds-suite",
        str(args.episode_seconds_suite),
        "--frame-skip",
        str(args.frame_skip),
        "--yaw-target-rate-per-command",
        str(args.yaw_target_rate),
        "--yaw-final-drift-limit-m",
        str(args.final_drift_limit_m),
        "--yaw-mean-drift-limit-m",
        str(args.mean_drift_limit_m),
        "--yaw-max-drift-limit-m",
        str(args.max_drift_limit_m),
        "--yaw-min-target-frac",
        str(args.yaw_min_target_frac),
        "--seed",
        str(seed),
        "--out-dir",
        str(out_dir),
    ]
    if init_params is not None:
        cmd.extend(
            [
                "--init-params",
                str(init_params),
                "--init-std-scale",
                str(args.init_std_scale),
            ]
        )
    run(cmd, cwd=REPO_ROOT)
    return load_search_artifact(direction, seed, out_dir / "best_params.json")


def load_search_artifact(direction: str, seed: int, params_path: Path) -> SearchArtifact:
    payload = json.loads(params_path.read_text())
    result = payload.get("result", {})
    return SearchArtifact(
        direction=direction,
        seed=seed,
        params_path=params_path,
        score=float(result.get("score", float("-inf"))),
        yaw_distance=float(result.get("yaw_distance", 0.0)),
        target_fraction=float(result.get("yaw_target_fraction", 0.0)),
        final_drift=float(result.get("planar_drift", 0.0)),
        mean_drift=float(result.get("mean_planar_drift", 0.0)),
        max_drift=float(result.get("max_planar_drift", 0.0)),
        gate_pass=bool(result.get("yaw_gate_pass", False)),
        terminated=bool(result.get("terminated", False)),
    )


def select_best(artifacts: list[SearchArtifact]) -> SearchArtifact:
    if not artifacts:
        raise ValueError("no search artifacts to select from")
    return max(
        artifacts,
        key=lambda item: (
            int(item.gate_pass),
            -int(item.terminated),
            item.score,
            item.target_fraction,
            -item.max_drift,
        ),
    )


def write_artifact_summary(
    run_dir: Path,
    direction: str,
    artifacts: list[SearchArtifact],
    best: SearchArtifact,
) -> None:
    rows = [
        {
            "direction": item.direction,
            "seed": item.seed,
            "params_path": str(item.params_path),
            "score": item.score,
            "yaw_distance": item.yaw_distance,
            "target_fraction": item.target_fraction,
            "final_drift": item.final_drift,
            "mean_drift": item.mean_drift,
            "max_drift": item.max_drift,
            "gate_pass": item.gate_pass,
            "terminated": item.terminated,
            "selected": item == best,
        }
        for item in artifacts
    ]
    (run_dir / f"{direction}_search_summary.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )


def export_selected_tables(
    args: argparse.Namespace,
    selected: dict[str, Path],
    table_dir: Path,
) -> dict[str, Path]:
    table_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}
    for direction, params_path in selected.items():
        cmd = [
            sys.executable,
            str(MOVEMENT_V2_DIR / "export_gait_params_table.py"),
            "--params",
            str(params_path),
            "--out-dir",
            str(table_dir),
            "--primitive",
            direction,
            "--policy-hz",
            str(args.policy_hz),
            "--output-hz",
            str(args.output_hz),
            "--cycles",
            str(args.cycles),
        ]
        run(cmd, cwd=REPO_ROOT)
        exported[direction] = table_dir / f"{direction}_gait_table.json"
    return exported


def run_residual(
    args: argparse.Namespace,
    run_dir: Path,
    tables: dict[str, Path],
) -> None:
    left_table = tables.get("yaw-left")
    right_table = tables.get("yaw-right")
    if left_table is None or right_table is None:
        raise ValueError("residual training requires both yaw-left and yaw-right tables")
    cmd = [
        sys.executable,
        str(MOVEMENT_V2_DIR / "train_yaw_residual_ppo.py"),
        "--left-table",
        str(left_table),
        "--right-table",
        str(right_table),
        "--total-steps",
        str(args.residual_steps),
        "--num-envs",
        str(args.residual_num_envs),
        "--rollout-steps",
        str(args.residual_rollout_steps),
        "--residual-limit",
        str(args.residual_limit),
        "--target-yaw-rate",
        str(args.yaw_target_rate),
        "--final-drift-limit-m",
        str(args.final_drift_limit_m),
        "--mean-drift-limit-m",
        str(args.mean_drift_limit_m),
        "--max-drift-limit-m",
        str(args.max_drift_limit_m),
        "--train-command",
        str(args.residual_train_command),
        "--eval-suite",
        str(args.residual_eval_suite),
        "--output-dir",
        str(run_dir / "residual"),
    ]
    if args.no_randomize_reset:
        cmd.append("--no-randomize-reset")
    run(cmd, cwd=REPO_ROOT)


def run(cmd: list[str], *, cwd: Path) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def vars_for_json(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


if __name__ == "__main__":
    main()
