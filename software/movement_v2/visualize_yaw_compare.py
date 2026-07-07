from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

from visualize_policy_primitives import (
    CASES,
    DEFAULT_YAW,
    PolicyHistory,
    VisualCase,
    configure_viewer,
    load_agent,
    make_env,
    maybe_relaunch_with_mjpython,
    print_summary,
    rollout_case,
)


DEFAULT_BAD_YAW = Path(
    "software/movement_v2/runs/rl_primitives/yaw_primitive_20260706_182238/policy.pt"
)
YAW_CASES = ("yaw_pos", "yaw_neg", "yaw_pos_half", "yaw_neg_half")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize old/current yaw policy vs a degraded yaw checkpoint."
    )
    parser.add_argument(
        "--case",
        choices=(*YAW_CASES, "suite"),
        default="suite",
        help="Yaw command to compare. 'suite' runs +1, -1, +0.5, -0.5.",
    )
    parser.add_argument("--old-yaw-checkpoint", type=Path, default=DEFAULT_YAW)
    parser.add_argument("--new-yaw-checkpoint", type=Path, default=DEFAULT_BAD_YAW)
    parser.add_argument("--old-label", default="old")
    parser.add_argument("--new-label", default="bad")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--frame-skip", type=int, default=25)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--repeat", action="store_true")
    parser.add_argument("--pause-between", type=float, default=0.8)
    parser.add_argument("--print-every", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    maybe_relaunch_with_mjpython(args)
    old_agent = load_agent(args.old_yaw_checkpoint, "yaw")
    new_agent = load_agent(args.new_yaw_checkpoint, "yaw")
    selected = selected_cases(args)
    policies = ((args.old_label, old_agent), (args.new_label, new_agent))
    if args.headless:
        run_headless(policies, selected, args)
    else:
        run_viewer(policies, selected, args)


def selected_cases(args: argparse.Namespace) -> list[VisualCase]:
    names = YAW_CASES if args.case == "suite" else (args.case,)
    return [CASES[name] for name in names]


def labeled_case(label: str, case: VisualCase) -> VisualCase:
    return replace(case, name=f"{label}_{case.name}")


def run_headless(policies, cases: list[VisualCase], args: argparse.Namespace) -> None:
    env = make_env(args, cases[0])
    history = PolicyHistory()
    try:
        for case_index, case in enumerate(cases):
            for policy_index, (label, agent) in enumerate(policies):
                case_with_label = labeled_case(label, case)
                stats = rollout_case(
                    env,
                    history,
                    agent,
                    case_with_label,
                    args,
                    seed=args.seed + 1000 * case_index + policy_index,
                    viewer=None,
                )
                print_summary(case_with_label, stats)
    finally:
        env.close()


def run_viewer(policies, cases: list[VisualCase], args: argparse.Namespace) -> None:
    import mujoco.viewer

    env = make_env(args, cases[0])
    history = PolicyHistory()
    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            configure_viewer(viewer)
            while viewer.is_running():
                for case_index, case in enumerate(cases):
                    for policy_index, (label, agent) in enumerate(policies):
                        if not viewer.is_running():
                            break
                        rollout_case(
                            env,
                            history,
                            agent,
                            labeled_case(label, case),
                            args,
                            seed=args.seed + 1000 * case_index + policy_index,
                            viewer=viewer,
                        )
                        if args.pause_between > 0.0:
                            time.sleep(args.pause_between)
                if not args.repeat:
                    return
    finally:
        env.close()


if __name__ == "__main__":
    main()
