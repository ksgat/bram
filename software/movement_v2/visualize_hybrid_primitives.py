from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
GAIT_DISCOVERY_DIR = REPO_ROOT / "software" / "gait_discovery"
if str(GAIT_DISCOVERY_DIR) not in sys.path:
    sys.path.insert(0, str(GAIT_DISCOVERY_DIR))

from bram_controller import BramGridController  # noqa: E402
from bram_env import BramTripodEnv  # noqa: E402
from visualize_policy_primitives import (  # noqa: E402
    DEFAULT_BACKWARD,
    DEFAULT_FORWARD,
    PolicyHistory,
    RolloutStats,
    VisualCase,
    configure_viewer,
    deterministic_action,
    load_agent,
    maybe_relaunch_with_mjpython,
    print_status,
    print_summary,
)


DEFAULT_CONTROLLER_EXPORT = (
    REPO_ROOT / "software" / "movement_v2" / "exports" / "bram_v2_primitives.json"
)

CASES: dict[str, VisualCase] = {
    "idle": VisualCase("idle", "idle", 0.0, 0.0),
    "forward": VisualCase("forward", "forward", 1.0, 0.0),
    "backward": VisualCase("backward", "backward", -1.0, 0.0),
    "yaw_pos": VisualCase("yaw_pos", "yaw", 0.0, 1.0),
    "yaw_neg": VisualCase("yaw_neg", "yaw", 0.0, -1.0),
    "yaw_pos_half": VisualCase("yaw_pos_half", "yaw", 0.0, 0.5),
    "yaw_neg_half": VisualCase("yaw_neg_half", "yaw", 0.0, -0.5),
}
SUITE_CASES = (
    "idle",
    "forward",
    "backward",
    "yaw_pos",
    "yaw_neg",
    "yaw_pos_half",
    "yaw_neg_half",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize the current movement_v2 hybrid primitive stack: "
            "forward/back online actors, yaw base controller."
        )
    )
    parser.add_argument(
        "--case",
        choices=(*CASES.keys(), "suite"),
        default="suite",
    )
    parser.add_argument("--forward-checkpoint", type=Path, default=DEFAULT_FORWARD)
    parser.add_argument("--backward-checkpoint", type=Path, default=DEFAULT_BACKWARD)
    parser.add_argument("--controller-export", type=Path, default=DEFAULT_CONTROLLER_EXPORT)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=25,
        help="MuJoCo sim steps per action. 25 runs the firmware loop at 20 Hz.",
    )
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--repeat", action="store_true")
    parser.add_argument("--print-every", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    maybe_relaunch_with_mjpython(args)
    selected = selected_cases(args)
    agents = {
        "forward": load_agent(args.forward_checkpoint, "forward"),
        "backward": load_agent(args.backward_checkpoint, "backward"),
    }
    controller = BramGridController.from_export(args.controller_export)
    if args.headless:
        run_headless(agents, controller, selected, args)
    else:
        run_viewer(agents, controller, selected, args)


def selected_cases(args: argparse.Namespace) -> list[VisualCase]:
    if args.case == "suite":
        return [CASES[name] for name in SUITE_CASES]
    return [CASES[args.case]]


def make_env(args: argparse.Namespace, first_case: VisualCase) -> BramTripodEnv:
    return BramTripodEnv(
        frame_skip=args.frame_skip,
        episode_seconds=max(float(args.seconds), 0.05),
        randomize_reset=False,
        domain_randomization=False,
        randomize_command=False,
        command_forward=first_case.forward_cmd,
        command_yaw_rate=first_case.yaw_cmd,
    )


def run_headless(
    agents: dict,
    controller: BramGridController,
    cases: list[VisualCase],
    args: argparse.Namespace,
) -> None:
    env = make_env(args, cases[0])
    history = PolicyHistory()
    try:
        for case_index, case in enumerate(cases):
            stats = rollout_case(
                env,
                history,
                agents,
                controller,
                case,
                args,
                seed=args.seed + case_index,
                viewer=None,
            )
            print_summary(case, stats)
    finally:
        env.close()


def run_viewer(
    agents: dict,
    controller: BramGridController,
    cases: list[VisualCase],
    args: argparse.Namespace,
) -> None:
    import mujoco.viewer

    env = make_env(args, cases[0])
    history = PolicyHistory()
    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            configure_viewer(viewer)
            pass_index = 0
            while viewer.is_running():
                for case_index, case in enumerate(cases):
                    if not viewer.is_running():
                        break
                    stats = rollout_case(
                        env,
                        history,
                        agents,
                        controller,
                        case,
                        args,
                        seed=args.seed + pass_index * len(cases) + case_index,
                        viewer=viewer,
                    )
                    print_summary(case, stats)
                    if not args.repeat and case_index == len(cases) - 1:
                        return
                pass_index += 1
    finally:
        env.close()


def rollout_case(
    env: BramTripodEnv,
    history: PolicyHistory,
    agents: dict,
    controller: BramGridController,
    case: VisualCase,
    args: argparse.Namespace,
    *,
    seed: int,
    viewer,
) -> RolloutStats:
    print(
        f"case={case.name} primitive={case.primitive} "
        f"forward={case.forward_cmd:+.2f} yaw={case.yaw_cmd:+.2f}",
        flush=True,
    )
    env.reset(
        seed=seed,
        options={
            "randomize": False,
            "forward_command": case.forward_cmd,
            "yaw_rate_command": case.yaw_cmd,
        },
    )
    history.reset(env)
    stats = RolloutStats()
    for step in range(env.max_steps):
        if viewer is not None and not viewer.is_running():
            break
        started = time.monotonic()
        action = hybrid_action(env, history, agents, controller, case, step)
        delta = action - history.previous_action
        _, reward, terminated, truncated, info = env.step(action)
        history.append(env, action)

        stats.reward += float(reward)
        stats.length = step + 1
        stats.path_length += float(info.get("planar_speed", 0.0)) * env.dt
        stats.max_tilt_rad = max(stats.max_tilt_rad, float(info.get("level_tilt_rad", 0.0)))
        stats.min_height_m = min(stats.min_height_m, float(info.get("height", float("inf"))))
        stats.action_delta_squares.append(float(np.mean(np.square(delta))))
        stats.final_info = dict(info)
        stats.terminated = bool(terminated)

        if viewer is not None:
            viewer.sync()
        if args.print_every > 0 and step % args.print_every == 0:
            print_status(case, step, env, stats)
        if terminated or truncated:
            break
        sleep_time = (env.dt / max(args.speed, 1.0e-6)) - (time.monotonic() - started)
        if viewer is not None and sleep_time > 0.0:
            time.sleep(sleep_time)
    return stats


def hybrid_action(
    env: BramTripodEnv,
    history: PolicyHistory,
    agents: dict,
    controller: BramGridController,
    case: VisualCase,
    step: int,
) -> np.ndarray:
    if case.primitive == "idle":
        return np.zeros(env.action_space.shape[0], dtype=np.float32)
    if case.primitive in ("forward", "backward"):
        obs = history.observation(0.0)
        return deterministic_action(agents[case.primitive], obs)

    controller_step = int(np.floor((step * env.dt) / controller.dt + 1.0e-9))
    return controller.action(
        0.0,
        case.yaw_cmd,
        controller_step,
        heading_error=float(env.heading_error),
        yaw_rate=float(env.measured_gyro[2]),
    ).astype(np.float32)


if __name__ == "__main__":
    main()
