from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from bram_controller import BramGridController
from bram_env import BramTripodEnv


DEFAULT_EXPORT = Path("exports/bram_grid_controller_export.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the current Bram controller on a figure-8 command."
    )
    parser.add_argument("--controller-export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--seconds", type=float, default=24.0)
    parser.add_argument("--forward", type=float, default=0.65)
    parser.add_argument("--yaw-amplitude", type=float, default=0.85)
    parser.add_argument(
        "--period",
        type=float,
        default=8.0,
        help="Seconds for one full left/right yaw command cycle.",
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--print-every",
        type=int,
        default=25,
        help="Print one status line every N controller steps. Set 0 to disable.",
    )
    return parser.parse_args()


def maybe_relaunch_with_mjpython(args: argparse.Namespace) -> None:
    if args.headless or platform.system() != "Darwin":
        return
    if Path(sys.executable).name == "mjpython" or os.environ.get("MJPYTHON_BIN"):
        return
    mjpython = Path(sys.executable).with_name("mjpython")
    if mjpython.exists():
        os.execv(str(mjpython), [str(mjpython), *sys.argv])


def main() -> None:
    args = parse_args()
    maybe_relaunch_with_mjpython(args)
    controller = BramGridController.from_export(args.controller_export)
    if args.headless:
        run_headless(controller, args)
    else:
        run_viewer(controller, args)


def figure8_command(args: argparse.Namespace, elapsed: float) -> tuple[float, float]:
    forward = float(np.clip(args.forward, -1.0, 1.0))
    period = max(1e-6, float(args.period))
    yaw = float(args.yaw_amplitude * np.sin(2.0 * np.pi * elapsed / period))
    return forward, float(np.clip(yaw, -1.0, 1.0))


def set_env_command(env: BramTripodEnv, forward: float, yaw: float) -> None:
    env.forward_command = float(np.clip(forward, -1.0, 1.0))
    env.yaw_rate_command = float(np.clip(yaw, -1.0, 1.0))
    env.command[:] = [env.forward_command, env.yaw_rate_command]


def reset_env(args: argparse.Namespace) -> BramTripodEnv:
    forward, yaw = figure8_command(args, 0.0)
    episode_seconds = args.seconds + 5.0 if args.seconds > 0.0 else 3600.0
    env = BramTripodEnv(
        episode_seconds=max(episode_seconds, 1.0),
        randomize_reset=False,
        domain_randomization=False,
        randomize_command=False,
        command_forward=forward,
        command_yaw_rate=yaw,
    )
    env.reset(seed=args.seed, options={"forward_command": forward, "yaw_rate_command": yaw})
    return env


def run_headless(controller: BramGridController, args: argparse.Namespace) -> None:
    env = reset_env(args)
    final_info = {}
    try:
        steps = max(1, int(args.seconds / env.dt))
        for step in range(steps):
            elapsed = step * env.dt
            final_info = step_controller(controller, env, args, step, elapsed)
            if args.print_every > 0 and step % args.print_every == 0:
                print_status(step, elapsed, env, final_info)
    finally:
        env.close()
    print("final", summarize(final_info))


def run_viewer(controller: BramGridController, args: argparse.Namespace) -> None:
    import mujoco
    import mujoco.viewer

    env = reset_env(args)
    started = time.perf_counter()
    final_info = {}
    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            configure_viewer(viewer)
            while viewer.is_running():
                loop_started = time.perf_counter()
                wall_elapsed = loop_started - started
                if args.seconds > 0.0 and wall_elapsed >= args.seconds:
                    break
                step = env.steps
                sim_elapsed = step * env.dt
                final_info = step_controller(controller, env, args, step, sim_elapsed)
                viewer.sync()
                if args.print_every > 0 and step % args.print_every == 0:
                    print_status(step, sim_elapsed, env, final_info)
                sleep_time = (env.dt / max(args.speed, 1e-6)) - (
                    time.perf_counter() - loop_started
                )
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
    finally:
        env.close()
    print("final", summarize(final_info))


def step_controller(
    controller: BramGridController,
    env: BramTripodEnv,
    args: argparse.Namespace,
    step: int,
    elapsed: float,
) -> dict:
    forward, yaw = figure8_command(args, elapsed)
    set_env_command(env, forward, yaw)
    action = controller.action(
        forward,
        yaw,
        step,
        heading_error=float(getattr(env, "heading_error", 0.0)),
        yaw_rate=0.0,
    )
    _, _, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"episode ended at step={step} elapsed={elapsed:.2f}s; resetting")
        env.reset(
            seed=args.seed + step + 1,
            options={"forward_command": forward, "yaw_rate_command": yaw},
        )
    return info


def configure_viewer(viewer) -> None:
    import mujoco

    for flag in (
        mujoco.mjtVisFlag.mjVIS_INERTIA,
        mujoco.mjtVisFlag.mjVIS_SCLINERTIA,
        mujoco.mjtVisFlag.mjVIS_CONTACTPOINT,
        mujoco.mjtVisFlag.mjVIS_CONTACTFORCE,
    ):
        viewer.opt.flags[int(flag)] = 0


def print_status(step: int, elapsed: float, env: BramTripodEnv, info: dict) -> None:
    print(
        f"step={step:04d} t={elapsed:5.2f} "
        f"cmd=({env.forward_command:+.2f},{env.yaw_rate_command:+.2f}) "
        f"xy=({float(info.get('x_distance', 0.0)):+.3f},"
        f"{float(info.get('y_distance', 0.0)):+.3f}) "
        f"yaw_dist={float(info.get('yaw_distance', 0.0)):+.3f}",
        flush=True,
    )


def summarize(info: dict) -> str:
    return (
        f"cmd={float(info.get('command_distance', 0.0)):.4f} "
        f"line={float(info.get('line_distance', 0.0)):.4f} "
        f"yaw={float(info.get('yaw_distance', 0.0)):.4f} "
        f"xy=({float(info.get('x_distance', 0.0)):.4f},"
        f"{float(info.get('y_distance', 0.0)):.4f})"
    )


if __name__ == "__main__":
    main()
