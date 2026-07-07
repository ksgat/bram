from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

GAIT_DISCOVERY_DIR = Path(__file__).resolve().parents[1] / "gait_discovery"
if str(GAIT_DISCOVERY_DIR) not in sys.path:
    sys.path.insert(0, str(GAIT_DISCOVERY_DIR))

from bram_env import BramTripodEnv  # noqa: E402
from train_ppo import ActorCritic  # noqa: E402
from yaw_env import (  # noqa: E402
    ACTION_HISTORY_FRAMES,
    IMU_FRAME_DIM,
    IMU_HISTORY_FRAMES,
    OBS_DIM,
    SERVO_COMMAND_DIM,
    YAW_ENV_COMMAND_MODE,
    translation_cone_gate,
    translation_displacement_error,
)


DEFAULT_FORWARD = Path(
    "software/movement_v2/runs/rl_primitives/forward_primitive_20260706_173603/policy_best.pt"
)
DEFAULT_BACKWARD = Path(
    "software/movement_v2/runs/rl_primitives/backward_primitive_20260706_173609/policy_best.pt"
)
DEFAULT_YAW = Path(
    "software/movement_v2/runs/rl_primitives/yaw_primitive_20260706_174051/policy_best.pt"
)


@dataclass(frozen=True)
class VisualCase:
    name: str
    primitive: str
    forward_cmd: float
    yaw_cmd: float


@dataclass
class RolloutStats:
    reward: float = 0.0
    length: int = 0
    path_length: float = 0.0
    max_tilt_rad: float = 0.0
    min_height_m: float = float("inf")
    action_delta_squares: list[float] | None = None
    final_info: dict | None = None
    terminated: bool = False

    def __post_init__(self) -> None:
        if self.action_delta_squares is None:
            self.action_delta_squares = []
        if self.final_info is None:
            self.final_info = {}


CASES: dict[str, VisualCase] = {
    "forward": VisualCase("forward", "forward", 1.0, 0.0),
    "backward": VisualCase("backward", "backward", -1.0, 0.0),
    "yaw_pos": VisualCase("yaw_pos", "yaw", 0.0, 1.0),
    "yaw_neg": VisualCase("yaw_neg", "yaw", 0.0, -1.0),
    "yaw_pos_half": VisualCase("yaw_pos_half", "yaw", 0.0, 0.5),
    "yaw_neg_half": VisualCase("yaw_neg_half", "yaw", 0.0, -0.5),
}
SUITE_CASES = (
    "forward",
    "backward",
    "yaw_pos",
    "yaw_neg",
    "yaw_pos_half",
    "yaw_neg_half",
)


class PolicyHistory:
    def __init__(self) -> None:
        self.imu_history: deque[np.ndarray] = deque(maxlen=IMU_HISTORY_FRAMES)
        self.action_history: deque[np.ndarray] = deque(maxlen=ACTION_HISTORY_FRAMES)
        self.previous_action = np.zeros(SERVO_COMMAND_DIM, dtype=np.float32)

    def reset(self, env: BramTripodEnv) -> None:
        self.imu_history.clear()
        self.action_history.clear()
        self.previous_action[:] = 0.0
        quat = imu_quat(env)
        zero_action = np.zeros(SERVO_COMMAND_DIM, dtype=np.float32)
        for _ in range(IMU_HISTORY_FRAMES):
            self.imu_history.append(quat.copy())
        for _ in range(ACTION_HISTORY_FRAMES):
            self.action_history.append(zero_action.copy())

    def append(self, env: BramTripodEnv, action: np.ndarray) -> None:
        self.imu_history.append(imu_quat(env))
        self.action_history.append(action.astype(np.float32).copy())
        self.previous_action[:] = action

    def observation(self, command_scalar: float) -> np.ndarray:
        return np.concatenate(
            [
                np.concatenate(list(self.imu_history), axis=0),
                np.concatenate(list(self.action_history), axis=0),
                np.array([command_scalar], dtype=np.float32),
            ]
        ).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize movement_v2 online primitive policies in MuJoCo."
    )
    parser.add_argument(
        "--case",
        choices=(*CASES.keys(), "suite"),
        default="suite",
        help="Primitive to visualize. 'suite' runs all base primitives once.",
    )
    parser.add_argument("--forward-checkpoint", type=Path, default=DEFAULT_FORWARD)
    parser.add_argument("--backward-checkpoint", type=Path, default=DEFAULT_BACKWARD)
    parser.add_argument("--yaw-checkpoint", type=Path, default=DEFAULT_YAW)
    parser.add_argument(
        "--seconds",
        type=float,
        default=8.0,
        help="Seconds to run each selected case before switching or exiting.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=25,
        help="MuJoCo sim steps per actor step. 25 is 20 Hz with the current XML.",
    )
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--repeat",
        action="store_true",
        help="Loop the selected case/suite until the viewer is closed.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=20,
        help="Print a status row every N actor steps. Set 0 to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    maybe_relaunch_with_mjpython(args)
    agents = load_agents(args)
    selected = selected_cases(args)
    if args.headless:
        run_headless(agents, selected, args)
    else:
        run_viewer(agents, selected, args)


def maybe_relaunch_with_mjpython(args: argparse.Namespace) -> None:
    if args.headless or platform.system() != "Darwin" or os.environ.get("MJPYTHON_BIN"):
        return
    if Path(sys.executable).name == "mjpython":
        return
    mjpython = Path(sys.executable).with_name("mjpython")
    if not mjpython.exists():
        raise RuntimeError(
            "MuJoCo viewer requires mjpython on macOS. Run this command with "
            f"`mjpython {' '.join(sys.argv)}`."
        )
    print(f"macOS MuJoCo viewer requires mjpython; relaunching with {mjpython}")
    os.execv(str(mjpython), [str(mjpython), *sys.argv])


def selected_cases(args: argparse.Namespace) -> list[VisualCase]:
    if args.case == "suite":
        return [CASES[name] for name in SUITE_CASES]
    return [CASES[args.case]]


def load_agents(args: argparse.Namespace) -> dict[str, ActorCritic]:
    return {
        "forward": load_agent(args.forward_checkpoint, "forward"),
        "backward": load_agent(args.backward_checkpoint, "backward"),
        "yaw": load_agent(args.yaw_checkpoint, "yaw"),
    }


def load_agent(checkpoint: Path, primitive: str) -> ActorCritic:
    payload = torch.load(checkpoint, map_location="cpu")
    checkpoint_primitive = str(payload.get("args", {}).get("primitive", ""))
    if checkpoint_primitive != primitive:
        raise ValueError(
            f"{checkpoint} is primitive={checkpoint_primitive!r}, expected {primitive!r}"
        )
    if payload.get("env_command_mode") != YAW_ENV_COMMAND_MODE:
        raise ValueError(
            f"{checkpoint} env_command_mode={payload.get('env_command_mode')!r}; "
            f"expected {YAW_ENV_COMMAND_MODE!r}"
        )
    if int(payload.get("obs_dim", -1)) != OBS_DIM:
        raise ValueError(f"{checkpoint} obs_dim={payload.get('obs_dim')}, expected {OBS_DIM}")
    action_dim = int(payload.get("action_dim", -1))
    if action_dim != SERVO_COMMAND_DIM:
        raise ValueError(
            f"{checkpoint} action_dim={action_dim}, expected {SERVO_COMMAND_DIM}"
        )
    hidden_size = int(payload.get("args", {}).get("hidden_size", 64))
    agent = ActorCritic(OBS_DIM, SERVO_COMMAND_DIM, hidden_size)
    agent.load_state_dict(payload["model_state_dict"])
    agent.eval()
    print(
        f"loaded {primitive:8s} checkpoint={checkpoint} "
        f"score={float(payload.get('eval_score', float('nan'))):+.3f} "
        f"dist={float(payload.get('eval_primary_distance', float('nan'))):+.4f}"
    )
    return agent


def make_env(args: argparse.Namespace, first_case: VisualCase) -> BramTripodEnv:
    episode_seconds = max(float(args.seconds), 0.05)
    return BramTripodEnv(
        frame_skip=args.frame_skip,
        episode_seconds=episode_seconds,
        randomize_reset=False,
        domain_randomization=False,
        randomize_command=False,
        command_forward=first_case.forward_cmd,
        command_yaw_rate=first_case.yaw_cmd,
    )


def run_headless(
    agents: dict[str, ActorCritic],
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
                agents[case.primitive],
                case,
                args,
                seed=args.seed + case_index,
                viewer=None,
            )
            print_summary(case, stats)
    finally:
        env.close()


def run_viewer(
    agents: dict[str, ActorCritic],
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
                        agents[case.primitive],
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
    agent: ActorCritic,
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
    command_scalar = case.yaw_cmd if case.primitive == "yaw" else 0.0
    for step in range(env.max_steps):
        if viewer is not None and not viewer.is_running():
            break
        started = time.monotonic()
        obs = history.observation(command_scalar)
        action = deterministic_action(agent, obs)
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


def deterministic_action(agent: ActorCritic, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs[None, :], dtype=torch.float32)
        action = agent.deterministic_action(obs_tensor)
    return action.cpu().numpy()[0].astype(np.float32)


def imu_quat(env: BramTripodEnv) -> np.ndarray:
    quat = np.asarray(env.data.qpos[3:7], dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if norm > 1.0e-6:
        quat = quat / norm
    return quat.astype(np.float32)


def configure_viewer(viewer) -> None:
    import mujoco

    for flag in (
        mujoco.mjtVisFlag.mjVIS_INERTIA,
        mujoco.mjtVisFlag.mjVIS_SCLINERTIA,
        mujoco.mjtVisFlag.mjVIS_CONTACTPOINT,
        mujoco.mjtVisFlag.mjVIS_CONTACTFORCE,
    ):
        viewer.opt.flags[int(flag)] = 0


def print_status(
    case: VisualCase,
    step: int,
    env: BramTripodEnv,
    stats: RolloutStats,
) -> None:
    info = stats.final_info or {}
    print(
        f"{case.name:13s} step={step:04d} t={step * env.dt:5.2f} "
        f"xy=({float(info.get('x_distance', 0.0)):+.3f},"
        f"{float(info.get('y_distance', 0.0)):+.3f}) "
        f"line={float(info.get('line_distance', 0.0)):+.3f} "
        f"yaw={float(info.get('yaw_distance', 0.0)):+.3f} "
        f"tilt={stats.max_tilt_rad:.3f}",
        flush=True,
    )


def print_summary(case: VisualCase, stats: RolloutStats) -> None:
    info = stats.final_info or {}
    x_distance = float(info.get("x_distance", 0.0))
    y_distance = float(info.get("y_distance", 0.0))
    yaw_distance = float(info.get("yaw_distance", 0.0))
    line_distance = float(info.get("line_distance", 0.0))
    if case.primitive == "yaw":
        primary = yaw_distance
        path_waste = float(np.hypot(x_distance, y_distance))
    else:
        command_sign = 1.0 if case.forward_cmd >= 0.0 else -1.0
        net_distance = float(np.hypot(x_distance, y_distance))
        direction_error = translation_displacement_error(
            x_distance,
            y_distance,
            command_sign,
        )
        primary = net_distance * translation_cone_gate(direction_error)
        path_waste = max(0.0, stats.path_length - net_distance)
    action_delta_rms = float(np.sqrt(np.mean(stats.action_delta_squares or [0.0])))
    print(
        f"summary {case.name:13s} "
        f"primary={primary:+.4f} "
        f"line={line_distance:+.4f} "
        f"yaw={yaw_distance:+.4f} "
        f"xy=({x_distance:+.3f},{y_distance:+.3f}) "
        f"waste={path_waste:.3f} "
        f"tilt={stats.max_tilt_rad:.3f} "
        f"min_h={stats.min_height_m:.3f} "
        f"dact={action_delta_rms:.4f} "
        f"len={stats.length} "
        f"term={stats.terminated}",
        flush=True,
    )


if __name__ == "__main__":
    main()
