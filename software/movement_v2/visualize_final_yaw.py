from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
GAIT_DISCOVERY_DIR = REPO_ROOT / "software" / "gait_discovery"
if str(GAIT_DISCOVERY_DIR) not in sys.path:
    sys.path.insert(0, str(GAIT_DISCOVERY_DIR))

from train_ppo import ActorCritic  # noqa: E402
from yaw_residual_env import BramV2YawResidualEnv  # noqa: E402
from visualize_policy_primitives import configure_viewer  # noqa: E402


DEFAULT_EXPORT = REPO_ROOT / "software" / "movement_v2" / "exports" / "final_yaw_20260713"


@dataclass(frozen=True)
class YawCase:
    name: str
    yaw_cmd: float
    use_residual: bool
    residual_key: str | None
    default_frame_skip: int


CASES: dict[str, YawCase] = {
    "yaw_right_residual": YawCase("yaw_right_residual", -1.0, True, "yaw_right_full", 10),
    "yaw_right_base": YawCase("yaw_right_base", -1.0, False, None, 10),
    "yaw_left_residual": YawCase("yaw_left_residual", 1.0, True, "yaw_left_full", 50),
    "yaw_left_cpg": YawCase("yaw_left_cpg", 1.0, False, None, 50),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the packaged final movement_v2 yaw artifacts."
    )
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--case", choices=(*CASES.keys(), "suite"), default="yaw_right_residual")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=None,
        help="MuJoCo sim steps per control action. Defaults to each case's packaged rate.",
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--repeat", action="store_true")
    parser.add_argument("--print-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    maybe_relaunch_with_mjpython(args)
    manifest = load_manifest(args.export_dir)
    cases = selected_cases(args)
    if args.headless:
        run_headless(args, manifest, cases)
    else:
        run_viewer(args, manifest, cases)


def maybe_relaunch_with_mjpython(args: argparse.Namespace) -> None:
    if args.headless or platform.system() != "Darwin":
        return
    if Path(sys.executable).name == "mjpython" or os.environ.get("MJPYTHON_BIN"):
        return
    candidate = Path(sys.executable).with_name("mjpython")
    if candidate.exists():
        os.execv(str(candidate), [str(candidate), *sys.argv])
    raise RuntimeError(
        "MuJoCo viewer on macOS requires mjpython. Run this script with mjpython "
        "or install mujoco's viewer launcher in the active venv."
    )


def load_manifest(export_dir: Path) -> dict[str, Any]:
    return json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))


def selected_cases(args: argparse.Namespace) -> list[YawCase]:
    if args.case == "suite":
        return [CASES["yaw_right_residual"], CASES["yaw_left_residual"]]
    return [CASES[args.case]]


def load_residual_agent(path: Path) -> ActorCritic:
    payload = torch.load(path, map_location="cpu")
    obs_dim = int(payload["obs_dim"])
    action_dim = int(payload["action_dim"])
    hidden_size = int(payload.get("args", {}).get("hidden_size", 64))
    log_std_init = float(payload.get("args", {}).get("log_std_init", -2.0))
    agent = ActorCritic(obs_dim, action_dim, hidden_size, log_std_init)
    agent.load_state_dict(payload["model_state_dict"])
    agent.eval()
    return agent


def load_agent_for_case(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    case: YawCase,
) -> ActorCritic | None:
    if not case.use_residual:
        return None
    if case.residual_key is None:
        raise ValueError(f"{case.name} is residual but has no manifest key.")
    return load_residual_agent(args.export_dir / manifest[case.residual_key]["artifact"])


def make_env(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    case: YawCase,
) -> BramV2YawResidualEnv:
    left_base = manifest["yaw_left"]
    left_params = args.export_dir / left_base["artifact"]
    left_table = args.export_dir / left_base.get(
        "table_artifact_not_promoted",
        "yaw-left_base_table.json",
    )
    residual_config = manifest.get(case.residual_key or "", {})
    frame_skip = int(args.frame_skip or residual_config.get("frame_skip", case.default_frame_skip))
    return BramV2YawResidualEnv(
        left_table=left_table,
        right_table=args.export_dir / manifest["yaw_right_full"]["base_table"],
        left_params=left_params,
        frame_skip=frame_skip,
        episode_seconds=args.seconds,
        randomize_reset=False,
        domain_randomization=False,
        randomize_yaw_command=False,
        yaw_command=case.yaw_cmd,
        residual_limit=float(residual_config.get("residual_limit", 0.18)),
        target_yaw_rate=0.36,
        final_drift_limit_m=0.04,
        mean_drift_limit_m=0.025,
        max_drift_limit_m=0.04,
        slew_limit=float(residual_config.get("slew_limit", 0.25)),
    )


def run_headless(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    cases: list[YawCase],
) -> None:
    for index, case in enumerate(cases):
        agent = load_agent_for_case(args, manifest, case)
        env = make_env(args, manifest, case)
        try:
            stats = rollout_case(env, agent, case, args, seed=args.seed + index, viewer=None)
            print_summary(case, stats)
        finally:
            env.close()


def run_viewer(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    cases: list[YawCase],
) -> None:
    import mujoco.viewer

    pass_index = 0
    while True:
        for case_index, case in enumerate(cases):
            agent = load_agent_for_case(args, manifest, case)
            env = make_env(args, manifest, case)
            tripod_env = env.env.env
            try:
                with mujoco.viewer.launch_passive(tripod_env.model, tripod_env.data) as viewer:
                    configure_viewer(viewer)
                    while viewer.is_running():
                        stats = rollout_case(
                            env,
                            agent,
                            case,
                            args,
                            seed=args.seed + pass_index * len(cases) + case_index,
                            viewer=viewer,
                        )
                        print_summary(case, stats)
                        if not args.repeat:
                            break
            finally:
                env.close()
        if not args.repeat:
            return
        pass_index += 1


def rollout_case(
    env: BramV2YawResidualEnv,
    agent: ActorCritic | None,
    case: YawCase,
    args: argparse.Namespace,
    *,
    seed: int,
    viewer,
) -> dict[str, float | int | bool]:
    obs, _ = env.reset(seed=seed, options={"randomize": False, "yaw_cmd": case.yaw_cmd})
    print(f"case={case.name} yaw={case.yaw_cmd:+.2f} residual={int(case.use_residual)}", flush=True)
    total_reward = 0.0
    drift_values: list[float] = []
    max_tilt = 0.0
    min_height = float("inf")
    final_info: dict[str, Any] = {}
    terminated = False
    truncated = False
    length = 0
    previous_action = np.zeros(3, dtype=np.float32)
    action_delta_squares: list[float] = []
    for step in range(env.max_steps):
        if viewer is not None and not viewer.is_running():
            break
        started = time.monotonic()
        if case.use_residual:
            if agent is None:
                raise ValueError(f"{case.name} requires a residual agent.")
            residual = deterministic_action(agent, obs)
        else:
            residual = np.zeros(3, dtype=np.float32)
        obs, reward, terminated, truncated, final_info = env.step(residual)
        final_action = np.asarray(
            final_info.get("v2_residual_final_action", [0.0, 0.0, 0.0]),
            dtype=np.float32,
        )
        delta = final_action - previous_action
        previous_action = final_action
        action_delta_squares.append(float(np.mean(np.square(delta))))
        total_reward += float(reward)
        drift = float(
            np.hypot(
                float(final_info.get("x_distance", 0.0)),
                float(final_info.get("y_distance", 0.0)),
            )
        )
        drift_values.append(drift)
        max_tilt = max(max_tilt, float(final_info.get("level_tilt_rad", 0.0)))
        min_height = min(min_height, float(final_info.get("height", float("inf"))))
        length = step + 1
        if viewer is not None:
            viewer.sync()
        if args.print_every > 0 and step % args.print_every == 0:
            print_status(case, step, final_info, drift_values)
        if terminated or truncated:
            break
        sleep_time = (env.dt / max(args.speed, 1.0e-6)) - (time.monotonic() - started)
        if viewer is not None and sleep_time > 0.0:
            time.sleep(sleep_time)
    planar_drift = drift_values[-1] if drift_values else 0.0
    mean_drift = float(np.mean(drift_values)) if drift_values else 0.0
    max_drift = float(np.max(drift_values)) if drift_values else 0.0
    return {
        "reward": total_reward,
        "yaw_distance": float(final_info.get("yaw_distance", 0.0)),
        "planar_drift": planar_drift,
        "mean_planar_drift": mean_drift,
        "max_planar_drift": max_drift,
        "max_tilt_rad": max_tilt,
        "min_height_m": min_height,
        "action_delta_rms": float(np.sqrt(np.mean(action_delta_squares))) if action_delta_squares else 0.0,
        "length": length,
        "terminated": bool(terminated),
    }


def deterministic_action(agent: ActorCritic, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs[None, :], dtype=torch.float32)
        action = agent.deterministic_action(obs_tensor)
    return action.cpu().numpy()[0].astype(np.float32)


def print_status(
    case: YawCase,
    step: int,
    info: dict[str, Any],
    drift_values: list[float],
) -> None:
    print(
        f"  {case.name} step={step:03d} "
        f"yaw={float(info.get('yaw_distance', 0.0)):+.3f} "
        f"drift={drift_values[-1]:.4f} "
        f"max={float(np.max(drift_values)):.4f} "
        f"h={float(info.get('height', 0.0)):.3f}",
        flush=True,
    )


def print_summary(case: YawCase, stats: dict[str, float | int | bool]) -> None:
    print(
        f"summary {case.name}: "
        f"yaw={float(stats['yaw_distance']):+.4f} "
        f"drift={float(stats['planar_drift']):.4f} "
        f"mean={float(stats['mean_planar_drift']):.4f} "
        f"max={float(stats['max_planar_drift']):.4f} "
        f"tilt={float(stats['max_tilt_rad']):.3f} "
        f"min_h={float(stats['min_height_m']):.3f} "
        f"dact={float(stats['action_delta_rms']):.4f} "
        f"len={int(stats['length'])} "
        f"term={int(bool(stats['terminated']))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
