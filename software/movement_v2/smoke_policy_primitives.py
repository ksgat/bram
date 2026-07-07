from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


GAIT_DISCOVERY_DIR = Path(__file__).resolve().parents[1] / "gait_discovery"
if str(GAIT_DISCOVERY_DIR) not in sys.path:
    sys.path.insert(0, str(GAIT_DISCOVERY_DIR))

from train_ppo import ActorCritic  # noqa: E402
from yaw_env import BramV2PrimitiveEnv, YAW_ENV_COMMAND_MODE  # noqa: E402


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
class SmokeResult:
    name: str
    primitive: str
    yaw_cmd: float
    reward: float
    primary_distance: float
    line_distance: float
    yaw_distance: float
    x_distance: float
    y_distance: float
    path_waste: float
    max_tilt_rad: float
    min_height_m: float
    action_delta_rms: float
    length: int
    terminated: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Closed-loop movement_v2 policy smoke test using IMU/action history."
    )
    parser.add_argument("--forward-checkpoint", type=Path, default=DEFAULT_FORWARD)
    parser.add_argument("--backward-checkpoint", type=Path, default=DEFAULT_BACKWARD)
    parser.add_argument("--yaw-checkpoint", type=Path, default=DEFAULT_YAW)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--frame-skip", type=int, default=25)
    parser.add_argument("--seed", type=int, default=31)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = (
        ("forward", "forward", args.forward_checkpoint, 0.0),
        ("backward", "backward", args.backward_checkpoint, 0.0),
        ("yaw_pos", "yaw", args.yaw_checkpoint, 1.0),
        ("yaw_neg", "yaw", args.yaw_checkpoint, -1.0),
        ("yaw_pos_half", "yaw", args.yaw_checkpoint, 0.5),
        ("yaw_neg_half", "yaw", args.yaw_checkpoint, -0.5),
    )
    results = [
        rollout(args, index, name, primitive, checkpoint, yaw_cmd)
        for index, (name, primitive, checkpoint, yaw_cmd) in enumerate(cases)
    ]
    print_results(results)


def load_agent(checkpoint: Path, env: BramV2PrimitiveEnv) -> ActorCritic:
    payload = torch.load(checkpoint, map_location="cpu")
    if payload.get("env_command_mode") != YAW_ENV_COMMAND_MODE:
        raise ValueError(
            f"{checkpoint} env_command_mode={payload.get('env_command_mode')!r}; "
            f"expected {YAW_ENV_COMMAND_MODE!r}"
        )
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    if int(payload.get("obs_dim", -1)) != obs_dim or int(payload.get("action_dim", -1)) != action_dim:
        raise ValueError(f"{checkpoint} obs/action dims do not match movement_v2 env")
    hidden_size = int(payload.get("args", {}).get("hidden_size", 64))
    agent = ActorCritic(obs_dim, action_dim, hidden_size)
    agent.load_state_dict(payload["model_state_dict"])
    agent.eval()
    return agent


def rollout(
    args: argparse.Namespace,
    index: int,
    name: str,
    primitive: str,
    checkpoint: Path,
    yaw_cmd: float,
) -> SmokeResult:
    env = BramV2PrimitiveEnv(
        primitive=primitive,
        randomize_reset=False,
        randomize_yaw_command=False,
        episode_seconds=args.seconds,
        frame_skip=args.frame_skip,
    )
    try:
        agent = load_agent(checkpoint, env)
        options = {"randomize": False}
        if primitive == "yaw":
            options["yaw_cmd"] = yaw_cmd
        obs, _ = env.reset(seed=args.seed + 1000 * index, options=options)
        previous_action = np.zeros(env.action_space.shape[0], dtype=np.float32)
        action_delta_squares: list[float] = []
        total_reward = 0.0
        max_tilt = 0.0
        min_height = float("inf")
        final_info = {}
        terminated = False
        truncated = False
        length = 0
        with torch.no_grad():
            for step in range(env.max_steps):
                obs_tensor = torch.as_tensor(obs[None, :], dtype=torch.float32)
                action = agent.deterministic_action(obs_tensor).cpu().numpy()[0]
                action_delta_squares.append(float(np.mean(np.square(action - previous_action))))
                previous_action = action.astype(np.float32)
                obs, reward, terminated, truncated, final_info = env.step(previous_action)
                total_reward += float(reward)
                max_tilt = max(max_tilt, float(final_info.get("level_tilt_rad", 0.0)))
                min_height = min(min_height, float(final_info.get("height", float("inf"))))
                length = step + 1
                if terminated or truncated:
                    break

        x_distance = float(final_info.get("x_distance", 0.0))
        y_distance = float(final_info.get("y_distance", 0.0))
        if primitive == "yaw":
            primary_distance = float(final_info.get("yaw_distance", 0.0))
            path_waste = float(np.hypot(x_distance, y_distance))
        else:
            primary_distance = float(
                final_info.get(
                    "v2_translation_primary_distance",
                    final_info.get("line_distance", 0.0),
                )
            )
            path_waste = float(final_info.get("v2_translation_path_waste", 0.0))
        return SmokeResult(
            name=name,
            primitive=primitive,
            yaw_cmd=float(yaw_cmd),
            reward=float(total_reward),
            primary_distance=primary_distance,
            line_distance=float(final_info.get("line_distance", 0.0)),
            yaw_distance=float(final_info.get("yaw_distance", 0.0)),
            x_distance=x_distance,
            y_distance=y_distance,
            path_waste=path_waste,
            max_tilt_rad=max_tilt,
            min_height_m=min_height,
            action_delta_rms=float(np.sqrt(np.mean(action_delta_squares))),
            length=length,
            terminated=bool(terminated),
        )
    finally:
        env.close()


def print_results(results: list[SmokeResult]) -> None:
    for result in results:
        print(
            f"{result.name:13s} "
            f"primary={result.primary_distance:+.4f} "
            f"line={result.line_distance:+.4f} "
            f"yaw={result.yaw_distance:+.4f} "
            f"xy=({result.x_distance:+.3f},{result.y_distance:+.3f}) "
            f"waste={result.path_waste:.3f} "
            f"tilt={result.max_tilt_rad:.3f} "
            f"min_h={result.min_height_m:.3f} "
            f"dact={result.action_delta_rms:.4f} "
            f"len={result.length} "
            f"term={result.terminated}"
        )


if __name__ == "__main__":
    main()
