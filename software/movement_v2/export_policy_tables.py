from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


GAIT_DISCOVERY_DIR = Path(__file__).resolve().parents[1] / "gait_discovery"
if str(GAIT_DISCOVERY_DIR) not in sys.path:
    sys.path.insert(0, str(GAIT_DISCOVERY_DIR))

from train_ppo import ActorCritic  # noqa: E402
from yaw_env import BramV2PrimitiveEnv, YAW_ENV_COMMAND_MODE  # noqa: E402


DEFAULT_OUT_DIR = Path("software/movement_v2/exports/policy_tables")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export firmware action tables from movement_v2 PPO primitive checkpoints."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--primitive",
        choices=("auto", "forward", "backward", "yaw"),
        default="auto",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--episode-seconds", type=float, default=8.0)
    parser.add_argument("--frame-skip", type=int, default=25)
    parser.add_argument("--output-hz", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--allow-mode-mismatch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = torch.load(args.checkpoint, map_location="cpu")
    primitive = resolve_primitive(args, checkpoint_payload)
    env = BramV2PrimitiveEnv(
        episode_seconds=args.episode_seconds,
        frame_skip=args.frame_skip,
        randomize_reset=False,
        domain_randomization=False,
        randomize_yaw_command=False,
        primitive=primitive,
    )
    try:
        agent = load_agent(args.checkpoint, checkpoint_payload, env, args.allow_mode_mismatch)
        if primitive == "yaw":
            outputs = {
                "yaw_left_table": export_one(args, env, agent, "yaw-left", 1.0),
                "yaw_right_table": export_one(args, env, agent, "yaw-right", -1.0),
            }
        else:
            outputs = {
                f"{primitive}_table": export_one(args, env, agent, primitive, 0.0),
            }
    finally:
        env.close()

    summary_path = args.out_dir / f"{primitive}_summary.json"
    summary_path.write_text(json.dumps(outputs, indent=2) + "\n")
    print(f"primitive_table_summary={summary_path}")


def load_agent(
    checkpoint: Path,
    payload: dict[str, Any],
    env: BramV2PrimitiveEnv,
    allow_mode_mismatch: bool,
) -> ActorCritic:
    checkpoint_mode = payload.get("env_command_mode")
    if checkpoint_mode != YAW_ENV_COMMAND_MODE and not allow_mode_mismatch:
        raise ValueError(
            f"{checkpoint} has env_command_mode={checkpoint_mode!r}; "
            f"expected {YAW_ENV_COMMAND_MODE!r}."
        )
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    if (int(payload.get("obs_dim", -1)), int(payload.get("action_dim", -1))) != (
        obs_dim,
        action_dim,
    ):
        raise ValueError(
            f"{checkpoint} obs/action dims do not match env: "
            f"{payload.get('obs_dim')}/{payload.get('action_dim')} vs "
            f"{obs_dim}/{action_dim}."
        )
    hidden_size = int(payload.get("args", {}).get("hidden_size", 64))
    agent = ActorCritic(obs_dim, action_dim, hidden_size)
    agent.load_state_dict(payload["model_state_dict"])
    agent.eval()
    return agent


def export_one(
    args: argparse.Namespace,
    env: BramV2PrimitiveEnv,
    agent: ActorCritic,
    primitive: str,
    yaw_command: float,
) -> dict[str, Any]:
    options = {"randomize": False}
    if env.primitive == "yaw":
        options["yaw_cmd"] = yaw_command
    obs, _ = env.reset(seed=args.seed, options=options)
    policy_actions: list[np.ndarray] = []
    infos: list[dict[str, float]] = []
    with torch.no_grad():
        for _ in range(env.max_steps):
            obs_tensor = torch.as_tensor(obs[None, :], dtype=torch.float32)
            action = agent.deterministic_action(obs_tensor).cpu().numpy()[0].astype(
                np.float32
            )
            policy_actions.append(action)
            obs, _, terminated, truncated, info = env.step(action)
            infos.append(scalar_info(info))
            if terminated or truncated:
                break

    firmware_actions = interpolate_actions(
        np.asarray(policy_actions, dtype=np.float32),
        policy_dt=env.dt,
        output_dt=1.0 / args.output_hz,
        episode_seconds=args.episode_seconds,
    )
    payload = {
        "primitive": primitive,
        "checkpoint": str(args.checkpoint),
        "env_command_mode": YAW_ENV_COMMAND_MODE,
        "yaw_command": float(yaw_command),
        "policy_dt": env.dt,
        "policy_hz": 1.0 / env.dt,
        "control_hz": float(args.output_hz),
        "episode_seconds": float(args.episode_seconds),
        "actions": firmware_actions.astype(float).tolist(),
        "final_info": infos[-1] if infos else {},
    }
    path = args.out_dir / f"{primitive}_policy_table.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{primitive}_table={path}")
    return {
        "path": str(path),
        "rows": len(payload["actions"]),
        "final_info": payload["final_info"],
    }


def interpolate_actions(
    actions: np.ndarray,
    *,
    policy_dt: float,
    output_dt: float,
    episode_seconds: float,
) -> np.ndarray:
    if actions.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    policy_times = np.arange(actions.shape[0], dtype=np.float64) * policy_dt
    output_count = max(1, int(round(episode_seconds / output_dt)))
    output_times = np.arange(output_count, dtype=np.float64) * output_dt
    out = np.zeros((output_count, actions.shape[1]), dtype=np.float32)
    for servo in range(actions.shape[1]):
        out[:, servo] = np.interp(
            output_times,
            policy_times,
            actions[:, servo],
            left=actions[0, servo],
            right=actions[-1, servo],
        )
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def scalar_info(info: dict[str, Any]) -> dict[str, float]:
    keep = (
        "line_distance",
        "yaw_distance",
        "x_distance",
        "y_distance",
        "planar_speed",
        "yaw_rate",
        "height",
        "level_tilt_rad",
    )
    return {key: float(info.get(key, 0.0)) for key in keep}


def resolve_primitive(args: argparse.Namespace, payload: dict[str, Any]) -> str:
    if args.primitive != "auto":
        return args.primitive
    primitive = str(payload.get("args", {}).get("primitive", "yaw"))
    if primitive not in ("forward", "backward", "yaw"):
        raise ValueError(
            f"Cannot infer movement_v2 primitive from checkpoint args: {primitive!r}"
        )
    return primitive


if __name__ == "__main__":
    main()
