from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

GAIT_DISCOVERY_DIR = Path(__file__).resolve().parents[1] / "gait_discovery"
if str(GAIT_DISCOVERY_DIR) not in sys.path:
    sys.path.insert(0, str(GAIT_DISCOVERY_DIR))

from train_ppo import ActorCritic
from yaw_env import (
    BramV2PrimitiveEnv,
    TILT_FREE_RAD,
    TRANSLATION_MAX_CREDIT_CONE_RAD,
    YAW_TARGET_RATE_PER_COMMAND,
    YAW_ENV_COMMAND_MODE,
)
from bram_controller import (  # noqa: E402
    BramGridController,
    DEFAULT_BACKWARD_GAIT,
    DEFAULT_FORWARD_GAIT,
    DEFAULT_YAW_LEFT_TABLE,
    DEFAULT_YAW_RIGHT_TABLE,
)


EVAL_COMMANDS: tuple[tuple[str, float], ...] = (
    ("yaw_neg1", -1.0),
    ("yaw_neg05", -0.5),
    ("yaw_pos05", 0.5),
    ("yaw_pos1", 1.0),
)


@dataclass(frozen=True)
class EvalResult:
    command: str
    yaw_command: float
    reward: float
    score: float
    line_distance: float
    yaw_distance: float
    planar_drift: float
    x_distance: float
    y_distance: float
    max_tilt_rad: float
    min_height_m: float
    action_delta_rms: float
    action_accel_rms: float
    mean_action_delta_abs: float
    mean_action_accel_abs: float
    mean_abs_action: float
    mean_planar_drift: float
    max_planar_drift: float
    mean_abs_planar_speed: float
    max_abs_planar_speed: float
    mean_abs_yaw_error: float
    mean_abs_roll_pitch_rate: float
    mean_support_deficit: float
    mean_contact_foot_speed: float
    mean_height_warning_deficit: float
    max_height_warning_deficit: float
    mean_height_deficit: float
    max_height_deficit: float
    target_primary_distance: float
    length: int
    terminated: bool


@dataclass(frozen=True)
class EvalStats:
    reward: float
    score: float
    primary_distance: float
    planar_drift: float
    length: float
    per_command: dict[str, dict[str, float | bool | int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train movement_v2 PPO primitives: forward/back specialists and the "
            "yaw-only command-conditioned generalist."
        )
    )
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument(
        "--primitive",
        choices=("forward", "backward", "yaw"),
        default="yaw",
        help="Forward/backward are fixed-command specialists; yaw randomizes command.",
    )
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.20)
    parser.add_argument("--entropy-coef", type=float, default=0.004)
    parser.add_argument("--value-coef", type=float, default=0.50)
    parser.add_argument("--max-grad-norm", type=float, default=0.50)
    parser.add_argument("--target-kl", type=float, default=0.06)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--log-std-init", type=float, default=-1.35)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--episode-seconds", type=float, default=8.0)
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=25,
        help="MuJoCo sim steps per policy action. With the current XML timestep, 25 is 20 Hz.",
    )
    parser.add_argument("--yaw-min-magnitude", type=float, default=0.35)
    parser.add_argument("--idle-probability", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("software/movement_v2/runs/rl_primitives"),
    )
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--domain-randomization-strength", type=float, default=0.15)
    parser.add_argument("--randomize-reset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--bc-epochs",
        type=int,
        default=0,
        help="Optional behavior-cloning warm start epochs before PPO.",
    )
    parser.add_argument(
        "--bc-only",
        action="store_true",
        help="Save/evaluate the behavior-cloned policy and skip PPO updates.",
    )
    parser.add_argument("--bc-steps", type=int, default=400)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--bc-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--bc-dagger-iters", type=int, default=0)
    parser.add_argument("--bc-dagger-epochs", type=int, default=40)
    parser.add_argument(
        "--bc-anchor-coef",
        type=float,
        default=0.0,
        help=(
            "Auxiliary actor MSE coefficient against the gait_discovery "
            "teacher table during PPO updates. Useful for yaw at low rate."
        ),
    )
    parser.add_argument(
        "--bc-anchor-batch-size",
        type=int,
        default=256,
        help="Batch size for the optional PPO-time teacher anchor loss.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional movement_v2 actor checkpoint to fine-tune before BC/PPO.",
    )
    parser.add_argument(
        "--bc-controller-export",
        type=Path,
        default=None,
        help="Optional controller export to use as the BC teacher.",
    )
    parser.add_argument("--bc-forward-gait", type=Path, default=None)
    parser.add_argument("--bc-backward-gait", type=Path, default=None)
    parser.add_argument("--bc-yaw-left-table", type=Path, default=None)
    parser.add_argument("--bc-yaw-right-table", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    run_dir = args.output_dir / time.strftime(
        f"{args.primitive}_primitive_%Y%m%d_%H%M%S"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    checkpoint_path = run_dir / "policy.pt"
    best_checkpoint_path = run_dir / "policy_best.pt"

    envs = [make_env(args, randomize_yaw_command=True) for _ in range(args.num_envs)]
    obs_np = np.stack(
        [
            env.reset(seed=args.seed + index)[0]
            for index, env in enumerate(envs)
        ]
    ).astype(np.float32)
    obs_dim = obs_np.shape[1]
    action_dim = envs[0].action_space.shape[0]
    agent = ActorCritic(obs_dim, action_dim, args.hidden_size, args.log_std_init)
    initial_eval_stats: EvalStats | None = None
    if args.init_checkpoint is not None:
        load_initial_actor_checkpoint(agent, args.init_checkpoint, args)
        initial_eval_stats = evaluate_policy(agent, args)
        print(
            f"init_eval reward={initial_eval_stats.reward:.3f} "
            f"score={initial_eval_stats.score:.3f} "
            f"distance={initial_eval_stats.primary_distance:.4f} "
            f"drift={initial_eval_stats.planar_drift:.4f}"
        )
    if args.bc_epochs > 0:
        initial_eval_stats = behavior_clone_initial_policy(agent, args)
    if args.bc_only:
        if initial_eval_stats is None:
            raise ValueError("--bc-only requires --bc-epochs > 0")
        write_single_metrics_row(metrics_path, initial_eval_stats)
        save_checkpoint(checkpoint_path, agent, args, initial_eval_stats)
        save_checkpoint(best_checkpoint_path, agent, args, initial_eval_stats)
        print(
            f"bc_only_eval reward={initial_eval_stats.reward:.3f} "
            f"score={initial_eval_stats.score:.3f} "
            f"distance={initial_eval_stats.primary_distance:.4f} "
            f"drift={initial_eval_stats.planar_drift:.4f}"
        )
        print(f"saved_policy={checkpoint_path}")
        print(f"saved_best_policy={best_checkpoint_path}")
        print(f"metrics={metrics_path}")
        for env in envs:
            env.close()
        return
    anchor_dataset: tuple[torch.Tensor, torch.Tensor] | None = None
    if args.bc_anchor_coef > 0.0:
        anchor_obs_np, anchor_action_np = build_bc_dataset(args)
        anchor_dataset = (
            torch.as_tensor(anchor_obs_np, dtype=torch.float32),
            torch.as_tensor(anchor_action_np, dtype=torch.float32),
        )
        print(
            f"bc_anchor samples={anchor_obs_np.shape[0]} "
            f"coef={args.bc_anchor_coef:.4f} "
            f"teacher={bc_teacher_label(args)}"
        )
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    num_updates = max(1, args.total_steps // (args.num_envs * args.rollout_steps))
    actual_total_steps = num_updates * args.num_envs * args.rollout_steps
    obs_buf = torch.zeros((args.rollout_steps, args.num_envs, obs_dim), dtype=torch.float32)
    action_buf = torch.zeros((args.rollout_steps, args.num_envs, action_dim), dtype=torch.float32)
    logprob_buf = torch.zeros((args.rollout_steps, args.num_envs), dtype=torch.float32)
    reward_buf = torch.zeros((args.rollout_steps, args.num_envs), dtype=torch.float32)
    done_buf = torch.zeros((args.rollout_steps, args.num_envs), dtype=torch.float32)
    value_buf = torch.zeros((args.rollout_steps, args.num_envs), dtype=torch.float32)

    recent_returns: deque[float] = deque(maxlen=50)
    recent_primary_distance: deque[float] = deque(maxlen=50)
    recent_lengths: deque[int] = deque(maxlen=50)
    episode_returns = np.zeros(args.num_envs, dtype=np.float32)
    episode_primary_distance = np.zeros(args.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int32)

    print(
        f"training mode={YAW_ENV_COMMAND_MODE} primitive={args.primitive} "
        f"total_steps={actual_total_steps} "
        f"num_envs={args.num_envs} rollout_steps={args.rollout_steps} "
        f"obs_dim={obs_dim} action_dim={action_dim} "
        f"domain_randomization={args.domain_randomization} "
        f"randomize_reset={args.randomize_reset}"
    )

    best_score = -float("inf")
    global_step = 0
    start_time = time.perf_counter()
    with metrics_path.open("w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=metrics_fieldnames())
        writer.writeheader()
        if initial_eval_stats is not None:
            best_score = initial_eval_stats.score
            save_checkpoint(best_checkpoint_path, agent, args, initial_eval_stats)

        for update in range(1, num_updates + 1):
            for step in range(args.rollout_steps):
                global_step += args.num_envs
                obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32)
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(obs_tensor)
                action_np = action.cpu().numpy().astype(np.float32)

                next_obs = []
                rewards = []
                dones = []
                for env_index, env in enumerate(envs):
                    obs, reward, terminated, truncated, info = env.step(action_np[env_index])
                    done = terminated or truncated
                    episode_returns[env_index] += float(reward)
                    episode_primary_distance[env_index] = float(
                        primary_distance_info(info, args)
                    )
                    episode_lengths[env_index] += 1
                    if done:
                        recent_returns.append(float(episode_returns[env_index]))
                        recent_primary_distance.append(
                            float(episode_primary_distance[env_index])
                        )
                        recent_lengths.append(int(episode_lengths[env_index]))
                        episode_returns[env_index] = 0.0
                        episode_primary_distance[env_index] = 0.0
                        episode_lengths[env_index] = 0
                        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
                    next_obs.append(obs)
                    rewards.append(float(reward))
                    dones.append(done)

                obs_buf[step] = obs_tensor
                action_buf[step] = action
                logprob_buf[step] = logprob
                reward_buf[step] = torch.as_tensor(rewards, dtype=torch.float32)
                done_buf[step] = torch.as_tensor(dones, dtype=torch.float32)
                value_buf[step] = value
                obs_np = np.stack(next_obs).astype(np.float32)

            advantages, returns = compute_gae(agent, obs_np, reward_buf, done_buf, value_buf, args)
            batch_obs = obs_buf.reshape((-1, obs_dim))
            batch_actions = action_buf.reshape((-1, action_dim))
            batch_logprobs = logprob_buf.reshape(-1)
            batch_advantages = normalize_advantages(advantages.reshape(-1))
            batch_returns = returns.reshape(-1)

            policy_losses = []
            value_losses = []
            entropies = []
            approx_kls = []
            batch_size = args.num_envs * args.rollout_steps
            minibatch_size = min(args.minibatch_size, batch_size)
            stop_update = False
            for _ in range(args.update_epochs):
                indices = torch.randperm(batch_size)
                for start in range(0, batch_size, minibatch_size):
                    mb_idx = indices[start : start + minibatch_size]
                    _, new_logprob, entropy, new_value = agent.get_action_and_value(
                        batch_obs[mb_idx],
                        batch_actions[mb_idx],
                    )
                    log_ratio = new_logprob - batch_logprobs[mb_idx]
                    ratio = log_ratio.exp()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    pg_loss_1 = -batch_advantages[mb_idx] * ratio
                    pg_loss_2 = -batch_advantages[mb_idx] * torch.clamp(
                        ratio,
                        1.0 - args.clip_coef,
                        1.0 + args.clip_coef,
                    )
                    policy_loss = torch.max(pg_loss_1, pg_loss_2).mean()
                    value_loss = 0.5 * ((new_value - batch_returns[mb_idx]) ** 2).mean()
                    entropy_loss = entropy.mean()
                    anchor_loss = ppo_anchor_loss(agent, anchor_dataset, args)
                    loss = (
                        policy_loss
                        - args.entropy_coef * entropy_loss
                        + args.value_coef * value_loss
                        + args.bc_anchor_coef * anchor_loss
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                    policy_losses.append(float(policy_loss.detach()))
                    value_losses.append(float(value_loss.detach()))
                    entropies.append(float(entropy_loss.detach()))
                    approx_kls.append(float(approx_kl.detach()))
                    if args.target_kl > 0.0 and float(approx_kl.detach()) > args.target_kl:
                        stop_update = True
                        break
                if stop_update:
                    break

            should_eval = update == 1 or update == num_updates or update % args.eval_interval == 0
            eval_stats = EvalStats(
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                {},
            )
            if should_eval:
                eval_stats = evaluate_policy(agent, args)
                if eval_stats.score > best_score:
                    best_score = eval_stats.score
                    save_checkpoint(best_checkpoint_path, agent, args, eval_stats)

            elapsed = time.perf_counter() - start_time
            row = {
                "update": update,
                "global_step": global_step,
                "sps": int(global_step / max(elapsed, 1e-9)),
                "recent_return": mean_or_nan(recent_returns),
                "recent_primary_distance": mean_or_nan(recent_primary_distance),
                "recent_length": mean_or_nan(recent_lengths),
                "eval_reward": eval_stats.reward,
                "eval_score": eval_stats.score,
                "eval_primary_distance": eval_stats.primary_distance,
                "eval_planar_drift": eval_stats.planar_drift,
                "eval_length": eval_stats.length,
                "policy_loss": float(np.mean(policy_losses)),
                "value_loss": float(np.mean(value_losses)),
                "entropy": float(np.mean(entropies)),
                "approx_kl": float(np.mean(approx_kls)),
            }
            row.update(per_command_row(eval_stats))
            writer.writerow(row)
            log_file.flush()

            if update == 1 or update == num_updates or update % 10 == 0 or should_eval:
                eval_score = f"{eval_stats.score:.3f}" if should_eval else "skip"
                print(
                    f"update={update:04d}/{num_updates} step={global_step} "
                    f"recent_return={row['recent_return']:.3f} "
                    f"recent_dist={row['recent_primary_distance']:.4f} "
                    f"eval_score={eval_score} entropy={row['entropy']:.3f}"
                )

    final_stats = evaluate_policy(agent, args)
    save_checkpoint(checkpoint_path, agent, args, final_stats)
    print(
        f"final_eval reward={final_stats.reward:.3f} score={final_stats.score:.3f} "
        f"distance={final_stats.primary_distance:.4f} drift={final_stats.planar_drift:.4f}"
    )
    print(f"saved_policy={checkpoint_path}")
    print(f"saved_best_policy={best_checkpoint_path}")
    print(f"metrics={metrics_path}")
    for env in envs:
        env.close()


def make_env(args: argparse.Namespace, *, randomize_yaw_command: bool) -> BramV2PrimitiveEnv:
    return BramV2PrimitiveEnv(
        episode_seconds=args.episode_seconds,
        frame_skip=args.frame_skip,
        randomize_reset=args.randomize_reset,
        domain_randomization=args.domain_randomization,
        domain_randomization_strength=args.domain_randomization_strength,
        primitive=args.primitive,
        randomize_yaw_command=randomize_yaw_command and args.primitive == "yaw",
        yaw_min_magnitude=args.yaw_min_magnitude,
        idle_probability=args.idle_probability,
    )


def load_initial_actor_checkpoint(
    agent: ActorCritic,
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_mode = payload.get("env_command_mode")
    if checkpoint_mode != YAW_ENV_COMMAND_MODE:
        raise ValueError(
            f"{checkpoint_path} has env_command_mode={checkpoint_mode!r}; "
            f"expected {YAW_ENV_COMMAND_MODE!r}."
        )
    if int(payload.get("obs_dim", -1)) != agent.obs_dim or int(
        payload.get("action_dim", -1)
    ) != agent.action_dim:
        raise ValueError(
            f"{checkpoint_path} dims {payload.get('obs_dim')}/"
            f"{payload.get('action_dim')} do not match {agent.obs_dim}/"
            f"{agent.action_dim}."
        )
    checkpoint_hidden = int(payload.get("args", {}).get("hidden_size", agent.hidden_size))
    if checkpoint_hidden != agent.hidden_size:
        raise ValueError(
            f"{checkpoint_path} hidden_size={checkpoint_hidden}, "
            f"expected {agent.hidden_size}."
        )
    checkpoint_primitive = str(payload.get("args", {}).get("primitive", ""))
    if checkpoint_primitive != args.primitive:
        raise ValueError(
            f"{checkpoint_path} primitive={checkpoint_primitive!r}, "
            f"expected {args.primitive!r}."
        )
    agent.load_state_dict(payload["model_state_dict"])
    print(f"loaded_init_checkpoint={checkpoint_path}")


def behavior_clone_initial_policy(agent: ActorCritic, args: argparse.Namespace) -> EvalStats:
    obs_np, action_np = build_bc_dataset(args)
    optimizer = torch.optim.Adam(agent.actor.parameters(), lr=args.bc_learning_rate)
    agent.train()
    print(
        f"bc_start primitive={args.primitive} samples={obs_np.shape[0]} "
        f"epochs={args.bc_epochs} teacher={bc_teacher_label(args)}"
    )
    train_bc_epochs(agent, optimizer, obs_np, action_np, args, args.bc_epochs, label="bc")
    stats = evaluate_policy(agent, args)
    print(
        f"bc_eval reward={stats.reward:.3f} score={stats.score:.3f} "
        f"distance={stats.primary_distance:.4f} drift={stats.planar_drift:.4f}"
    )
    for dagger_iter in range(1, args.bc_dagger_iters + 1):
        dagger_obs, dagger_actions = build_bc_dataset(args, policy_agent=agent)
        obs_np = np.concatenate([obs_np, dagger_obs], axis=0)
        action_np = np.concatenate([action_np, dagger_actions], axis=0)
        print(
            f"bc_dagger_iter={dagger_iter}/{args.bc_dagger_iters} "
            f"samples={obs_np.shape[0]} new_samples={dagger_obs.shape[0]}"
        )
        train_bc_epochs(
            agent,
            optimizer,
            obs_np,
            action_np,
            args,
            args.bc_dagger_epochs,
            label=f"bc_dagger_{dagger_iter}",
        )
        stats = evaluate_policy(agent, args)
        print(
            f"bc_dagger_eval iter={dagger_iter} reward={stats.reward:.3f} "
            f"score={stats.score:.3f} distance={stats.primary_distance:.4f} "
            f"drift={stats.planar_drift:.4f}"
        )
    return stats


def train_bc_epochs(
    agent: ActorCritic,
    optimizer: torch.optim.Optimizer,
    obs_np: np.ndarray,
    action_np: np.ndarray,
    args: argparse.Namespace,
    epochs: int,
    *,
    label: str,
) -> None:
    obs = torch.as_tensor(obs_np, dtype=torch.float32)
    actions = torch.as_tensor(action_np, dtype=torch.float32)
    batch_size = min(args.bc_batch_size, obs.shape[0])
    for epoch in range(1, epochs + 1):
        indices = torch.randperm(obs.shape[0])
        losses = []
        for start in range(0, obs.shape[0], batch_size):
            batch = indices[start : start + batch_size]
            predicted = agent.deterministic_action(obs[batch])
            loss = torch.mean(torch.square(predicted - actions[batch]))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.actor.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 1 or epoch == epochs or epoch % 10 == 0:
            print(f"{label}_epoch={epoch:04d}/{epochs} mse={float(np.mean(losses)):.6f}")


def ppo_anchor_loss(
    agent: ActorCritic,
    anchor_dataset: tuple[torch.Tensor, torch.Tensor] | None,
    args: argparse.Namespace,
) -> torch.Tensor:
    if anchor_dataset is None or args.bc_anchor_coef <= 0.0:
        return next(agent.parameters()).new_tensor(0.0)
    obs, actions = anchor_dataset
    batch_size = min(int(args.bc_anchor_batch_size), obs.shape[0])
    indices = torch.randint(0, obs.shape[0], (batch_size,))
    predicted = agent.deterministic_action(obs[indices])
    return torch.mean(torch.square(predicted - actions[indices]))


def build_bc_dataset(
    args: argparse.Namespace,
    policy_agent: ActorCritic | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    controller = make_bc_controller(args)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for command_index, (forward_command, yaw_command) in enumerate(bc_commands(args)):
        env = make_env(args, randomize_yaw_command=False)
        try:
            options = {"randomize": False}
            if args.primitive == "yaw":
                options["yaw_cmd"] = yaw_command
            obs, _ = env.reset(seed=args.seed + 70_000 + command_index, options=options)
            steps = min(args.bc_steps, env.max_steps)
            for step in range(steps):
                teacher_step = int(round(step * env.dt / controller.dt))
                teacher_action = controller.action(
                    forward_command,
                    yaw_command,
                    teacher_step,
                    heading_error=float(env.env.heading_error),
                    yaw_rate=float(env.env.measured_gyro[2]),
                )
                observations.append(obs.copy())
                actions.append(np.clip(teacher_action, -1.0, 1.0).astype(np.float32))
                if policy_agent is None:
                    rollout_action = teacher_action
                else:
                    with torch.no_grad():
                        obs_tensor = torch.as_tensor(obs[None, :], dtype=torch.float32)
                        rollout_action = (
                            policy_agent.deterministic_action(obs_tensor)
                            .cpu()
                            .numpy()[0]
                            .astype(np.float32)
                        )
                obs, _, terminated, truncated, _ = env.step(rollout_action)
                if terminated or truncated:
                    break
        finally:
            env.close()
    if not observations:
        raise RuntimeError("BC warm start produced no samples.")
    return (
        np.stack(observations).astype(np.float32),
        np.stack(actions).astype(np.float32),
    )


def make_bc_controller(args: argparse.Namespace) -> BramGridController:
    if args.bc_controller_export is not None:
        return BramGridController.from_export(args.bc_controller_export)
    return BramGridController.from_files(
        forward_gait=args.bc_forward_gait or DEFAULT_FORWARD_GAIT,
        backward_gait=args.bc_backward_gait or DEFAULT_BACKWARD_GAIT,
        yaw_left_table=args.bc_yaw_left_table or DEFAULT_YAW_LEFT_TABLE,
        yaw_right_table=args.bc_yaw_right_table or DEFAULT_YAW_RIGHT_TABLE,
    )


def bc_commands(args: argparse.Namespace) -> tuple[tuple[float, float], ...]:
    if args.primitive == "forward":
        return ((1.0, 0.0),)
    if args.primitive == "backward":
        return ((-1.0, 0.0),)
    return (
        (0.0, -1.0),
        (0.0, -0.5),
        (0.0, 0.5),
        (0.0, 1.0),
    )


def bc_teacher_label(args: argparse.Namespace) -> str:
    if args.bc_controller_export is not None:
        return str(args.bc_controller_export)
    if args.primitive == "forward" and args.bc_forward_gait is not None:
        return str(args.bc_forward_gait)
    if args.primitive == "backward" and args.bc_backward_gait is not None:
        return str(args.bc_backward_gait)
    if args.primitive == "yaw" and args.bc_yaw_right_table is not None:
        return f"{args.bc_yaw_left_table or DEFAULT_YAW_LEFT_TABLE},{args.bc_yaw_right_table}"
    return "pushed_current_policy_20260701"


def compute_gae(
    agent: ActorCritic,
    obs_np: np.ndarray,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        next_value = agent.get_value(torch.as_tensor(obs_np, dtype=torch.float32))
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(rewards.shape[1], dtype=torch.float32)
    for step in reversed(range(args.rollout_steps)):
        if step == args.rollout_steps - 1:
            next_nonterminal = 1.0 - dones[step]
            next_values = next_value
        else:
            next_nonterminal = 1.0 - dones[step + 1]
            next_values = values[step + 1]
        delta = rewards[step] + args.gamma * next_values * next_nonterminal - values[step]
        last_gae = delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae
        advantages[step] = last_gae
    return advantages, advantages + values


def normalize_advantages(values: torch.Tensor) -> torch.Tensor:
    std = values.std(unbiased=False)
    if values.numel() < 2 or float(std.detach()) < 1.0e-8:
        return torch.zeros_like(values)
    return (values - values.mean()) / (std + 1e-8)


def evaluate_policy(agent: ActorCritic, args: argparse.Namespace) -> EvalStats:
    results: list[EvalResult] = []
    agent.eval()
    with torch.no_grad():
        for command_name, yaw_command in eval_commands(args):
            for episode in range(args.eval_episodes):
                env = make_env(args, randomize_yaw_command=False)
                try:
                    result = rollout_eval(
                        env,
                        agent,
                        command_name,
                        yaw_command,
                        seed=args.seed + 50_000 + 1000 * episode,
                    )
                    results.append(result)
                finally:
                    env.close()
    agent.train()
    return make_eval_stats(results)


def rollout_eval(
    env: BramV2PrimitiveEnv,
    agent: ActorCritic,
    command_name: str,
    yaw_command: float,
    *,
    seed: int,
) -> EvalResult:
    options = {"randomize": False}
    if env.primitive == "yaw":
        options["yaw_cmd"] = yaw_command
    obs, _ = env.reset(seed=seed, options=options)
    total_reward = 0.0
    previous_action = np.zeros(env.action_space.shape[0], dtype=np.float32)
    previous_delta = np.zeros(env.action_space.shape[0], dtype=np.float32)
    action_delta_squares: list[float] = []
    action_accel_squares: list[float] = []
    action_delta_abs: list[float] = []
    action_accel_abs: list[float] = []
    abs_actions: list[float] = []
    planar_drifts: list[float] = []
    planar_speeds: list[float] = []
    abs_yaw_errors: list[float] = []
    abs_roll_pitch_rates: list[float] = []
    support_deficits: list[float] = []
    contact_foot_speeds: list[float] = []
    height_warning_deficits: list[float] = []
    height_deficits: list[float] = []
    max_tilt = 0.0
    min_height = float("inf")
    final_info: dict[str, float] = {}
    terminated = False
    truncated = False
    length = 0
    for step in range(env.max_steps):
        obs_tensor = torch.as_tensor(obs[None, :], dtype=torch.float32)
        action = agent.deterministic_action(obs_tensor).cpu().numpy()[0].astype(np.float32)
        delta = action - previous_action
        accel = delta - previous_delta
        previous_action = action
        previous_delta = delta
        action_delta_squares.append(float(np.mean(np.square(delta))))
        action_accel_squares.append(float(np.mean(np.square(accel))))
        action_delta_abs.append(float(np.mean(np.abs(delta))))
        action_accel_abs.append(float(np.mean(np.abs(accel))))
        abs_actions.append(float(np.mean(np.abs(action))))
        obs, reward, terminated, truncated, final_info = env.step(action)
        total_reward += float(reward)
        current_x = float(final_info.get("x_distance", 0.0))
        current_y = float(final_info.get("y_distance", 0.0))
        planar_drifts.append(float(np.hypot(current_x, current_y)))
        planar_speeds.append(abs(float(final_info.get("planar_speed", 0.0))))
        abs_yaw_errors.append(abs(float(final_info.get("yaw_error", 0.0))))
        abs_roll_pitch_rates.append(abs(float(final_info.get("roll_pitch_rate", 0.0))))
        support_deficits.append(float(final_info.get("support_deficit", 0.0)))
        contact_foot_speeds.append(float(final_info.get("mean_contact_foot_speed", 0.0)))
        height_warning_deficits.append(
            float(final_info.get("body_height_warning_deficit", 0.0))
        )
        height_deficits.append(float(final_info.get("body_height_deficit", 0.0)))
        max_tilt = max(max_tilt, float(final_info.get("level_tilt_rad", 0.0)))
        min_height = min(min_height, float(final_info.get("height", float("inf"))))
        length = step + 1
        if terminated or truncated:
            break

    x_distance = float(final_info.get("x_distance", 0.0))
    y_distance = float(final_info.get("y_distance", 0.0))
    tilt_excess = max(0.0, max_tilt - TILT_FREE_RAD)
    action_delta_rms = float(np.sqrt(np.mean(action_delta_squares)))
    action_accel_rms = float(np.sqrt(np.mean(action_accel_squares)))
    yaw_distance = float(final_info.get("yaw_distance", 0.0))
    raw_line_distance = float(final_info.get("line_distance", 0.0))
    mean_planar_drift = mean_float(planar_drifts)
    max_planar_drift = max_float(planar_drifts)
    mean_abs_planar_speed = mean_float(planar_speeds)
    max_abs_planar_speed = max_float(planar_speeds)
    mean_abs_yaw_error = mean_float(abs_yaw_errors)
    mean_abs_roll_pitch_rate = mean_float(abs_roll_pitch_rates)
    mean_support_deficit = mean_float(support_deficits)
    mean_contact_foot_speed = mean_float(contact_foot_speeds)
    mean_height_warning_deficit = mean_float(height_warning_deficits)
    max_height_warning_deficit = max_float(height_warning_deficits)
    mean_height_deficit = mean_float(height_deficits)
    max_height_deficit = max_float(height_deficits)
    mean_action_delta_abs = mean_float(action_delta_abs)
    mean_action_accel_abs = mean_float(action_accel_abs)
    mean_abs_action = mean_float(abs_actions)
    if env.primitive == "yaw":
        line_distance = raw_line_distance
        primary_distance = yaw_distance
        planar_drift = float(np.hypot(x_distance, y_distance))
        target_primary_distance = yaw_target_distance(
            yaw_command,
            length,
            env.dt,
        )
        score = yaw_in_place_score(
            progress=primary_distance,
            planar_drift=planar_drift,
            mean_planar_drift=mean_planar_drift,
            max_planar_drift=max_planar_drift,
            mean_abs_planar_speed=mean_abs_planar_speed,
            max_abs_planar_speed=max_abs_planar_speed,
            mean_abs_yaw_error=mean_abs_yaw_error,
            mean_abs_roll_pitch_rate=mean_abs_roll_pitch_rate,
            mean_support_deficit=mean_support_deficit,
            mean_contact_foot_speed=mean_contact_foot_speed,
            mean_height_warning_deficit=mean_height_warning_deficit,
            max_height_warning_deficit=max_height_warning_deficit,
            mean_height_deficit=mean_height_deficit,
            max_height_deficit=max_height_deficit,
            mean_action_delta_abs=mean_action_delta_abs,
            mean_action_accel_abs=mean_action_accel_abs,
            mean_abs_action=mean_abs_action,
            target_progress=target_primary_distance,
            terminated=terminated,
            length=length,
            max_steps=env.max_steps,
        )
    else:
        net_distance = float(
            final_info.get(
                "v2_translation_net_distance",
                np.hypot(x_distance, y_distance),
            )
        )
        primary_distance = float(
            final_info.get("v2_translation_primary_distance", net_distance)
        )
        path_length = float(
            final_info.get("v2_translation_path_length", max(0.0, net_distance))
        )
        path_waste = float(
            final_info.get(
                "v2_translation_path_waste",
                max(0.0, path_length - net_distance),
            )
        )
        straightness = float(final_info.get("v2_translation_straightness", 1.0))
        direction_error = float(
            final_info.get("v2_translation_direction_error_rad", 0.0)
        )
        yaw_drift = abs(float(final_info.get("heading", 0.0)))
        direction_excess = max(0.0, direction_error - TRANSLATION_MAX_CREDIT_CONE_RAD)
        movement_floor = 0.10
        target_primary_distance = movement_floor
        no_motion_penalty = 0.35 if primary_distance < movement_floor else 0.0
        score = (
            primary_distance
            - 0.60 * path_waste
            - 0.35 * max(0.0, 0.70 - straightness)
            - 0.25 * direction_excess
            - 0.20 * yaw_drift
            - 0.10 * tilt_excess
            - 0.05 * action_delta_rms
            - no_motion_penalty
            - (0.45 if terminated else 0.0)
        )
        line_distance = primary_distance
        planar_drift = path_waste
    return EvalResult(
        command=command_name,
        yaw_command=float(yaw_command),
        reward=float(total_reward),
        score=float(score),
        line_distance=line_distance,
        yaw_distance=yaw_distance,
        planar_drift=planar_drift,
        x_distance=x_distance,
        y_distance=y_distance,
        max_tilt_rad=max_tilt,
        min_height_m=min_height,
        action_delta_rms=action_delta_rms,
        action_accel_rms=action_accel_rms,
        mean_action_delta_abs=mean_action_delta_abs,
        mean_action_accel_abs=mean_action_accel_abs,
        mean_abs_action=mean_abs_action,
        mean_planar_drift=mean_planar_drift,
        max_planar_drift=max_planar_drift,
        mean_abs_planar_speed=mean_abs_planar_speed,
        max_abs_planar_speed=max_abs_planar_speed,
        mean_abs_yaw_error=mean_abs_yaw_error,
        mean_abs_roll_pitch_rate=mean_abs_roll_pitch_rate,
        mean_support_deficit=mean_support_deficit,
        mean_contact_foot_speed=mean_contact_foot_speed,
        mean_height_warning_deficit=mean_height_warning_deficit,
        max_height_warning_deficit=max_height_warning_deficit,
        mean_height_deficit=mean_height_deficit,
        max_height_deficit=max_height_deficit,
        target_primary_distance=target_primary_distance,
        length=length,
        terminated=bool(terminated),
    )


def make_eval_stats(results: list[EvalResult]) -> EvalStats:
    command_deficits = [command_distance_deficit(result) for result in results]
    yaw_results = [result for result in results if result.command.startswith("yaw")]
    terminated_count = sum(1 for result in results if result.terminated)
    wrong_sign_count = sum(
        1
        for result in yaw_results
        if abs(result.yaw_command) > 0.05 and result.yaw_distance < 0.0
    )
    worst_score = min((result.score for result in results), default=0.0)
    worst_score_penalty = max(0.0, -worst_score)
    yaw_balance_penalty = (
        1.50 * worst_score_penalty
        + 2.00 * terminated_count
        + 1.50 * wrong_sign_count
        if yaw_results
        else 0.0
    )
    return EvalStats(
        reward=float(np.mean([result.reward for result in results])),
        score=float(
            np.mean([result.score for result in results])
            - 1.50 * np.mean(command_deficits)
            - 0.50 * max(command_deficits, default=0.0)
            - yaw_balance_penalty
        ),
        primary_distance=float(np.mean([primary_distance_result(result) for result in results])),
        planar_drift=float(np.mean([result.planar_drift for result in results])),
        length=float(np.mean([result.length for result in results])),
        per_command={
            result.command: {
                key: value
                for key, value in asdict(result).items()
                if key != "command"
            }
            for result in average_by_command(results)
        },
    )


def yaw_target_distance(yaw_command: float, length: int, dt: float) -> float:
    return float(
        YAW_TARGET_RATE_PER_COMMAND
        * abs(float(yaw_command))
        * max(1, int(length))
        * float(dt)
    )


def yaw_in_place_score(
    *,
    progress: float,
    planar_drift: float,
    mean_planar_drift: float,
    max_planar_drift: float,
    mean_abs_planar_speed: float,
    max_abs_planar_speed: float,
    mean_abs_yaw_error: float,
    mean_abs_roll_pitch_rate: float,
    mean_support_deficit: float,
    mean_contact_foot_speed: float,
    mean_height_warning_deficit: float,
    max_height_warning_deficit: float,
    mean_height_deficit: float,
    max_height_deficit: float,
    mean_action_delta_abs: float,
    mean_action_accel_abs: float,
    mean_abs_action: float,
    target_progress: float,
    terminated: bool,
    length: int,
    max_steps: int,
) -> float:
    """Compact version of gait_discovery's strict yaw-in-place score.

    It rewards signed yaw up to a reachable target while punishing translation,
    drift during the rollout, body thrash, support loss, and rough commands.
    """

    target_progress = max(1.0e-6, float(target_progress))
    useful_progress = max(0.0, float(progress))
    rewarded_progress = min(useful_progress, target_progress)
    target_error = abs(float(progress) - target_progress)
    wrong_way_penalty = 6.00 * max(0.0, -float(progress))
    underspin_penalty = 3.00 * max(0.0, 0.65 * target_progress - useful_progress)
    no_turn_penalty = 5.00 * max(0.0, 0.25 * target_progress - useful_progress)
    overspin_penalty = 2.75 * max(0.0, useful_progress - 1.12 * target_progress)
    score = (
        3.60 * rewarded_progress
        - 1.85 * target_error
        - 4.80 * planar_drift
        - 7.50 * mean_planar_drift
        - 9.50 * max_planar_drift
        - 1.50 * mean_abs_planar_speed
        - 0.75 * max_abs_planar_speed
        - 0.25 * mean_abs_yaw_error
        - 0.08 * mean_abs_roll_pitch_rate
        - 1.20 * mean_support_deficit
        - 2.00 * mean_contact_foot_speed
        - 24.0 * mean_height_warning_deficit
        - 36.0 * max_height_warning_deficit
        - 60.0 * mean_height_deficit
        - 90.0 * max_height_deficit
        - 0.75 * mean_action_delta_abs
        - 1.20 * mean_action_accel_abs
        - 0.10 * mean_abs_action
        - wrong_way_penalty
        - underspin_penalty
        - no_turn_penalty
        - overspin_penalty
    )
    if terminated:
        remaining_frac = max(0.0, (max_steps - length) / max(1, max_steps))
        score -= 2.50 + 2.00 * remaining_frac
    return float(score)


def average_by_command(results: list[EvalResult]) -> list[EvalResult]:
    averaged = []
    for name, _ in sorted({(result.command, result.yaw_command) for result in results}):
        group = [result for result in results if result.command == name]
        if not group:
            continue
        values = {}
        for key in asdict(group[0]):
            series = [getattr(result, key) for result in group]
            if key == "command":
                values[key] = name
            elif key == "terminated":
                values[key] = any(bool(value) for value in series)
            elif key == "length":
                values[key] = int(round(float(np.mean(series))))
            else:
                values[key] = float(np.mean(series))
        averaged.append(EvalResult(**values))
    return averaged


def save_checkpoint(
    path: Path,
    agent: ActorCritic,
    args: argparse.Namespace,
    stats: EvalStats,
) -> None:
    torch.save(
        {
            "model_state_dict": agent.state_dict(),
            "args": vars(args),
            "env_command_mode": YAW_ENV_COMMAND_MODE,
            "obs_dim": agent.obs_dim,
            "action_dim": agent.action_dim,
            "eval_reward": stats.reward,
            "eval_score": stats.score,
            "eval_primary_distance": stats.primary_distance,
            "eval_planar_drift": stats.planar_drift,
            "eval_length": stats.length,
            "eval_per_command": stats.per_command,
        },
        path,
    )


def metrics_fieldnames() -> list[str]:
    fields = [
        "update",
        "global_step",
        "sps",
        "recent_return",
        "recent_primary_distance",
        "recent_length",
        "eval_reward",
        "eval_score",
        "eval_primary_distance",
        "eval_planar_drift",
        "eval_length",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
    ]
    for command, _ in ALL_METRIC_COMMANDS:
        for suffix in ("score", "distance", "planar_drift", "length", "terminated"):
            fields.append(f"{command}_{suffix}")
    return fields


def write_single_metrics_row(path: Path, stats: EvalStats) -> None:
    row = {
        "update": 0,
        "global_step": 0,
        "sps": 0,
        "recent_return": float("nan"),
        "recent_primary_distance": float("nan"),
        "recent_length": float("nan"),
        "eval_reward": stats.reward,
        "eval_score": stats.score,
        "eval_primary_distance": stats.primary_distance,
        "eval_planar_drift": stats.planar_drift,
        "eval_length": stats.length,
        "policy_loss": float("nan"),
        "value_loss": float("nan"),
        "entropy": float("nan"),
        "approx_kl": float("nan"),
    }
    row.update(per_command_row(stats))
    with path.open("w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=metrics_fieldnames())
        writer.writeheader()
        writer.writerow(row)


def per_command_row(stats: EvalStats) -> dict[str, float | int]:
    row: dict[str, float | int] = {}
    for command, _ in ALL_METRIC_COMMANDS:
        values = stats.per_command.get(command, {})
        row[f"{command}_score"] = float(values.get("score", float("nan")))
        distance = primary_distance_values(command, values)
        row[f"{command}_distance"] = distance
        row[f"{command}_planar_drift"] = float(values.get("planar_drift", float("nan")))
        row[f"{command}_length"] = int(values.get("length", 0) or 0)
        row[f"{command}_terminated"] = int(bool(values.get("terminated", False)))
    return row


TRANSLATION_EVAL_COMMANDS: tuple[tuple[str, float], ...] = (
    ("forward", 0.0),
    ("backward", 0.0),
)
ALL_METRIC_COMMANDS = EVAL_COMMANDS + TRANSLATION_EVAL_COMMANDS


def eval_commands(args: argparse.Namespace) -> tuple[tuple[str, float], ...]:
    if args.primitive == "yaw":
        return EVAL_COMMANDS
    return ((args.primitive, 0.0),)


def primary_distance_info(info: dict, args: argparse.Namespace) -> float:
    if args.primitive == "yaw":
        return float(info.get("yaw_distance", 0.0))
    return float(
        info.get(
            "v2_translation_primary_distance",
            info.get("line_distance", 0.0),
        )
    )


def primary_distance_result(result: EvalResult) -> float:
    if result.command.startswith("yaw"):
        return float(result.yaw_distance)
    return float(result.line_distance)


def command_distance_deficit(result: EvalResult) -> float:
    distance = primary_distance_result(result)
    if result.command.startswith("yaw"):
        floor = 0.55 * max(
            0.0,
            float(result.target_primary_distance),
        )
    else:
        floor = 0.10
    return max(0.0, floor - distance)


def primary_distance_values(
    command: str,
    values: dict[str, float | bool | int],
) -> float:
    if command.startswith("yaw"):
        return float(values.get("yaw_distance", float("nan")))
    line_distance = values.get("line_distance")
    if line_distance is not None:
        return float(line_distance)
    return float(values.get("yaw_distance", float("nan")))


def mean_or_nan(values: deque[float] | deque[int]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def mean_float(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def max_float(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.max(values))


if __name__ == "__main__":
    main()
