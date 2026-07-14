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
from yaw_residual_env import (
    DEFAULT_LEFT_TABLE,
    DEFAULT_RIGHT_TABLE,
    YAW_RESIDUAL_COMMAND_MODE,
    BramV2YawResidualEnv,
    residual_env_metadata,
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
    yaw_distance: float
    target_yaw_distance: float
    x_distance: float
    y_distance: float
    planar_drift: float
    mean_planar_drift: float
    max_planar_drift: float
    rms_planar_drift: float
    max_tilt_rad: float
    min_height_m: float
    residual_delta_abs: float
    residual_accel_abs: float
    final_delta_abs: float
    final_accel_abs: float
    mean_abs_residual: float
    mean_abs_final_action: float
    length: int
    terminated: bool
    pass_gate: bool


@dataclass(frozen=True)
class EvalStats:
    reward: float
    score: float
    yaw_distance: float
    planar_drift: float
    mean_planar_drift: float
    max_planar_drift: float
    pass_count: int
    length: float
    per_command: dict[str, dict[str, float | bool | int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Yaw-only residual PPO around fixed movement_v2 yaw tables."
    )
    parser.add_argument("--left-table", type=Path, default=DEFAULT_LEFT_TABLE)
    parser.add_argument("--right-table", type=Path, default=DEFAULT_RIGHT_TABLE)
    parser.add_argument(
        "--left-params",
        type=Path,
        default=None,
        help="Use searched gait params directly as the left yaw base instead of a table.",
    )
    parser.add_argument(
        "--right-params",
        type=Path,
        default=None,
        help="Use searched gait params directly as the right yaw base instead of a table.",
    )
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.18)
    parser.add_argument("--entropy-coef", type=float, default=0.004)
    parser.add_argument("--value-coef", type=float, default=0.50)
    parser.add_argument("--max-grad-norm", type=float, default=0.50)
    parser.add_argument("--target-kl", type=float, default=0.05)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--log-std-init", type=float, default=-2.0)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--episode-seconds", type=float, default=8.0)
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=10,
        help="MuJoCo sim steps per residual action. 10 is 50 Hz with the current XML.",
    )
    parser.add_argument("--yaw-min-magnitude", type=float, default=0.35)
    parser.add_argument(
        "--train-command",
        choices=("all", "yaw_neg1", "yaw_neg05", "yaw_pos05", "yaw_pos1"),
        default="all",
        help="Narrow residual PPO to one yaw command before training the full yaw set.",
    )
    parser.add_argument(
        "--eval-suite",
        choices=("train", "all"),
        default="train",
        help="Evaluate only the training command or the full yaw command suite.",
    )
    parser.add_argument("--residual-limit", type=float, default=0.18)
    parser.add_argument("--target-yaw-rate", type=float, default=0.36)
    parser.add_argument("--final-drift-limit-m", type=float, default=0.040)
    parser.add_argument("--mean-drift-limit-m", type=float, default=0.025)
    parser.add_argument("--max-drift-limit-m", type=float, default=0.040)
    parser.add_argument("--start-final-drift-limit-m", type=float, default=None)
    parser.add_argument("--start-mean-drift-limit-m", type=float, default=None)
    parser.add_argument("--start-max-drift-limit-m", type=float, default=None)
    parser.add_argument(
        "--drift-curriculum-steps",
        type=int,
        default=0,
        help="Linearly tighten training drift limits from start-* values to final limits.",
    )
    parser.add_argument("--slew-limit", type=float, default=0.25)
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--domain-randomization-strength", type=float, default=0.15)
    parser.add_argument("--randomize-reset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--zero-actor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("software/movement_v2/runs/yaw_residual"),
    )
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    run_dir = args.output_dir / time.strftime("yaw_residual_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    checkpoint_path = run_dir / "residual_policy.pt"
    best_checkpoint_path = run_dir / "residual_policy_best.pt"

    envs = [make_env(args, randomize_yaw_command=True) for _ in range(args.num_envs)]
    obs_np = np.stack(
        [env.reset(seed=args.seed + index)[0] for index, env in enumerate(envs)]
    ).astype(np.float32)
    obs_dim = obs_np.shape[1]
    action_dim = envs[0].action_space.shape[0]
    agent = ActorCritic(obs_dim, action_dim, args.hidden_size, args.log_std_init)
    if args.zero_actor and args.init_checkpoint is None:
        zero_actor_output(agent)
    if args.init_checkpoint is not None:
        load_checkpoint(agent, args.init_checkpoint, args)

    initial_stats = evaluate_policy(agent, args)
    print_eval("initial_eval", initial_stats)
    if args.eval_only:
        write_metrics_row(metrics_path, 0, 0, 0, initial_stats, {}, {})
        save_checkpoint(checkpoint_path, agent, args, initial_stats, envs[0])
        save_checkpoint(best_checkpoint_path, agent, args, initial_stats, envs[0])
        close_envs(envs)
        return

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
    recent_yaw: deque[float] = deque(maxlen=50)
    recent_lengths: deque[int] = deque(maxlen=50)
    episode_returns = np.zeros(args.num_envs, dtype=np.float32)
    episode_yaw = np.zeros(args.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int32)

    print(
        f"training mode={YAW_RESIDUAL_COMMAND_MODE} total_steps={actual_total_steps} "
        f"num_envs={args.num_envs} rollout_steps={args.rollout_steps} "
        f"obs_dim={obs_dim} action_dim={action_dim} residual_limit={args.residual_limit:.3f} "
        f"train_command={args.train_command} eval_suite={args.eval_suite} "
        f"drift_limits=({args.final_drift_limit_m:.3f},"
        f"{args.mean_drift_limit_m:.3f},{args.max_drift_limit_m:.3f}) "
        f"left_table={args.left_table} right_table={args.right_table} "
        f"left_params={args.left_params} right_params={args.right_params}"
    )

    best_score = initial_stats.score
    save_checkpoint(best_checkpoint_path, agent, args, initial_stats, envs[0])
    global_step = 0
    start_time = time.perf_counter()
    with metrics_path.open("w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=metrics_fieldnames())
        writer.writeheader()
        write_metrics_row(
            metrics_path,
            0,
            0,
            0,
            initial_stats,
            {},
            {},
            writer=writer,
            log_file=log_file,
        )

        for update in range(1, num_updates + 1):
            set_training_drift_limits(envs, args, global_step)
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
                    episode_yaw[env_index] = float(info.get("yaw_distance", 0.0))
                    episode_lengths[env_index] += 1
                    if done:
                        recent_returns.append(float(episode_returns[env_index]))
                        recent_yaw.append(float(episode_yaw[env_index]))
                        recent_lengths.append(int(episode_lengths[env_index]))
                        episode_returns[env_index] = 0.0
                        episode_yaw[env_index] = 0.0
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
                    loss = (
                        policy_loss
                        - args.entropy_coef * entropy_loss
                        + args.value_coef * value_loss
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
            eval_stats = empty_eval_stats()
            if should_eval:
                eval_stats = evaluate_policy(agent, args)
                if eval_stats.score > best_score:
                    best_score = eval_stats.score
                    save_checkpoint(best_checkpoint_path, agent, args, eval_stats, envs[0])

            elapsed = time.perf_counter() - start_time
            train_stats = {
                "sps": int(global_step / max(elapsed, 1.0e-9)),
                "recent_return": mean_or_nan(recent_returns),
                "recent_yaw": mean_or_nan(recent_yaw),
                "recent_length": mean_or_nan(recent_lengths),
            }
            loss_stats = {
                "policy_loss": mean_or_nan(policy_losses),
                "value_loss": mean_or_nan(value_losses),
                "entropy": mean_or_nan(entropies),
                "approx_kl": mean_or_nan(approx_kls),
            }
            write_metrics_row(
                metrics_path,
                update,
                global_step,
                train_stats["sps"],
                eval_stats,
                train_stats,
                loss_stats,
                writer=writer,
                log_file=log_file,
            )
            if update == 1 or update == num_updates or update % 10 == 0 or should_eval:
                eval_score = f"{eval_stats.score:.3f}" if should_eval else "skip"
                print(
                    f"update={update:04d}/{num_updates} step={global_step} "
                    f"recent_return={train_stats['recent_return']:.3f} "
                    f"recent_yaw={train_stats['recent_yaw']:.3f} "
                    f"eval_score={eval_score} "
                    f"passes={eval_stats.pass_count if should_eval else 'skip'} "
                    f"entropy={loss_stats['entropy']:.3f}"
                )

    final_stats = evaluate_policy(agent, args)
    save_checkpoint(checkpoint_path, agent, args, final_stats, envs[0])
    print_eval("final_eval", final_stats)
    print(f"saved_policy={checkpoint_path}")
    print(f"saved_best_policy={best_checkpoint_path}")
    print(f"metrics={metrics_path}")
    close_envs(envs)


def make_env(args: argparse.Namespace, *, randomize_yaw_command: bool) -> BramV2YawResidualEnv:
    train_yaw = train_yaw_command(args)
    use_random_yaw = randomize_yaw_command and args.train_command == "all"
    return BramV2YawResidualEnv(
        left_table=args.left_table,
        right_table=args.right_table,
        left_params=args.left_params,
        right_params=args.right_params,
        frame_skip=args.frame_skip,
        episode_seconds=args.episode_seconds,
        randomize_reset=args.randomize_reset,
        domain_randomization=args.domain_randomization,
        domain_randomization_strength=args.domain_randomization_strength,
        randomize_yaw_command=use_random_yaw,
        yaw_command=train_yaw,
        yaw_min_magnitude=args.yaw_min_magnitude,
        residual_limit=args.residual_limit,
        target_yaw_rate=args.target_yaw_rate,
        final_drift_limit_m=args.final_drift_limit_m,
        mean_drift_limit_m=args.mean_drift_limit_m,
        max_drift_limit_m=args.max_drift_limit_m,
        slew_limit=args.slew_limit,
    )


def train_yaw_command(args: argparse.Namespace) -> float:
    commands = {
        "all": 1.0,
        "yaw_neg1": -1.0,
        "yaw_neg05": -0.5,
        "yaw_pos05": 0.5,
        "yaw_pos1": 1.0,
    }
    return commands[args.train_command]


def eval_commands(args: argparse.Namespace) -> tuple[tuple[str, float], ...]:
    if args.eval_suite == "all" or args.train_command == "all":
        return EVAL_COMMANDS
    return tuple(command for command in EVAL_COMMANDS if command[0] == args.train_command)


def set_training_drift_limits(
    envs: list[BramV2YawResidualEnv],
    args: argparse.Namespace,
    global_step: int,
) -> None:
    final_limit, mean_limit, max_limit = drift_limits_for_step(args, global_step)
    for env in envs:
        env.final_drift_limit_m = final_limit
        env.mean_drift_limit_m = mean_limit
        env.max_drift_limit_m = max_limit


def drift_limits_for_step(args: argparse.Namespace, global_step: int) -> tuple[float, float, float]:
    if args.drift_curriculum_steps <= 0:
        return (
            float(args.final_drift_limit_m),
            float(args.mean_drift_limit_m),
            float(args.max_drift_limit_m),
        )
    start_final = (
        float(args.start_final_drift_limit_m)
        if args.start_final_drift_limit_m is not None
        else float(args.final_drift_limit_m)
    )
    start_mean = (
        float(args.start_mean_drift_limit_m)
        if args.start_mean_drift_limit_m is not None
        else float(args.mean_drift_limit_m)
    )
    start_max = (
        float(args.start_max_drift_limit_m)
        if args.start_max_drift_limit_m is not None
        else float(args.max_drift_limit_m)
    )
    alpha = float(np.clip(global_step / max(1, args.drift_curriculum_steps), 0.0, 1.0))
    return (
        lerp(start_final, float(args.final_drift_limit_m), alpha),
        lerp(start_mean, float(args.mean_drift_limit_m), alpha),
        lerp(start_max, float(args.max_drift_limit_m), alpha),
    )


def lerp(start: float, end: float, alpha: float) -> float:
    return float((1.0 - alpha) * start + alpha * end)


def zero_actor_output(agent: ActorCritic) -> None:
    last = agent.actor[-1]
    if not isinstance(last, nn.Linear):
        raise TypeError("ActorCritic actor output layer is not Linear.")
    nn.init.constant_(last.weight, 0.0)
    nn.init.constant_(last.bias, 0.0)


def load_checkpoint(agent: ActorCritic, path: Path, args: argparse.Namespace) -> None:
    payload = torch.load(path, map_location="cpu")
    mode = payload.get("env_command_mode")
    if mode != YAW_RESIDUAL_COMMAND_MODE:
        raise ValueError(
            f"{path} has env_command_mode={mode!r}; expected {YAW_RESIDUAL_COMMAND_MODE!r}"
        )
    if int(payload.get("obs_dim", -1)) != agent.obs_dim or int(
        payload.get("action_dim", -1)
    ) != agent.action_dim:
        raise ValueError(f"{path} obs/action dims do not match current agent.")
    hidden_size = int(payload.get("args", {}).get("hidden_size", agent.hidden_size))
    if hidden_size != args.hidden_size:
        raise ValueError(f"{path} hidden_size={hidden_size}; expected {args.hidden_size}")
    agent.load_state_dict(payload["model_state_dict"])
    print(f"loaded_init_checkpoint={path}")


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
    return (values - values.mean()) / (std + 1.0e-8)


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
                        args,
                        seed=args.seed + 80_000 + 1000 * episode,
                    )
                    results.append(result)
                finally:
                    env.close()
    agent.train()
    return make_eval_stats(results)


def rollout_eval(
    env: BramV2YawResidualEnv,
    agent: ActorCritic,
    command_name: str,
    yaw_command: float,
    args: argparse.Namespace,
    *,
    seed: int,
) -> EvalResult:
    obs, _ = env.reset(seed=seed, options={"randomize": False, "yaw_cmd": yaw_command})
    total_reward = 0.0
    residual_delta_abs: list[float] = []
    residual_accel_abs: list[float] = []
    residual_abs: list[float] = []
    final_delta_abs: list[float] = []
    final_accel_abs: list[float] = []
    final_abs: list[float] = []
    drifts: list[float] = []
    max_tilt = 0.0
    min_height = float("inf")
    final_info: dict[str, float] = {}
    terminated = False
    truncated = False
    length = 0
    for step in range(env.max_steps):
        obs_tensor = torch.as_tensor(obs[None, :], dtype=torch.float32)
        residual = agent.deterministic_action(obs_tensor).cpu().numpy()[0].astype(np.float32)
        obs, reward, terminated, truncated, final_info = env.step(residual)
        total_reward += float(reward)
        x_distance = float(final_info.get("x_distance", 0.0))
        y_distance = float(final_info.get("y_distance", 0.0))
        drifts.append(float(np.hypot(x_distance, y_distance)))
        residual_delta_abs.append(float(final_info.get("v2_residual_delta_abs", 0.0)))
        residual_accel_abs.append(float(final_info.get("v2_residual_accel_abs", 0.0)))
        residual_abs.append(float(final_info.get("v2_residual_abs", 0.0)))
        final_delta_abs.append(float(final_info.get("v2_residual_final_delta_abs", 0.0)))
        final_accel_abs.append(float(final_info.get("v2_residual_final_accel_abs", 0.0)))
        final_action = final_info.get("v2_residual_final_action", [0.0, 0.0, 0.0])
        final_abs.append(float(np.mean(np.abs(np.asarray(final_action, dtype=np.float32)))))
        max_tilt = max(max_tilt, float(final_info.get("level_tilt_rad", 0.0)))
        min_height = min(min_height, float(final_info.get("height", float("inf"))))
        length = step + 1
        if terminated or truncated:
            break

    x_distance = float(final_info.get("x_distance", 0.0))
    y_distance = float(final_info.get("y_distance", 0.0))
    planar_drift = float(np.hypot(x_distance, y_distance))
    mean_drift = mean_float(drifts)
    max_drift = max_float(drifts)
    rms_drift = float(np.sqrt(np.mean(np.square(drifts)))) if drifts else 0.0
    yaw_distance = float(final_info.get("yaw_distance", 0.0))
    elapsed = length * env.dt
    target_yaw_distance = abs(float(yaw_command)) * args.target_yaw_rate * elapsed
    pass_gate = (
        planar_drift <= args.final_drift_limit_m
        and mean_drift <= args.mean_drift_limit_m
        and max_drift <= args.max_drift_limit_m
        and yaw_distance >= 0.65 * target_yaw_distance
        and not terminated
    )
    score = residual_eval_score(
        yaw_distance=yaw_distance,
        target_yaw_distance=target_yaw_distance,
        planar_drift=planar_drift,
        mean_planar_drift=mean_drift,
        max_planar_drift=max_drift,
        residual_delta_abs=mean_float(residual_delta_abs),
        residual_accel_abs=mean_float(residual_accel_abs),
        final_delta_abs=mean_float(final_delta_abs),
        final_accel_abs=mean_float(final_accel_abs),
        final_drift_limit_m=args.final_drift_limit_m,
        mean_drift_limit_m=args.mean_drift_limit_m,
        max_drift_limit_m=args.max_drift_limit_m,
        terminated=terminated,
        length=length,
        max_steps=env.max_steps,
    )
    return EvalResult(
        command=command_name,
        yaw_command=float(yaw_command),
        reward=float(total_reward),
        score=float(score),
        yaw_distance=yaw_distance,
        target_yaw_distance=target_yaw_distance,
        x_distance=x_distance,
        y_distance=y_distance,
        planar_drift=planar_drift,
        mean_planar_drift=mean_drift,
        max_planar_drift=max_drift,
        rms_planar_drift=rms_drift,
        max_tilt_rad=max_tilt,
        min_height_m=min_height,
        residual_delta_abs=mean_float(residual_delta_abs),
        residual_accel_abs=mean_float(residual_accel_abs),
        final_delta_abs=mean_float(final_delta_abs),
        final_accel_abs=mean_float(final_accel_abs),
        mean_abs_residual=mean_float(residual_abs),
        mean_abs_final_action=mean_float(final_abs),
        length=length,
        terminated=bool(terminated),
        pass_gate=pass_gate,
    )


def residual_eval_score(
    *,
    yaw_distance: float,
    target_yaw_distance: float,
    planar_drift: float,
    mean_planar_drift: float,
    max_planar_drift: float,
    residual_delta_abs: float,
    residual_accel_abs: float,
    final_delta_abs: float,
    final_accel_abs: float,
    final_drift_limit_m: float,
    mean_drift_limit_m: float,
    max_drift_limit_m: float,
    terminated: bool,
    length: int,
    max_steps: int,
) -> float:
    target = max(1.0e-6, target_yaw_distance)
    useful = max(0.0, yaw_distance)
    final_excess = max(0.0, planar_drift - final_drift_limit_m)
    mean_excess = max(0.0, mean_planar_drift - mean_drift_limit_m)
    max_excess = max(0.0, max_planar_drift - max_drift_limit_m)
    wrong_way = max(0.0, -yaw_distance)
    underspin = max(0.0, 0.65 * target - useful)
    overspin = max(0.0, useful - 1.35 * target)
    score = (
        5.0 * min(useful, target)
        - 1.5 * abs(yaw_distance - target)
        - 12.0 * planar_drift
        - 18.0 * mean_planar_drift
        - 26.0 * max_planar_drift
        - 180.0 * final_excess
        - 260.0 * mean_excess
        - 340.0 * max_excess
        - 8.0 * wrong_way
        - 3.0 * underspin
        - 1.2 * overspin
        - 0.8 * residual_delta_abs
        - 1.3 * residual_accel_abs
        - 0.9 * final_delta_abs
        - 1.4 * final_accel_abs
    )
    if terminated:
        score -= 12.0 + 8.0 * max(0.0, (max_steps - length) / max(1, max_steps))
    return float(score)


def make_eval_stats(results: list[EvalResult]) -> EvalStats:
    if not results:
        return empty_eval_stats()
    pass_count = sum(1 for result in results if result.pass_gate)
    fail_penalty = 2.5 * (len(results) - pass_count)
    worst_score_penalty = max(0.0, -min(result.score for result in results))
    return EvalStats(
        reward=float(np.mean([result.reward for result in results])),
        score=float(np.mean([result.score for result in results]) - fail_penalty - worst_score_penalty),
        yaw_distance=float(np.mean([result.yaw_distance for result in results])),
        planar_drift=float(np.mean([result.planar_drift for result in results])),
        mean_planar_drift=float(np.mean([result.mean_planar_drift for result in results])),
        max_planar_drift=float(np.mean([result.max_planar_drift for result in results])),
        pass_count=int(pass_count),
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


def average_by_command(results: list[EvalResult]) -> list[EvalResult]:
    averaged = []
    for command in sorted({result.command for result in results}):
        group = [result for result in results if result.command == command]
        values = {}
        for key in asdict(group[0]):
            series = [getattr(result, key) for result in group]
            if key == "command":
                values[key] = command
            elif key in ("terminated", "pass_gate"):
                values[key] = all(bool(value) for value in series)
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
    env: BramV2YawResidualEnv,
) -> None:
    torch.save(
        {
            "model_state_dict": agent.state_dict(),
            "args": vars(args),
            "env_command_mode": YAW_RESIDUAL_COMMAND_MODE,
            "obs_dim": agent.obs_dim,
            "action_dim": agent.action_dim,
            "eval_reward": stats.reward,
            "eval_score": stats.score,
            "eval_yaw_distance": stats.yaw_distance,
            "eval_planar_drift": stats.planar_drift,
            "eval_mean_planar_drift": stats.mean_planar_drift,
            "eval_max_planar_drift": stats.max_planar_drift,
            "eval_pass_count": stats.pass_count,
            "eval_per_command": stats.per_command,
            "residual_env": residual_env_metadata(env),
        },
        path,
    )


def metrics_fieldnames() -> list[str]:
    fields = [
        "update",
        "global_step",
        "sps",
        "recent_return",
        "recent_yaw",
        "recent_length",
        "eval_reward",
        "eval_score",
        "eval_yaw_distance",
        "eval_planar_drift",
        "eval_mean_planar_drift",
        "eval_max_planar_drift",
        "eval_pass_count",
        "eval_length",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
    ]
    for command, _ in EVAL_COMMANDS:
        for suffix in (
            "score",
            "yaw_distance",
            "target_yaw_distance",
            "planar_drift",
            "mean_planar_drift",
            "max_planar_drift",
            "pass_gate",
            "length",
            "terminated",
        ):
            fields.append(f"{command}_{suffix}")
    return fields


def write_metrics_row(
    path: Path,
    update: int,
    global_step: int,
    sps: int,
    stats: EvalStats,
    train_stats: dict[str, float],
    loss_stats: dict[str, float],
    *,
    writer: csv.DictWriter | None = None,
    log_file=None,
) -> None:
    row: dict[str, float | int] = {
        "update": update,
        "global_step": global_step,
        "sps": sps,
        "recent_return": float(train_stats.get("recent_return", float("nan"))),
        "recent_yaw": float(train_stats.get("recent_yaw", float("nan"))),
        "recent_length": float(train_stats.get("recent_length", float("nan"))),
        "eval_reward": stats.reward,
        "eval_score": stats.score,
        "eval_yaw_distance": stats.yaw_distance,
        "eval_planar_drift": stats.planar_drift,
        "eval_mean_planar_drift": stats.mean_planar_drift,
        "eval_max_planar_drift": stats.max_planar_drift,
        "eval_pass_count": stats.pass_count,
        "eval_length": stats.length,
        "policy_loss": float(loss_stats.get("policy_loss", float("nan"))),
        "value_loss": float(loss_stats.get("value_loss", float("nan"))),
        "entropy": float(loss_stats.get("entropy", float("nan"))),
        "approx_kl": float(loss_stats.get("approx_kl", float("nan"))),
    }
    row.update(per_command_row(stats))
    if writer is None:
        with path.open("w", newline="") as log_file_obj:
            writer = csv.DictWriter(log_file_obj, fieldnames=metrics_fieldnames())
            writer.writeheader()
            writer.writerow(row)
        return
    writer.writerow(row)
    if log_file is not None:
        log_file.flush()


def per_command_row(stats: EvalStats) -> dict[str, float | int]:
    row: dict[str, float | int] = {}
    for command, _ in EVAL_COMMANDS:
        values = stats.per_command.get(command, {})
        row[f"{command}_score"] = float(values.get("score", float("nan")))
        row[f"{command}_yaw_distance"] = float(values.get("yaw_distance", float("nan")))
        row[f"{command}_target_yaw_distance"] = float(
            values.get("target_yaw_distance", float("nan"))
        )
        row[f"{command}_planar_drift"] = float(values.get("planar_drift", float("nan")))
        row[f"{command}_mean_planar_drift"] = float(
            values.get("mean_planar_drift", float("nan"))
        )
        row[f"{command}_max_planar_drift"] = float(
            values.get("max_planar_drift", float("nan"))
        )
        row[f"{command}_pass_gate"] = int(bool(values.get("pass_gate", False)))
        row[f"{command}_length"] = int(values.get("length", 0) or 0)
        row[f"{command}_terminated"] = int(bool(values.get("terminated", False)))
    return row


def empty_eval_stats() -> EvalStats:
    return EvalStats(
        reward=float("nan"),
        score=float("nan"),
        yaw_distance=float("nan"),
        planar_drift=float("nan"),
        mean_planar_drift=float("nan"),
        max_planar_drift=float("nan"),
        pass_count=0,
        length=float("nan"),
        per_command={},
    )


def print_eval(label: str, stats: EvalStats) -> None:
    print(
        f"{label} reward={stats.reward:.3f} score={stats.score:.3f} "
        f"yaw={stats.yaw_distance:.3f} drift={stats.planar_drift:.4f} "
        f"mean_drift={stats.mean_planar_drift:.4f} "
        f"max_drift={stats.max_planar_drift:.4f} "
        f"pass={stats.pass_count}/{len(stats.per_command)}"
    )
    for command, values in stats.per_command.items():
        print(
            f"  {command}: yaw={float(values['yaw_distance']):+.3f} "
            f"target={float(values['target_yaw_distance']):.3f} "
            f"drift={float(values['planar_drift']):.4f} "
            f"mean={float(values['mean_planar_drift']):.4f} "
            f"max={float(values['max_planar_drift']):.4f} "
            f"pass={int(bool(values['pass_gate']))} "
            f"term={int(bool(values['terminated']))}"
        )


def mean_float(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def max_float(values: list[float]) -> float:
    return float(np.max(values)) if values else 0.0


def mean_or_nan(values) -> float:
    return float(np.mean(values)) if len(values) > 0 else float("nan")


def close_envs(envs: list[BramV2YawResidualEnv]) -> None:
    for env in envs:
        env.close()


if __name__ == "__main__":
    main()
