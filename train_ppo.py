from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from bram_env import BramTripodEnv, ENV_COMMAND_MODE


@dataclass
class EvalStats:
    reward: float
    distance: float
    length: float


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int,
        log_std_init: float = -1.0,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, action_dim), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = torch.tanh(self.actor(obs))
        std = torch.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action.clamp(-1.0, 1.0), log_prob, entropy, self.get_value(obs)

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.actor(obs)).clamp(-1.0, 1.0)


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small PPO smoke trainer for BramTripodEnv.")
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--randomize-command", action="store_true")
    parser.add_argument("--forward-command", type=float, default=1.0)
    parser.add_argument("--yaw-rate-command", type=float, default=0.0)
    parser.add_argument("--log-std-init", type=float, default=-1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    run_dir = args.output_dir / time.strftime("ppo_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "metrics.csv"
    checkpoint_path = run_dir / "policy.pt"
    best_checkpoint_path = run_dir / "policy_best.pt"

    envs = [
        BramTripodEnv(
            domain_randomization=args.domain_randomization,
            randomize_command=args.randomize_command,
            command_forward=args.forward_command,
            command_yaw_rate=args.yaw_rate_command,
        )
        for _ in range(args.num_envs)
    ]
    obs_list = []
    for index, env in enumerate(envs):
        obs, _ = env.reset(seed=args.seed + index)
        obs_list.append(obs)
    obs_np = np.stack(obs_list).astype(np.float32)

    obs_dim = obs_np.shape[1]
    action_dim = envs[0].action_space.shape[0]
    agent = ActorCritic(obs_dim, action_dim, args.hidden_size, args.log_std_init)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    num_updates = max(1, args.total_steps // (args.num_envs * args.rollout_steps))
    actual_total_steps = num_updates * args.num_envs * args.rollout_steps

    obs_buf = torch.zeros((args.rollout_steps, args.num_envs, obs_dim), dtype=torch.float32)
    action_buf = torch.zeros((args.rollout_steps, args.num_envs, action_dim), dtype=torch.float32)
    logprob_buf = torch.zeros((args.rollout_steps, args.num_envs), dtype=torch.float32)
    reward_buf = torch.zeros((args.rollout_steps, args.num_envs), dtype=torch.float32)
    done_buf = torch.zeros((args.rollout_steps, args.num_envs), dtype=torch.float32)
    value_buf = torch.zeros((args.rollout_steps, args.num_envs), dtype=torch.float32)

    episode_returns = np.zeros(args.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int32)
    recent_returns: deque[float] = deque(maxlen=50)
    recent_distances: deque[float] = deque(maxlen=50)
    recent_lengths: deque[int] = deque(maxlen=50)

    random_stats = evaluate_random(args.eval_episodes, args.seed + 10_000, args)
    print(
        "random_baseline "
        f"reward={random_stats.reward:.3f} "
        f"distance={random_stats.distance:.4f} "
        f"length={random_stats.length:.1f}"
    )
    print(
        f"training total_steps={actual_total_steps} num_envs={args.num_envs} "
        f"rollout_steps={args.rollout_steps} obs_dim={obs_dim} action_dim={action_dim} "
        f"domain_randomization={args.domain_randomization} "
        f"randomize_command={args.randomize_command} "
        f"forward_command={args.forward_command:.2f} "
        f"yaw_rate_command={args.yaw_rate_command:.2f}"
    )

    best_eval_distance = -float("inf")
    global_step = 0
    start_time = time.perf_counter()

    with log_path.open("w", newline="") as log_file:
        writer = csv.DictWriter(
            log_file,
            fieldnames=[
                "update",
                "global_step",
                "sps",
                "recent_return",
                "recent_distance",
                "recent_length",
                "eval_reward",
                "eval_distance",
                "eval_length",
                "policy_loss",
                "value_loss",
                "entropy",
                "approx_kl",
            ],
        )
        writer.writeheader()

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
                    episode_returns[env_index] += reward
                    episode_lengths[env_index] += 1

                    if done:
                        recent_returns.append(float(episode_returns[env_index]))
                        recent_distances.append(distance_from_info(info))
                        recent_lengths.append(int(episode_lengths[env_index]))
                        episode_returns[env_index] = 0.0
                        episode_lengths[env_index] = 0
                        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))

                    next_obs.append(obs)
                    rewards.append(reward)
                    dones.append(done)

                obs_buf[step] = obs_tensor
                action_buf[step] = action
                logprob_buf[step] = logprob
                reward_buf[step] = torch.as_tensor(rewards, dtype=torch.float32)
                done_buf[step] = torch.as_tensor(dones, dtype=torch.float32)
                value_buf[step] = value
                obs_np = np.stack(next_obs).astype(np.float32)

            with torch.no_grad():
                next_obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32)
                next_value = agent.get_value(next_obs_tensor)
                advantages = torch.zeros_like(reward_buf)
                last_gae = torch.zeros(args.num_envs, dtype=torch.float32)
                for step in reversed(range(args.rollout_steps)):
                    if step == args.rollout_steps - 1:
                        next_nonterminal = 1.0 - done_buf[step]
                        next_values = next_value
                    else:
                        next_nonterminal = 1.0 - done_buf[step + 1]
                        next_values = value_buf[step + 1]
                    delta = (
                        reward_buf[step]
                        + args.gamma * next_values * next_nonterminal
                        - value_buf[step]
                    )
                    last_gae = (
                        delta
                        + args.gamma * args.gae_lambda * next_nonterminal * last_gae
                    )
                    advantages[step] = last_gae
                returns = advantages + value_buf

            batch_obs = obs_buf.reshape((-1, obs_dim))
            batch_actions = action_buf.reshape((-1, action_dim))
            batch_logprobs = logprob_buf.reshape(-1)
            batch_advantages = advantages.reshape(-1)
            batch_returns = returns.reshape(-1)
            batch_values = value_buf.reshape(-1)

            batch_advantages = (
                batch_advantages - batch_advantages.mean()
            ) / (batch_advantages.std() + 1e-8)

            batch_size = args.num_envs * args.rollout_steps
            minibatch_size = min(args.minibatch_size, batch_size)
            policy_losses = []
            value_losses = []
            entropies = []
            approx_kls = []
            for _ in range(args.update_epochs):
                indices = torch.randperm(batch_size)
                for start in range(0, batch_size, minibatch_size):
                    mb_idx = indices[start : start + minibatch_size]
                    _, new_logprob, entropy, new_value = agent.get_action_and_value(
                        batch_obs[mb_idx], batch_actions[mb_idx]
                    )
                    log_ratio = new_logprob - batch_logprobs[mb_idx]
                    ratio = log_ratio.exp()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - log_ratio).mean()

                    pg_loss_1 = -batch_advantages[mb_idx] * ratio
                    pg_loss_2 = -batch_advantages[mb_idx] * torch.clamp(
                        ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef
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

            should_eval = update == 1 or update == num_updates or update % args.eval_interval == 0
            eval_stats = EvalStats(float("nan"), float("nan"), float("nan"))
            if should_eval:
                eval_stats = evaluate_policy(
                    agent, args.eval_episodes, args.seed + 20_000 + update, args
                )
                if eval_stats.distance > best_eval_distance:
                    best_eval_distance = eval_stats.distance
                    save_checkpoint(best_checkpoint_path, agent, args, eval_stats)

            elapsed = time.perf_counter() - start_time
            sps = int(global_step / max(elapsed, 1e-9))
            row = {
                "update": update,
                "global_step": global_step,
                "sps": sps,
                "recent_return": mean_or_nan(recent_returns),
                "recent_distance": mean_or_nan(recent_distances),
                "recent_length": mean_or_nan(recent_lengths),
                "eval_reward": eval_stats.reward,
                "eval_distance": eval_stats.distance,
                "eval_length": eval_stats.length,
                "policy_loss": float(np.mean(policy_losses)),
                "value_loss": float(np.mean(value_losses)),
                "entropy": float(np.mean(entropies)),
                "approx_kl": float(np.mean(approx_kls)),
            }
            writer.writerow(row)
            log_file.flush()

            if update == 1 or update == num_updates or update % 10 == 0 or should_eval:
                eval_dist = f"{row['eval_distance']:.4f}" if should_eval else "skip"
                print(
                    f"update={update:04d}/{num_updates} "
                    f"step={global_step} sps={sps} "
                    f"recent_return={row['recent_return']:.3f} "
                    f"recent_dist={row['recent_distance']:.4f} "
                    f"eval_dist={eval_dist} "
                    f"entropy={row['entropy']:.3f}"
                )

    final_stats = evaluate_policy(agent, args.eval_episodes, args.seed + 30_000, args)
    save_checkpoint(checkpoint_path, agent, args, final_stats)
    print(
        "final_eval "
        f"reward={final_stats.reward:.3f} "
        f"distance={final_stats.distance:.4f} "
        f"length={final_stats.length:.1f}"
    )
    print(f"saved_policy={checkpoint_path}")
    print(f"saved_best_policy={best_checkpoint_path}")
    print(f"metrics={log_path}")


def evaluate_random(episodes: int, seed: int, args: argparse.Namespace) -> EvalStats:
    env = BramTripodEnv(
        randomize_reset=False,
        randomize_command=args.randomize_command,
        command_forward=args.forward_command,
        command_yaw_rate=args.yaw_rate_command,
    )
    rewards = []
    distances = []
    lengths = []
    rng = np.random.default_rng(seed)
    for episode in range(episodes):
        obs, _ = env.reset(
            seed=seed + episode,
            options=eval_command(episode, episodes, args),
        )
        total_reward = 0.0
        final_info = {"command_distance": 0.0}
        for length in range(env.max_steps):
            action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
            obs, reward, terminated, truncated, final_info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
        distances.append(distance_from_info(final_info))
        lengths.append(length + 1)
    return EvalStats(float(np.mean(rewards)), float(np.mean(distances)), float(np.mean(lengths)))


def evaluate_policy(
    agent: ActorCritic,
    episodes: int,
    seed: int,
    args: argparse.Namespace,
) -> EvalStats:
    env = BramTripodEnv(
        randomize_reset=False,
        randomize_command=args.randomize_command,
        command_forward=args.forward_command,
        command_yaw_rate=args.yaw_rate_command,
    )
    rewards = []
    distances = []
    lengths = []
    agent.eval()
    with torch.no_grad():
        for episode in range(episodes):
            obs, _ = env.reset(
                seed=seed + episode,
                options=eval_command(episode, episodes, args),
            )
            total_reward = 0.0
            final_info = {"command_distance": 0.0}
            for length in range(env.max_steps):
                obs_tensor = torch.as_tensor(obs[None, :], dtype=torch.float32)
                action = agent.deterministic_action(obs_tensor).cpu().numpy()[0]
                obs, reward, terminated, truncated, final_info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            rewards.append(total_reward)
            distances.append(distance_from_info(final_info))
            lengths.append(length + 1)
    agent.train()
    return EvalStats(float(np.mean(rewards)), float(np.mean(distances)), float(np.mean(lengths)))


def save_checkpoint(
    path: Path,
    agent: ActorCritic,
    args: argparse.Namespace,
    eval_stats: EvalStats,
) -> None:
    torch.save(
        {
            "model_state_dict": agent.state_dict(),
            "args": vars(args),
            "env_command_mode": ENV_COMMAND_MODE,
            "obs_dim": agent.obs_dim,
            "action_dim": agent.action_dim,
            "eval_reward": eval_stats.reward,
            "eval_distance": eval_stats.distance,
            "eval_length": eval_stats.length,
        },
        path,
    )


def mean_or_nan(values: deque[float] | deque[int]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def distance_from_info(info: dict) -> float:
    return float(info.get("command_distance", info.get("x_distance", 0.0)))


def eval_command(
    episode: int,
    episodes: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    if not args.randomize_command:
        return {
            "forward_command": float(np.clip(args.forward_command, -1.0, 1.0)),
            "yaw_rate_command": float(np.clip(args.yaw_rate_command, -1.0, 1.0)),
        }

    eval_commands = [
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (0.7, 0.7),
        (0.7, -0.7),
    ]
    forward, yaw_rate = eval_commands[episode % len(eval_commands)]
    return {"forward_command": forward, "yaw_rate_command": yaw_rate}


if __name__ == "__main__":
    main()
