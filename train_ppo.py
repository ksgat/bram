from __future__ import annotations

import argparse
import copy
import csv
import math
import time
from collections import deque
from dataclasses import dataclass, field
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
    score: float
    worst_reward: float
    worst_length: float
    primitive_pass_count: int = 0
    primitive_deficit: float = 0.0
    command_deficit: float = 0.0
    weak_commands: tuple[str, ...] = ()
    per_command: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class CommandSpec:
    name: str
    forward: float
    yaw_rate: float


@dataclass
class EpisodeResult:
    name: str
    reward: float
    command_distance: float
    line_distance: float
    yaw_distance: float
    length: int


@dataclass
class ExpertPolicies:
    forward: ActorCritic | None = None
    backward: ActorCritic | None = None
    yaw_left: ActorCritic | None = None
    yaw_right: ActorCritic | None = None
    arc_fl: ActorCritic | None = None
    arc_fr: ActorCritic | None = None
    arc_bl: ActorCritic | None = None
    arc_br: ActorCritic | None = None


@dataclass
class ExpertDataset:
    obs: torch.Tensor
    actions: torch.Tensor


EVAL_COMMANDS = (
    CommandSpec("idle", 0.0, 0.0),
    CommandSpec("fwd1", 1.0, 0.0),
    CommandSpec("back1", -1.0, 0.0),
    CommandSpec("fwd05", 0.5, 0.0),
    CommandSpec("back05", -0.5, 0.0),
    CommandSpec("yaw_l1", 0.0, 1.0),
    CommandSpec("yaw_r1", 0.0, -1.0),
    CommandSpec("yaw_l05", 0.0, 0.5),
    CommandSpec("yaw_r05", 0.0, -0.5),
    CommandSpec("arc_fl", 0.7, 0.7),
    CommandSpec("arc_fr", 0.7, -0.7),
    CommandSpec("arc_bl", -0.7, 0.7),
    CommandSpec("arc_br", -0.7, -0.7),
)
CURRICULUM_STAGE_LEVELS = (0.0, 0.20, 0.40, 0.60, 0.82, 1.0)
OBS_FORWARD_COMMAND_INDEX = 9
OBS_YAW_COMMAND_INDEX = 10


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


def load_initial_checkpoint(
    path: Path,
    agent: ActorCritic,
    *,
    allow_mode_mismatch: bool = False,
    label: str = "init_checkpoint",
) -> None:
    payload = torch.load(path, map_location="cpu")
    checkpoint_mode = payload.get("env_command_mode")
    if checkpoint_mode != ENV_COMMAND_MODE and not allow_mode_mismatch:
        raise ValueError(
            f"{path} was trained with env_command_mode={checkpoint_mode!r}, "
            f"but current mode is {ENV_COMMAND_MODE!r}"
        )
    if checkpoint_mode != ENV_COMMAND_MODE and allow_mode_mismatch:
        print(
            "checkpoint_mode_mismatch_allowed "
            f"label={label} "
            f"path={path} "
            f"checkpoint_mode={checkpoint_mode!r} "
            f"current_mode={ENV_COMMAND_MODE!r}"
        )

    checkpoint_obs_dim = int(payload.get("obs_dim", -1))
    checkpoint_action_dim = int(payload.get("action_dim", -1))
    if (checkpoint_obs_dim, checkpoint_action_dim) != (
        agent.obs_dim,
        agent.action_dim,
    ):
        raise ValueError(
            f"{path} has obs/action dims "
            f"{checkpoint_obs_dim}/{checkpoint_action_dim}, expected "
            f"{agent.obs_dim}/{agent.action_dim}"
        )

    agent.load_state_dict(payload["model_state_dict"])
    print(
        f"loaded_{label} "
        f"path={path} "
        f"eval_reward={payload.get('eval_reward', float('nan')):.3f} "
        f"eval_distance={payload.get('eval_distance', float('nan')):.4f}"
    )


def load_expert_checkpoint(
    path: Path,
    obs_dim: int,
    action_dim: int,
    hidden_size: int,
    *,
    allow_mode_mismatch: bool,
    label: str,
) -> ActorCritic:
    expert = ActorCritic(obs_dim, action_dim, hidden_size)
    load_initial_checkpoint(
        path,
        expert,
        allow_mode_mismatch=allow_mode_mismatch,
        label=label,
    )
    expert.eval()
    for parameter in expert.parameters():
        parameter.requires_grad = False
    return expert


def load_expert_policies(
    args: argparse.Namespace,
    obs_dim: int,
    action_dim: int,
) -> ExpertPolicies:
    experts = ExpertPolicies()
    if args.forward_expert is not None:
        experts.forward = load_expert_checkpoint(
            args.forward_expert,
            obs_dim,
            action_dim,
            args.hidden_size,
            allow_mode_mismatch=args.allow_expert_mode_mismatch,
            label="forward_expert",
        )
    if args.backward_expert is not None:
        experts.backward = load_expert_checkpoint(
            args.backward_expert,
            obs_dim,
            action_dim,
            args.hidden_size,
            allow_mode_mismatch=args.allow_expert_mode_mismatch,
            label="backward_expert",
        )
    if args.yaw_left_expert is not None:
        experts.yaw_left = load_expert_checkpoint(
            args.yaw_left_expert,
            obs_dim,
            action_dim,
            args.hidden_size,
            allow_mode_mismatch=args.allow_expert_mode_mismatch,
            label="yaw_left_expert",
        )
    if args.yaw_right_expert is not None:
        experts.yaw_right = load_expert_checkpoint(
            args.yaw_right_expert,
            obs_dim,
            action_dim,
            args.hidden_size,
            allow_mode_mismatch=args.allow_expert_mode_mismatch,
            label="yaw_right_expert",
        )
    if args.arc_fl_expert is not None:
        experts.arc_fl = load_expert_checkpoint(
            args.arc_fl_expert,
            obs_dim,
            action_dim,
            args.hidden_size,
            allow_mode_mismatch=args.allow_expert_mode_mismatch,
            label="arc_fl_expert",
        )
    if args.arc_fr_expert is not None:
        experts.arc_fr = load_expert_checkpoint(
            args.arc_fr_expert,
            obs_dim,
            action_dim,
            args.hidden_size,
            allow_mode_mismatch=args.allow_expert_mode_mismatch,
            label="arc_fr_expert",
        )
    if args.arc_bl_expert is not None:
        experts.arc_bl = load_expert_checkpoint(
            args.arc_bl_expert,
            obs_dim,
            action_dim,
            args.hidden_size,
            allow_mode_mismatch=args.allow_expert_mode_mismatch,
            label="arc_bl_expert",
        )
    if args.arc_br_expert is not None:
        experts.arc_br = load_expert_checkpoint(
            args.arc_br_expert,
            obs_dim,
            action_dim,
            args.hidden_size,
            allow_mode_mismatch=args.allow_expert_mode_mismatch,
            label="arc_br_expert",
        )
    return experts


def has_experts(experts: ExpertPolicies) -> bool:
    return any(
        expert is not None
        for expert in (
            experts.forward,
            experts.backward,
            experts.yaw_left,
            experts.yaw_right,
            experts.arc_fl,
            experts.arc_fr,
            experts.arc_bl,
            experts.arc_br,
        )
    )


def expert_behavior_cloning_loss(
    agent: ActorCritic,
    experts: ExpertPolicies,
    obs: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    if args.expert_bc_coef <= 0.0 or not has_experts(experts):
        return obs.new_tensor(0.0)

    forward_command = obs[:, OBS_FORWARD_COMMAND_INDEX]
    yaw_command = obs[:, OBS_YAW_COMMAND_INDEX]
    threshold = args.expert_bc_command_threshold
    masks_and_experts = (
        (
            (forward_command > threshold) & (yaw_command.abs() < threshold),
            experts.forward,
        ),
        (
            (forward_command < -threshold) & (yaw_command.abs() < threshold),
            experts.backward,
        ),
        (
            (yaw_command > threshold) & (forward_command.abs() < threshold),
            experts.yaw_left,
        ),
        (
            (yaw_command < -threshold) & (forward_command.abs() < threshold),
            experts.yaw_right,
        ),
        (
            (forward_command > threshold) & (yaw_command > threshold),
            experts.arc_fl,
        ),
        (
            (forward_command > threshold) & (yaw_command < -threshold),
            experts.arc_fr,
        ),
        (
            (forward_command < -threshold) & (yaw_command > threshold),
            experts.arc_bl,
        ),
        (
            (forward_command < -threshold) & (yaw_command < -threshold),
            experts.arc_br,
        ),
    )

    losses = []
    counts = []
    for mask, expert in masks_and_experts:
        if expert is None or not bool(mask.any()):
            continue
        selected_obs = obs[mask]
        student_action = agent.deterministic_action(selected_obs)
        with torch.no_grad():
            expert_action = expert.deterministic_action(selected_obs)
        losses.append(torch.mean((student_action - expert_action) ** 2))
        counts.append(float(mask.sum().item()))

    if not losses:
        return obs.new_tensor(0.0)

    weights = obs.new_tensor(counts) / max(1.0, float(sum(counts)))
    return torch.stack(losses).mul(weights).sum()


def pretrain_from_experts(
    agent: ActorCritic,
    dataset: ExpertDataset,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> None:
    if args.expert_pretrain_steps <= 0:
        return

    if dataset.obs.shape[0] == 0:
        print("expert_pretrain skipped=no_samples")
        return

    batch_size = min(args.expert_pretrain_batch_size, dataset.obs.shape[0])
    agent.train()
    for step in range(1, args.expert_pretrain_steps + 1):
        indices = torch.randint(0, dataset.obs.shape[0], (batch_size,))
        obs_batch = dataset.obs[indices]
        action_batch = dataset.actions[indices]
        pred_action = agent.deterministic_action(obs_batch)
        loss = torch.mean((pred_action - action_batch) ** 2)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.actor.parameters(), args.max_grad_norm)
        optimizer.step()

        if step == 1 or step == args.expert_pretrain_steps or step % 250 == 0:
            print(
                "expert_pretrain "
                f"step={step}/{args.expert_pretrain_steps} "
                f"samples={dataset.obs.shape[0]} "
                f"loss={float(loss.detach()):.6f}"
            )


def expert_replay_behavior_cloning_loss(
    agent: ActorCritic,
    dataset: ExpertDataset,
    args: argparse.Namespace,
) -> torch.Tensor:
    if args.expert_replay_bc_coef <= 0.0 or dataset.obs.shape[0] == 0:
        return next(agent.parameters()).new_tensor(0.0)

    batch_size = min(args.expert_replay_batch_size, dataset.obs.shape[0])
    indices = torch.randint(0, dataset.obs.shape[0], (batch_size,))
    obs_batch = dataset.obs[indices]
    action_batch = dataset.actions[indices]
    pred_action = agent.deterministic_action(obs_batch)
    return torch.mean((pred_action - action_batch) ** 2)


def clone_frozen_agent(agent: ActorCritic) -> ActorCritic:
    anchor = ActorCritic(
        agent.obs_dim,
        agent.action_dim,
        agent.hidden_size,
        float(agent.log_std.detach().mean()),
    )
    anchor.load_state_dict(copy.deepcopy(agent.state_dict()))
    anchor.eval()
    for parameter in anchor.parameters():
        parameter.requires_grad = False
    return anchor


def bc_anchor_loss(
    agent: ActorCritic,
    anchor: ActorCritic | None,
    obs: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    if args.bc_anchor_coef <= 0.0 or anchor is None:
        return obs.new_tensor(0.0)

    student_action = agent.deterministic_action(obs)
    with torch.no_grad():
        anchor_action = anchor.deterministic_action(obs)
    return torch.mean((student_action - anchor_action) ** 2)


def collect_expert_dataset(
    experts: ExpertPolicies,
    args: argparse.Namespace,
) -> ExpertDataset:
    command_experts: tuple[tuple[CommandSpec, ActorCritic | None], ...] = (
        (CommandSpec("expert_fwd", 1.0, 0.0), experts.forward),
        (CommandSpec("expert_back", -1.0, 0.0), experts.backward),
        (CommandSpec("expert_yaw_l", 0.0, 1.0), experts.yaw_left),
        (CommandSpec("expert_yaw_r", 0.0, -1.0), experts.yaw_right),
        (CommandSpec("expert_arc_fl", 0.7, 0.7), experts.arc_fl),
        (CommandSpec("expert_arc_fr", 0.7, -0.7), experts.arc_fr),
        (CommandSpec("expert_arc_bl", -0.7, 0.7), experts.arc_bl),
        (CommandSpec("expert_arc_br", -0.7, -0.7), experts.arc_br),
    )
    env = BramTripodEnv(
        randomize_reset=True,
        randomize_command=False,
        domain_randomization=False,
    )
    obs_samples: list[np.ndarray] = []
    action_samples: list[np.ndarray] = []
    with torch.no_grad():
        for command_index, (command, expert) in enumerate(command_experts):
            if expert is None:
                continue
            for episode in range(args.expert_pretrain_episodes):
                obs, _ = env.reset(
                    seed=args.seed + 50_000 + 1000 * command_index + episode,
                    options=command_options(command),
                )
                for _ in range(env.max_steps):
                    obs_tensor = torch.as_tensor(obs[None, :], dtype=torch.float32)
                    action = expert.deterministic_action(obs_tensor).cpu().numpy()[0]
                    obs_samples.append(obs.astype(np.float32).copy())
                    action_samples.append(action.astype(np.float32).copy())
                    obs, _, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:
                        break

    if not obs_samples:
        return ExpertDataset(
            torch.empty((0, env.observation_space.shape[0])),
            torch.empty((0, env.action_space.shape[0])),
        )
    return ExpertDataset(
        torch.as_tensor(np.stack(obs_samples), dtype=torch.float32),
        torch.as_tensor(np.stack(action_samples), dtype=torch.float32),
    )


def empty_expert_dataset(obs_dim: int, action_dim: int) -> ExpertDataset:
    return ExpertDataset(torch.empty((0, obs_dim)), torch.empty((0, action_dim)))


def normalize_advantages(
    advantages: torch.Tensor,
    obs: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    if args.advantage_normalization == "none":
        return advantages
    if args.advantage_normalization == "global":
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    if args.advantage_normalization != "command":
        raise ValueError(f"Unknown advantage normalization: {args.advantage_normalization}")

    normalized = torch.empty_like(advantages)
    family_ids = command_family_ids(obs)
    for family_id in torch.unique(family_ids):
        mask = family_ids == family_id
        values = advantages[mask]
        if values.numel() <= 1:
            normalized[mask] = values - values.mean()
        else:
            normalized[mask] = (values - values.mean()) / (values.std() + 1e-8)
    return normalized


def command_family_ids(obs: torch.Tensor) -> torch.Tensor:
    forward_command = obs[:, OBS_FORWARD_COMMAND_INDEX]
    yaw_command = obs[:, OBS_YAW_COMMAND_INDEX]
    threshold = 0.20
    family_ids = torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)
    family_ids[(forward_command > threshold) & (yaw_command.abs() <= threshold)] = 1
    family_ids[(forward_command < -threshold) & (yaw_command.abs() <= threshold)] = 2
    family_ids[(yaw_command > threshold) & (forward_command.abs() <= threshold)] = 3
    family_ids[(yaw_command < -threshold) & (forward_command.abs() <= threshold)] = 4
    family_ids[
        (forward_command.abs() > threshold) & (yaw_command.abs() > threshold)
    ] = 5
    return family_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small PPO smoke trainer for BramTripodEnv.")
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.004)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.06)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=13)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--domain-randomization-strength", type=float, default=0.45)
    parser.add_argument("--randomize-command", action="store_true")
    parser.add_argument(
        "--command-curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stage randomized commands from straight motion to full joystick control.",
    )
    parser.add_argument(
        "--gated-curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Unlock randomized-command stages using deterministic per-command eval.",
    )
    parser.add_argument(
        "--stage-hold-updates",
        type=int,
        default=5,
        help="Minimum PPO updates to spend in a gated curriculum stage before advancing.",
    )
    parser.add_argument(
        "--start-command-stage",
        type=int,
        default=None,
        help="Initial gated curriculum stage for warm-started randomized-command runs.",
    )
    parser.add_argument(
        "--gate-pass-evals",
        type=int,
        default=2,
        help="Consecutive passing evals required before advancing a gated stage.",
    )
    parser.add_argument("--gate-forward-distance", type=float, default=0.22)
    parser.add_argument("--gate-backward-distance", type=float, default=0.12)
    parser.add_argument("--gate-yaw-distance", type=float, default=0.10)
    parser.add_argument("--gate-arc-distance", type=float, default=0.05)
    parser.add_argument(
        "--weak-command-boost",
        type=float,
        default=3.0,
        help="Sampling weight multiplier for commands that failed the latest eval gate.",
    )
    parser.add_argument(
        "--advantage-normalization",
        choices=("command", "global", "none"),
        default="command",
        help="Normalize PPO advantages globally or within command families.",
    )
    parser.add_argument("--forward-command", type=float, default=1.0)
    parser.add_argument("--yaw-rate-command", type=float, default=0.0)
    parser.add_argument("--log-std-init", type=float, default=-1.2)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Initialize actor/critic weights from an existing policy checkpoint.",
    )
    parser.add_argument(
        "--allow-init-mode-mismatch",
        action="store_true",
        help="Allow --init-checkpoint from another env_command_mode if obs/action dims match.",
    )
    parser.add_argument("--forward-expert", type=Path, default=None)
    parser.add_argument("--backward-expert", type=Path, default=None)
    parser.add_argument("--yaw-left-expert", type=Path, default=None)
    parser.add_argument("--yaw-right-expert", type=Path, default=None)
    parser.add_argument("--arc-fl-expert", type=Path, default=None)
    parser.add_argument("--arc-fr-expert", type=Path, default=None)
    parser.add_argument("--arc-bl-expert", type=Path, default=None)
    parser.add_argument("--arc-br-expert", type=Path, default=None)
    parser.add_argument(
        "--allow-expert-mode-mismatch",
        action="store_true",
        help="Allow expert checkpoints from another env_command_mode if obs/action dims match.",
    )
    parser.add_argument(
        "--expert-bc-coef",
        type=float,
        default=0.0,
        help="Actor MSE loss weight for matching command-specific expert actions.",
    )
    parser.add_argument(
        "--expert-replay-bc-coef",
        type=float,
        default=0.0,
        help="Actor MSE loss weight for replaying stored primitive expert rollouts.",
    )
    parser.add_argument(
        "--bc-anchor-coef",
        type=float,
        default=0.0,
        help="Actor MSE weight for staying near the frozen post-BC policy during PPO.",
    )
    parser.add_argument(
        "--expert-bc-command-threshold",
        type=float,
        default=0.25,
        help="Minimum absolute command magnitude for applying expert BC loss.",
    )
    parser.add_argument(
        "--expert-pretrain-steps",
        type=int,
        default=0,
        help="Supervised actor pretraining steps from expert rollout datasets.",
    )
    parser.add_argument(
        "--expert-pretrain-episodes",
        type=int,
        default=3,
        help="Expert rollout episodes per provided expert for pretraining.",
    )
    parser.add_argument(
        "--expert-pretrain-batch-size",
        type=int,
        default=256,
        help="Minibatch size for expert pretraining.",
    )
    parser.add_argument(
        "--expert-replay-batch-size",
        type=int,
        default=256,
        help="Minibatch size for expert replay BC during PPO updates.",
    )
    parser.add_argument(
        "--bc-only",
        action="store_true",
        help="Run expert BC pretraining, evaluate/save, and exit before PPO.",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=0,
        help="Save policy snapshots every N PPO updates for retroactive visualization.",
    )
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
    snapshot_dir = run_dir / "snapshots"
    if args.snapshot_interval > 0:
        snapshot_dir.mkdir(exist_ok=True)

    use_gated_curriculum = (
        args.randomize_command and args.command_curriculum and args.gated_curriculum
    )
    command_stage = 0 if use_gated_curriculum else 4
    if use_gated_curriculum and args.start_command_stage is not None:
        command_stage = int(
            np.clip(args.start_command_stage, 0, len(CURRICULUM_STAGE_LEVELS) - 1)
        )
    stage_enter_update = 1
    stage_pass_streak = 0
    weak_commands: tuple[str, ...] = ()
    initial_curriculum_level = stage_to_curriculum_level(command_stage)
    if not use_gated_curriculum:
        initial_curriculum_level = 0.0 if args.command_curriculum else 1.0

    envs = [
        BramTripodEnv(
            domain_randomization=args.domain_randomization,
            domain_randomization_strength=args.domain_randomization_strength,
            randomize_command=args.randomize_command,
            command_curriculum=args.command_curriculum,
            command_curriculum_level=initial_curriculum_level,
            command_forward=args.forward_command,
            command_yaw_rate=args.yaw_rate_command,
        )
        for _ in range(args.num_envs)
    ]
    obs_list = []
    for index, env in enumerate(envs):
        options = training_command_options(command_stage, weak_commands, rng, args)
        obs, _ = env.reset(seed=args.seed + index, options=options)
        obs_list.append(obs)
    obs_np = np.stack(obs_list).astype(np.float32)

    obs_dim = obs_np.shape[1]
    action_dim = envs[0].action_space.shape[0]
    agent = ActorCritic(obs_dim, action_dim, args.hidden_size, args.log_std_init)
    if args.init_checkpoint is not None:
        load_initial_checkpoint(
            args.init_checkpoint,
            agent,
            allow_mode_mismatch=args.allow_init_mode_mismatch,
            label="init_checkpoint",
        )
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    experts = load_expert_policies(args, obs_dim, action_dim)
    expert_dataset = empty_expert_dataset(obs_dim, action_dim)
    if has_experts(experts) and (
        args.expert_pretrain_steps > 0
        or args.expert_replay_bc_coef > 0.0
        or args.bc_only
    ):
        expert_dataset = collect_expert_dataset(experts, args)
        print(f"expert_dataset samples={expert_dataset.obs.shape[0]}")
    pretrain_from_experts(agent, expert_dataset, optimizer, args)
    bc_anchor = clone_frozen_agent(agent) if args.bc_anchor_coef > 0.0 else None

    if args.bc_only:
        bc_stats = evaluate_policy(agent, args.eval_episodes, args.seed + 30_000, args)
        save_checkpoint(checkpoint_path, agent, args, bc_stats)
        save_checkpoint(best_checkpoint_path, agent, args, bc_stats)
        write_single_eval_metrics(
            log_path,
            bc_stats,
            command_stage=command_stage,
            command_curriculum_level=initial_curriculum_level,
        )
        print(
            "bc_only_eval "
            f"reward={bc_stats.reward:.3f} "
            f"score={bc_stats.score:.3f} "
            f"distance={bc_stats.distance:.4f} "
            f"passes={bc_stats.primitive_pass_count} "
            f"deficit={bc_stats.primitive_deficit:.3f} "
            f"length={bc_stats.length:.1f}"
        )
        print(f"saved_policy={checkpoint_path}")
        print(f"saved_best_policy={best_checkpoint_path}")
        print(f"metrics={log_path}")
        return

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
        f"score={random_stats.score:.3f} "
        f"distance={random_stats.distance:.4f} "
        f"length={random_stats.length:.1f}"
    )
    print(
        f"training total_steps={actual_total_steps} num_envs={args.num_envs} "
        f"rollout_steps={args.rollout_steps} obs_dim={obs_dim} action_dim={action_dim} "
        f"domain_randomization={args.domain_randomization} "
        f"domain_randomization_strength={args.domain_randomization_strength:.2f} "
        f"randomize_command={args.randomize_command} "
        f"command_curriculum={args.command_curriculum} "
        f"gated_curriculum={use_gated_curriculum} "
        f"command_stage={command_stage} "
        f"advantage_normalization={args.advantage_normalization} "
        f"expert_replay_bc_coef={args.expert_replay_bc_coef:.3f} "
        f"bc_anchor_coef={args.bc_anchor_coef:.3f} "
        f"forward_command={args.forward_command:.2f} "
        f"yaw_rate_command={args.yaw_rate_command:.2f}"
    )

    best_eval_score = -float("inf")
    global_step = 0
    start_time = time.perf_counter()

    with log_path.open("w", newline="") as log_file:
        writer = csv.DictWriter(
            log_file,
            fieldnames=metrics_fieldnames(),
        )
        writer.writeheader()

        for update in range(1, num_updates + 1):
            rollout_stage = command_stage
            command_curriculum_level = active_curriculum_level(
                update,
                num_updates,
                rollout_stage,
                args,
            )
            for env in envs:
                env.set_command_curriculum_level(command_curriculum_level)

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
                        options = training_command_options(
                            rollout_stage,
                            weak_commands,
                            rng,
                            args,
                        )
                        obs, _ = env.reset(
                            seed=int(rng.integers(0, 2**31 - 1)),
                            options=options,
                        )

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

            batch_advantages = normalize_advantages(batch_advantages, batch_obs, args)

            batch_size = args.num_envs * args.rollout_steps
            minibatch_size = min(args.minibatch_size, batch_size)
            policy_losses = []
            expert_bc_losses = []
            bc_anchor_losses = []
            value_losses = []
            entropies = []
            approx_kls = []
            stop_policy_update = False
            for _ in range(args.update_epochs):
                indices = torch.randperm(batch_size)
                for start in range(0, batch_size, minibatch_size):
                    mb_idx = indices[start : start + minibatch_size]
                    _, new_logprob, entropy, new_value = agent.get_action_and_value(
                        batch_obs[mb_idx], batch_actions[mb_idx]
                    )
                    rollout_expert_bc_loss = expert_behavior_cloning_loss(
                        agent,
                        experts,
                        batch_obs[mb_idx],
                        args,
                    )
                    replay_expert_bc_loss = expert_replay_behavior_cloning_loss(
                        agent,
                        expert_dataset,
                        args,
                    )
                    anchor_loss = bc_anchor_loss(
                        agent,
                        bc_anchor,
                        batch_obs[mb_idx],
                        args,
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
                        + args.expert_bc_coef * rollout_expert_bc_loss
                        + args.expert_replay_bc_coef * replay_expert_bc_loss
                        + args.bc_anchor_coef * anchor_loss
                        - args.entropy_coef * entropy_loss
                        + args.value_coef * value_loss
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                    policy_losses.append(float(policy_loss.detach()))
                    expert_bc_losses.append(
                        float(
                            (
                                rollout_expert_bc_loss
                                + replay_expert_bc_loss
                            ).detach()
                        )
                    )
                    bc_anchor_losses.append(float(anchor_loss.detach()))
                    value_losses.append(float(value_loss.detach()))
                    entropies.append(float(entropy_loss.detach()))
                    approx_kls.append(float(approx_kl.detach()))
                    if args.target_kl > 0.0 and float(approx_kl.detach()) > args.target_kl:
                        stop_policy_update = True
                        break
                if stop_policy_update:
                    break

            should_snapshot = (
                args.snapshot_interval > 0
                and (
                    update == 1
                    or update == num_updates
                    or update % args.snapshot_interval == 0
                )
            )
            should_eval = (
                update == 1
                or update == num_updates
                or update % args.eval_interval == 0
                or should_snapshot
            )
            eval_stats = EvalStats(
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
            )
            if should_eval:
                eval_stats = evaluate_policy(
                    agent, args.eval_episodes, args.seed + 20_000 + update, args
                )
                if args.randomize_command:
                    weak_commands = weak_command_names(eval_stats, command_stage, args)
                else:
                    weak_commands = ()
                if use_gated_curriculum:
                    if stage_passes(command_stage, eval_stats, args):
                        stage_pass_streak += 1
                    else:
                        stage_pass_streak = 0
                    next_stage = maybe_advance_stage(
                        command_stage,
                        eval_stats,
                        update - stage_enter_update + 1,
                        stage_pass_streak,
                        args,
                    )
                    if next_stage != command_stage:
                        command_stage = next_stage
                        stage_enter_update = update + 1
                        stage_pass_streak = 0
                        weak_commands = weak_command_names(
                            eval_stats,
                            command_stage,
                            args,
                        )
                if eval_stats.score > best_eval_score:
                    best_eval_score = eval_stats.score
                    save_checkpoint(best_checkpoint_path, agent, args, eval_stats)
                if should_snapshot:
                    save_checkpoint(
                        snapshot_dir
                        / f"policy_update_{update:05d}_step_{global_step}.pt",
                        agent,
                        args,
                        eval_stats,
                    )

            elapsed = time.perf_counter() - start_time
            sps = int(global_step / max(elapsed, 1e-9))
            row = {
                "update": update,
                "global_step": global_step,
                "command_stage": rollout_stage,
                "command_curriculum_level": command_curriculum_level,
                "stage_pass_streak": stage_pass_streak,
                "weak_commands": "|".join(weak_commands),
                "primitive_pass_count": eval_stats.primitive_pass_count,
                "sps": sps,
                "recent_return": mean_or_nan(recent_returns),
                "recent_distance": mean_or_nan(recent_distances),
                "recent_length": mean_or_nan(recent_lengths),
                "eval_reward": eval_stats.reward,
                "eval_score": eval_stats.score,
                "eval_distance": eval_stats.distance,
                "eval_length": eval_stats.length,
                "eval_worst_reward": eval_stats.worst_reward,
                "eval_worst_length": eval_stats.worst_length,
                "eval_primitive_deficit": eval_stats.primitive_deficit,
                "eval_command_deficit": eval_stats.command_deficit,
                "policy_loss": float(np.mean(policy_losses)),
                "expert_bc_loss": float(np.mean(expert_bc_losses)),
                "bc_anchor_loss": float(np.mean(bc_anchor_losses)),
                "value_loss": float(np.mean(value_losses)),
                "entropy": float(np.mean(entropies)),
                "approx_kl": float(np.mean(approx_kls)),
            }
            row.update(per_command_row(eval_stats))
            writer.writerow(row)
            log_file.flush()

            if update == 1 or update == num_updates or update % 10 == 0 or should_eval:
                eval_dist = f"{row['eval_distance']:.4f}" if should_eval else "skip"
                eval_score = f"{row['eval_score']:.3f}" if should_eval else "skip"
                print(
                    f"update={update:04d}/{num_updates} "
                    f"step={global_step} sps={sps} "
                    f"stage={rollout_stage}->{command_stage} "
                    f"cmd_level={command_curriculum_level:.2f} "
                    f"recent_return={row['recent_return']:.3f} "
                    f"recent_dist={row['recent_distance']:.4f} "
                    f"eval_dist={eval_dist} "
                    f"eval_score={eval_score} "
                    f"passes={row['primitive_pass_count']} "
                    f"streak={row['stage_pass_streak']} "
                    f"weak={row['weak_commands'] or '-'} "
                    f"entropy={row['entropy']:.3f}"
                )

    final_stats = evaluate_policy(agent, args.eval_episodes, args.seed + 30_000, args)
    save_checkpoint(checkpoint_path, agent, args, final_stats)
    print(
        "final_eval "
        f"reward={final_stats.reward:.3f} "
        f"score={final_stats.score:.3f} "
        f"distance={final_stats.distance:.4f} "
        f"length={final_stats.length:.1f}"
    )
    print(f"saved_policy={checkpoint_path}")
    print(f"saved_best_policy={best_checkpoint_path}")
    print(f"metrics={log_path}")


def evaluate_random(episodes: int, seed: int, args: argparse.Namespace) -> EvalStats:
    env = BramTripodEnv(
        randomize_reset=False,
        randomize_command=False,
        domain_randomization_strength=args.domain_randomization_strength,
        command_forward=args.forward_command,
        command_yaw_rate=args.yaw_rate_command,
    )
    results: list[EpisodeResult] = []
    rng = np.random.default_rng(seed)
    for episode, command in enumerate(eval_command_specs(args, episodes)):
        obs, _ = env.reset(
            seed=seed + episode,
            options=command_options(command),
        )
        total_reward = 0.0
        final_info = empty_final_info()
        for length in range(env.max_steps):
            action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
            obs, reward, terminated, truncated, final_info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        results.append(episode_result(command.name, total_reward, final_info, length + 1))
    return make_eval_stats(results, env.max_steps, args)


def evaluate_policy(
    agent: ActorCritic,
    episodes: int,
    seed: int,
    args: argparse.Namespace,
) -> EvalStats:
    env = BramTripodEnv(
        randomize_reset=False,
        randomize_command=False,
        domain_randomization_strength=args.domain_randomization_strength,
        command_forward=args.forward_command,
        command_yaw_rate=args.yaw_rate_command,
    )
    results: list[EpisodeResult] = []
    agent.eval()
    with torch.no_grad():
        for episode, command in enumerate(eval_command_specs(args, episodes)):
            obs, _ = env.reset(
                seed=seed + episode,
                options=command_options(command),
            )
            total_reward = 0.0
            final_info = empty_final_info()
            for length in range(env.max_steps):
                obs_tensor = torch.as_tensor(obs[None, :], dtype=torch.float32)
                action = agent.deterministic_action(obs_tensor).cpu().numpy()[0]
                obs, reward, terminated, truncated, final_info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            results.append(episode_result(command.name, total_reward, final_info, length + 1))
    agent.train()
    return make_eval_stats(results, env.max_steps, args)


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
            "eval_score": eval_stats.score,
            "eval_distance": eval_stats.distance,
            "eval_length": eval_stats.length,
            "eval_worst_reward": eval_stats.worst_reward,
            "eval_worst_length": eval_stats.worst_length,
            "eval_primitive_pass_count": eval_stats.primitive_pass_count,
            "eval_primitive_deficit": eval_stats.primitive_deficit,
            "eval_command_deficit": eval_stats.command_deficit,
            "eval_weak_commands": eval_stats.weak_commands,
            "eval_per_command": eval_stats.per_command,
        },
        path,
    )


def metrics_fieldnames() -> list[str]:
    return [
        "update",
        "global_step",
        "command_stage",
        "command_curriculum_level",
        "stage_pass_streak",
        "weak_commands",
        "primitive_pass_count",
        "sps",
        "recent_return",
        "recent_distance",
        "recent_length",
        "eval_reward",
        "eval_score",
        "eval_distance",
        "eval_length",
        "eval_worst_reward",
        "eval_worst_length",
        "eval_primitive_deficit",
        "eval_command_deficit",
        "policy_loss",
        "expert_bc_loss",
        "bc_anchor_loss",
        "value_loss",
        "entropy",
        "approx_kl",
    ] + per_command_fieldnames()


def write_single_eval_metrics(
    path: Path,
    eval_stats: EvalStats,
    *,
    command_stage: int,
    command_curriculum_level: float,
) -> None:
    row = {
        "update": 0,
        "global_step": 0,
        "command_stage": command_stage,
        "command_curriculum_level": command_curriculum_level,
        "stage_pass_streak": 0,
        "weak_commands": "|".join(eval_stats.weak_commands),
        "primitive_pass_count": eval_stats.primitive_pass_count,
        "sps": 0,
        "recent_return": float("nan"),
        "recent_distance": float("nan"),
        "recent_length": float("nan"),
        "eval_reward": eval_stats.reward,
        "eval_score": eval_stats.score,
        "eval_distance": eval_stats.distance,
        "eval_length": eval_stats.length,
        "eval_worst_reward": eval_stats.worst_reward,
        "eval_worst_length": eval_stats.worst_length,
        "eval_primitive_deficit": eval_stats.primitive_deficit,
        "eval_command_deficit": eval_stats.command_deficit,
        "policy_loss": float("nan"),
        "expert_bc_loss": float("nan"),
        "bc_anchor_loss": float("nan"),
        "value_loss": float("nan"),
        "entropy": float("nan"),
        "approx_kl": float("nan"),
    }
    row.update(per_command_row(eval_stats))
    with path.open("w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=metrics_fieldnames())
        writer.writeheader()
        writer.writerow(row)


def mean_or_nan(values: deque[float] | deque[int]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def active_curriculum_level(
    update: int,
    num_updates: int,
    command_stage: int,
    args: argparse.Namespace,
) -> float:
    if not args.randomize_command or not args.command_curriculum:
        return 1.0
    if args.gated_curriculum:
        return stage_to_curriculum_level(command_stage)

    progress = (update - 1) / max(1, num_updates - 1)
    if progress < 0.28:
        return 0.0
    if progress < 0.48:
        return 0.25
    if progress < 0.68:
        return 0.50
    if progress < 0.86:
        return 0.72
    return 1.0


def stage_to_curriculum_level(stage: int) -> float:
    index = int(np.clip(stage, 0, len(CURRICULUM_STAGE_LEVELS) - 1))
    return CURRICULUM_STAGE_LEVELS[index]


def distance_from_info(info: dict) -> float:
    return float(info.get("command_distance", info.get("x_distance", 0.0)))


def make_eval_stats(
    results: list[EpisodeResult],
    max_steps: int,
    args: argparse.Namespace,
) -> EvalStats:
    rewards = [result.reward for result in results]
    distances = [result.command_distance for result in results]
    lengths = [result.length for result in results]
    mean_reward = float(np.mean(rewards))
    mean_distance = float(np.mean(distances))
    mean_length = float(np.mean(lengths))
    worst_reward = float(np.min(rewards))
    worst_length = float(np.min(lengths))
    survival_shortfall = max(0.0, 0.90 * max_steps - worst_length)
    per_command = aggregate_command_results(results)
    primitive_pass_count = 0
    primitive_deficit = 0.0
    command_deficit = 0.0
    if args.randomize_command:
        primitive_pass_count = sum(
            1 for name in ("fwd1", "back1", "yaw_l1", "yaw_r1")
            if command_passes(name, per_command, args)
        )
        primitive_deficit = primitive_command_deficit(per_command, args)
        command_deficit = command_deficit_for(stage_required_names(5), per_command, args)
    required_quality = required_command_quality(per_command, args)
    score = mean_reward + 55.0 * mean_distance - 0.50 * survival_shortfall
    if args.randomize_command:
        score = (
            mean_reward
            + 45.0 * mean_distance
            + 0.20 * worst_reward
            + 35.0 * primitive_pass_count
            + 60.0 * required_quality
            - 130.0 * primitive_deficit
            - 90.0 * command_deficit
            - 0.75 * survival_shortfall
        )
    weak = ()
    if args.randomize_command:
        weak = weak_command_names_from_per_command(per_command, 4, args)
    return EvalStats(
        mean_reward,
        mean_distance,
        mean_length,
        float(score),
        worst_reward,
        worst_length,
        primitive_pass_count,
        primitive_deficit,
        command_deficit,
        weak,
        per_command,
    )


def eval_command_specs(args: argparse.Namespace, episodes: int) -> list[CommandSpec]:
    if not args.randomize_command:
        fixed = CommandSpec(
            "fixed",
            float(np.clip(args.forward_command, -1.0, 1.0)),
            float(np.clip(args.yaw_rate_command, -1.0, 1.0)),
        )
        return [fixed for _ in range(max(1, episodes))]

    count = max(episodes, len(EVAL_COMMANDS))
    repeats = max(1, math.ceil(count / len(EVAL_COMMANDS)))
    return list((EVAL_COMMANDS * repeats)[:count])


def command_options(command: CommandSpec) -> dict[str, float]:
    return {
        "forward_command": float(np.clip(command.forward, -1.0, 1.0)),
        "yaw_rate_command": float(np.clip(command.yaw_rate, -1.0, 1.0)),
    }


def empty_final_info() -> dict[str, float]:
    return {"command_distance": 0.0, "line_distance": 0.0, "yaw_distance": 0.0}


def episode_result(
    name: str,
    total_reward: float,
    final_info: dict,
    length: int,
) -> EpisodeResult:
    return EpisodeResult(
        name=name,
        reward=float(total_reward),
        command_distance=distance_from_info(final_info),
        line_distance=float(final_info.get("line_distance", 0.0)),
        yaw_distance=float(final_info.get("yaw_distance", 0.0)),
        length=int(length),
    )


def aggregate_command_results(
    results: list[EpisodeResult],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[EpisodeResult]] = {}
    for result in results:
        grouped.setdefault(result.name, []).append(result)

    per_command: dict[str, dict[str, float]] = {}
    for name, command_results in grouped.items():
        per_command[name] = {
            "reward": float(np.mean([result.reward for result in command_results])),
            "cmd": float(
                np.mean([result.command_distance for result in command_results])
            ),
            "line": float(np.mean([result.line_distance for result in command_results])),
            "yaw": float(np.mean([result.yaw_distance for result in command_results])),
            "length": float(np.mean([result.length for result in command_results])),
        }
    return per_command


def per_command_fieldnames() -> list[str]:
    fields: list[str] = []
    for command in EVAL_COMMANDS:
        for metric in ("reward", "cmd", "line", "yaw", "length"):
            fields.append(f"eval_{command.name}_{metric}")
    return fields


def per_command_row(eval_stats: EvalStats) -> dict[str, float]:
    row: dict[str, float] = {}
    for command in EVAL_COMMANDS:
        values = eval_stats.per_command.get(command.name, {})
        for metric in ("reward", "cmd", "line", "yaw", "length"):
            row[f"eval_{command.name}_{metric}"] = float(
                values.get(metric, float("nan"))
            )
    return row


def command_thresholds(args: argparse.Namespace) -> dict[str, float]:
    return {
        "fwd1": args.gate_forward_distance,
        "back1": args.gate_backward_distance,
        "fwd05": 0.45 * args.gate_forward_distance,
        "back05": 0.45 * args.gate_backward_distance,
        "yaw_l1": args.gate_yaw_distance,
        "yaw_r1": args.gate_yaw_distance,
        "yaw_l05": 0.45 * args.gate_yaw_distance,
        "yaw_r05": 0.45 * args.gate_yaw_distance,
        "arc_fl": args.gate_arc_distance,
        "arc_fr": args.gate_arc_distance,
        "arc_bl": args.gate_arc_distance,
        "arc_br": args.gate_arc_distance,
    }


def command_passes(
    name: str,
    per_command: dict[str, dict[str, float]],
    args: argparse.Namespace,
) -> bool:
    threshold = command_thresholds(args).get(name)
    if threshold is None:
        return True
    return per_command.get(name, {}).get("cmd", -float("inf")) >= threshold


def required_command_quality(
    per_command: dict[str, dict[str, float]],
    args: argparse.Namespace,
) -> float:
    ratios = []
    thresholds = command_thresholds(args)
    for name in ("fwd1", "back1", "yaw_l1", "yaw_r1"):
        threshold = max(1e-6, thresholds[name])
        ratios.append(per_command.get(name, {}).get("cmd", -threshold) / threshold)
    return float(np.clip(min(ratios), -2.0, 2.0))


def primitive_command_deficit(
    per_command: dict[str, dict[str, float]],
    args: argparse.Namespace,
) -> float:
    return command_deficit_for(("fwd1", "back1", "yaw_l1", "yaw_r1"), per_command, args)


def command_deficit_for(
    names: tuple[str, ...],
    per_command: dict[str, dict[str, float]],
    args: argparse.Namespace,
) -> float:
    deficit = 0.0
    thresholds = command_thresholds(args)
    for name in names:
        if name not in thresholds:
            continue
        threshold = max(1e-6, thresholds[name])
        command_distance = per_command.get(name, {}).get("cmd", -threshold)
        deficit += max(0.0, (threshold - command_distance) / threshold)
    return float(deficit)


def stage_required_names(stage: int) -> tuple[str, ...]:
    if stage <= 0:
        return ("fwd1",)
    if stage == 1:
        return ("fwd1", "back1")
    if stage == 2:
        return ("fwd1", "back1", "yaw_l1")
    if stage == 3:
        return ("fwd1", "back1", "yaw_l1", "yaw_r1")
    if stage == 4:
        return (
            "fwd1",
            "back1",
            "yaw_l1",
            "yaw_r1",
            "arc_fl",
            "arc_fr",
            "arc_bl",
            "arc_br",
        )
    return (
        "fwd1",
        "back1",
        "fwd05",
        "back05",
        "yaw_l1",
        "yaw_r1",
        "yaw_l05",
        "yaw_r05",
        "arc_fl",
        "arc_fr",
        "arc_bl",
        "arc_br",
    )


def weak_command_names(
    eval_stats: EvalStats,
    stage: int,
    args: argparse.Namespace,
) -> tuple[str, ...]:
    return weak_command_names_from_per_command(eval_stats.per_command, stage, args)


def weak_command_names_from_per_command(
    per_command: dict[str, dict[str, float]],
    stage: int,
    args: argparse.Namespace,
) -> tuple[str, ...]:
    return tuple(
        name for name in stage_required_names(stage)
        if not command_passes(name, per_command, args)
    )


def maybe_advance_stage(
    stage: int,
    eval_stats: EvalStats,
    updates_in_stage: int,
    stage_pass_streak: int,
    args: argparse.Namespace,
) -> int:
    if stage >= len(CURRICULUM_STAGE_LEVELS) - 1:
        return stage
    if updates_in_stage < args.stage_hold_updates:
        return stage
    if stage_pass_streak >= args.gate_pass_evals and stage_passes(
        stage,
        eval_stats,
        args,
    ):
        return stage + 1
    return stage


def stage_passes(
    stage: int,
    eval_stats: EvalStats,
    args: argparse.Namespace,
) -> bool:
    required = stage_required_names(stage)
    return all(command_passes(name, eval_stats.per_command, args) for name in required)


def training_command_options(
    stage: int,
    weak_commands: tuple[str, ...],
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> dict[str, float] | None:
    if not args.randomize_command:
        return None
    command = sample_training_command(stage, weak_commands, rng, args)
    return command_options(command)


def sample_training_command(
    stage: int,
    weak_commands: tuple[str, ...],
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> CommandSpec:
    families = stage_training_families(stage)
    weak_families = command_families(weak_commands)
    weights = np.array(
        [
            weight
            * (
                args.weak_command_boost
                if stage > 0 and family in weak_families
                else 1.0
            )
            for family, weight in families
        ],
        dtype=np.float64,
    )
    weights = weights / np.sum(weights)
    index = int(rng.choice(len(families), p=weights))
    family = families[index][0]
    return sample_family_command(family, rng)


def stage_training_families(stage: int) -> tuple[tuple[str, float], ...]:
    if stage <= 0:
        return (("idle", 0.05), ("fwd", 0.75), ("back", 0.20))
    if stage == 1:
        return (("idle", 0.04), ("fwd", 0.43), ("back", 0.53))
    if stage == 2:
        return (
            ("idle", 0.03),
            ("fwd", 0.23),
            ("back", 0.23),
            ("yaw_l", 0.51),
        )
    if stage == 3:
        return (
            ("idle", 0.03),
            ("fwd", 0.18),
            ("back", 0.18),
            ("yaw_l", 0.18),
            ("yaw_r", 0.43),
        )
    if stage == 4:
        return (
            ("idle", 0.04),
            ("fwd", 0.14),
            ("back", 0.14),
            ("yaw_l", 0.14),
            ("yaw_r", 0.14),
            ("arc_fl", 0.10),
            ("arc_fr", 0.10),
            ("arc_bl", 0.10),
            ("arc_br", 0.10),
        )
    return (
        ("idle", 0.04),
        ("fwd", 0.10),
        ("back", 0.10),
        ("yaw_l", 0.10),
        ("yaw_r", 0.10),
        ("arc_fl", 0.14),
        ("arc_fr", 0.14),
        ("arc_bl", 0.14),
        ("arc_br", 0.14),
    )


def command_families(command_names: tuple[str, ...]) -> set[str]:
    families: set[str] = set()
    for name in command_names:
        if name.startswith("fwd"):
            families.add("fwd")
        elif name.startswith("back"):
            families.add("back")
        elif name.startswith("yaw_l"):
            families.add("yaw_l")
        elif name.startswith("yaw_r"):
            families.add("yaw_r")
        elif name.startswith("arc_"):
            families.add(name)
    return families


def sample_family_command(family: str, rng: np.random.Generator) -> CommandSpec:
    if family == "idle":
        return CommandSpec("idle", 0.0, 0.0)
    if family == "fwd":
        return CommandSpec("fwd_train", float(rng.uniform(0.45, 1.0)), 0.0)
    if family == "back":
        return CommandSpec("back_train", -float(rng.uniform(0.25, 0.85)), 0.0)
    if family == "yaw_l":
        return CommandSpec("yaw_l_train", 0.0, float(rng.uniform(0.35, 1.0)))
    if family == "yaw_r":
        return CommandSpec("yaw_r_train", 0.0, -float(rng.uniform(0.35, 1.0)))
    if family == "arc_fl":
        return CommandSpec(
            "arc_fl_train",
            float(rng.uniform(0.35, 0.90)),
            float(rng.uniform(0.25, 0.80)),
        )
    if family == "arc_fr":
        return CommandSpec(
            "arc_fr_train",
            float(rng.uniform(0.35, 0.90)),
            -float(rng.uniform(0.25, 0.80)),
        )
    if family == "arc_bl":
        return CommandSpec(
            "arc_bl_train",
            -float(rng.uniform(0.35, 0.90)),
            float(rng.uniform(0.25, 0.80)),
        )
    if family == "arc_br":
        return CommandSpec(
            "arc_br_train",
            -float(rng.uniform(0.35, 0.90)),
            -float(rng.uniform(0.25, 0.80)),
        )
    raise ValueError(f"Unknown command family: {family}")


if __name__ == "__main__":
    main()
