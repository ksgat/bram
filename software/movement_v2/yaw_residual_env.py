from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

GAIT_DISCOVERY_DIR = Path(__file__).resolve().parents[1] / "gait_discovery"
if str(GAIT_DISCOVERY_DIR) not in sys.path:
    sys.path.insert(0, str(GAIT_DISCOVERY_DIR))

from search_gait import gait_action, load_params  # noqa: E402
from yaw_env import BramV2PrimitiveEnv, OBS_DIM, YAW_ENV_COMMAND_MODE  # noqa: E402


YAW_RESIDUAL_COMMAND_MODE = f"{YAW_ENV_COMMAND_MODE}_yaw_table_residual"
DEFAULT_YAW_TABLE_DIR = (
    Path(__file__).resolve().parent
    / "exports"
    / "yaw_tables"
    / "gait_discovery_planar_40hz_20260711"
)
DEFAULT_LEFT_TABLE = DEFAULT_YAW_TABLE_DIR / "yaw-left_policy_table_40hz_scale0p35.json"
DEFAULT_RIGHT_TABLE = DEFAULT_YAW_TABLE_DIR / "yaw-right_policy_table_40hz_scale1p60.json"


class BramV2YawResidualEnv(gym.Env):
    """Yaw-only residual env around fixed table primitives.

    The base observation block is unchanged from movement_v2 primitives:
    previous 4 IMU quaternions, previous 6 emitted servo actions, yaw command.
    Residual-specific features append the current base action, previous base
    action, and base table phase sin/cos.

    The policy action is a normalized residual. The servo command sent to MuJoCo is:

        final = yaw_table(t, yaw_cmd) + residual_limit * abs(yaw_cmd) * residual

    Drift is tracked throughout the rollout with final/mean/max/RMS metrics so the
    yaw-in-place gate is visible during training and deterministic eval.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        left_table: str | Path = DEFAULT_LEFT_TABLE,
        right_table: str | Path = DEFAULT_RIGHT_TABLE,
        left_params: str | Path | None = None,
        right_params: str | Path | None = None,
        frame_skip: int = 10,
        episode_seconds: float = 8.0,
        randomize_reset: bool = True,
        domain_randomization: bool = False,
        domain_randomization_strength: float = 0.15,
        randomize_yaw_command: bool = True,
        yaw_command: float = 1.0,
        yaw_min_magnitude: float = 0.35,
        residual_limit: float = 0.18,
        target_yaw_rate: float = 0.36,
        final_drift_limit_m: float = 0.040,
        mean_drift_limit_m: float = 0.025,
        max_drift_limit_m: float = 0.040,
        slew_limit: float = 0.0,
        residual_delta_weight: float = 1.4,
        residual_accel_weight: float = 2.0,
        final_delta_weight: float = 1.2,
        final_accel_weight: float = 1.8,
    ) -> None:
        super().__init__()
        self.env = BramV2PrimitiveEnv(
            frame_skip=frame_skip,
            episode_seconds=episode_seconds,
            randomize_reset=randomize_reset,
            domain_randomization=domain_randomization,
            domain_randomization_strength=domain_randomization_strength,
            primitive="yaw",
            randomize_yaw_command=randomize_yaw_command,
            yaw_command=yaw_command,
            yaw_min_magnitude=yaw_min_magnitude,
        )
        self.left_base = load_base_source(table_path=left_table, params_path=left_params)
        self.right_base = load_base_source(table_path=right_table, params_path=right_params)
        self.left_table_path = str(left_table)
        self.right_table_path = str(right_table)
        self.left_params_path = None if left_params is None else str(left_params)
        self.right_params_path = None if right_params is None else str(right_params)
        self.residual_limit = float(residual_limit)
        self.target_yaw_rate = float(target_yaw_rate)
        self.final_drift_limit_m = float(final_drift_limit_m)
        self.mean_drift_limit_m = float(mean_drift_limit_m)
        self.max_drift_limit_m = float(max_drift_limit_m)
        self.slew_limit = float(max(0.0, slew_limit))
        self.residual_delta_weight = float(residual_delta_weight)
        self.residual_accel_weight = float(residual_accel_weight)
        self.final_delta_weight = float(final_delta_weight)
        self.final_accel_weight = float(final_accel_weight)

        self.previous_residual = np.zeros(3, dtype=np.float32)
        self.previous_residual_delta = np.zeros(3, dtype=np.float32)
        self.previous_base_action = np.zeros(3, dtype=np.float32)
        self.previous_final_action = np.zeros(3, dtype=np.float32)
        self.previous_final_delta = np.zeros(3, dtype=np.float32)
        self.drift_values: deque[float] = deque()

        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBS_DIM + 8,),
            dtype=np.float32,
        )

    @property
    def max_steps(self) -> int:
        return self.env.max_steps

    @property
    def dt(self) -> float:
        return self.env.dt

    @property
    def steps(self) -> int:
        return self.env.steps

    @property
    def yaw_command(self) -> float:
        return self.env.yaw_command

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.previous_residual[:] = 0.0
        self.previous_residual_delta[:] = 0.0
        self.previous_base_action[:] = self.base_action(0.0, self.yaw_command)
        self.previous_final_action[:] = 0.0
        self.previous_final_delta[:] = 0.0
        self.drift_values.clear()
        info = dict(info)
        info["env_command_mode"] = YAW_RESIDUAL_COMMAND_MODE
        return self._augment_obs(obs), info

    def step(
        self,
        residual_action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        residual = np.clip(np.asarray(residual_action, dtype=np.float32), -1.0, 1.0)
        base = self.base_action(self.env.steps * self.env.dt, self.yaw_command)
        previous_base = self.previous_base_action.copy()
        final_action = np.clip(
            base + self.residual_limit * abs(float(self.yaw_command)) * residual,
            -1.0,
            1.0,
        ).astype(np.float32)
        if self.slew_limit > 0.0:
            final_action = np.clip(
                final_action,
                self.previous_final_action - self.slew_limit,
                self.previous_final_action + self.slew_limit,
            ).astype(np.float32)

        residual_delta = residual - self.previous_residual
        residual_accel = residual_delta - self.previous_residual_delta
        final_delta = final_action - self.previous_final_action
        final_accel = final_delta - self.previous_final_delta

        obs, _, terminated, truncated, info = self.env.step(final_action)
        info = dict(info)
        reward = self.residual_reward(
            info=info,
            residual=residual,
            residual_delta=residual_delta,
            residual_accel=residual_accel,
            final_action=final_action,
            final_delta=final_delta,
            final_accel=final_accel,
            terminated=terminated,
        )

        self.previous_residual[:] = residual
        self.previous_residual_delta[:] = residual_delta
        self.previous_base_action[:] = base
        self.previous_final_action[:] = final_action
        self.previous_final_delta[:] = final_delta

        info["v2_residual_base_action"] = base.astype(float).tolist()
        info["v2_residual_previous_base_action"] = previous_base.astype(float).tolist()
        info["v2_residual_action"] = residual.astype(float).tolist()
        info["v2_residual_final_action"] = final_action.astype(float).tolist()
        info["v2_residual_reward"] = float(reward)
        info["env_command_mode"] = YAW_RESIDUAL_COMMAND_MODE
        return self._augment_obs(obs), float(reward), terminated, truncated, info

    def render(self) -> None:
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    def base_action(self, t: float, yaw_command: float) -> np.ndarray:
        magnitude = abs(float(yaw_command))
        if magnitude < 1.0e-6:
            return np.zeros(3, dtype=np.float32)
        source = self.left_base if yaw_command > 0.0 else self.right_base
        return np.clip(magnitude * sample_base_source(source, float(t)), -1.0, 1.0).astype(
            np.float32
        )

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        t = self.env.steps * self.env.dt
        base = self.base_action(t, self.yaw_command)
        phase = self.base_phase(t, self.yaw_command)
        residual_features = np.concatenate(
            [
                base.astype(np.float32),
                self.previous_base_action.astype(np.float32),
                phase.astype(np.float32),
            ]
        )
        return np.concatenate([obs.astype(np.float32), residual_features]).astype(np.float32)

    def base_phase(self, t: float, yaw_command: float) -> np.ndarray:
        source = self.left_base if yaw_command > 0.0 else self.right_base
        phase = base_source_phase(source, float(t))
        angle = 2.0 * np.pi * phase
        return np.asarray([np.sin(angle), np.cos(angle)], dtype=np.float32)

    def residual_reward(
        self,
        *,
        info: dict[str, Any],
        residual: np.ndarray,
        residual_delta: np.ndarray,
        residual_accel: np.ndarray,
        final_action: np.ndarray,
        final_delta: np.ndarray,
        final_accel: np.ndarray,
        terminated: bool,
    ) -> float:
        x_distance = float(info.get("x_distance", 0.0))
        y_distance = float(info.get("y_distance", 0.0))
        planar_drift = float(np.hypot(x_distance, y_distance))
        self.drift_values.append(planar_drift)
        drift_array = np.asarray(self.drift_values, dtype=np.float32)
        mean_drift = float(np.mean(drift_array))
        max_drift = float(np.max(drift_array))
        rms_drift = float(np.sqrt(np.mean(np.square(drift_array))))

        yaw_weight = abs(float(self.yaw_command))
        target_rate = self.target_yaw_rate * yaw_weight
        yaw_progress = float(info.get("yaw_progress", 0.0))
        useful_yaw = max(0.0, yaw_progress)
        wrong_yaw = max(0.0, -yaw_progress)
        rewarded_yaw = min(useful_yaw, 1.10 * target_rate)
        underspeed = max(0.0, 0.65 * target_rate - useful_yaw)
        no_turn = max(0.0, 0.20 * target_rate - useful_yaw)
        overspeed = max(0.0, useful_yaw - 1.35 * target_rate)

        drift_excess = max(0.0, planar_drift - self.final_drift_limit_m)
        mean_excess = max(0.0, mean_drift - self.mean_drift_limit_m)
        max_excess = max(0.0, max_drift - self.max_drift_limit_m)

        residual_delta_abs = float(np.mean(np.abs(residual_delta)))
        residual_accel_abs = float(np.mean(np.abs(residual_accel)))
        residual_abs = float(np.mean(np.abs(residual)))
        final_delta_abs = float(np.mean(np.abs(final_delta)))
        final_accel_abs = float(np.mean(np.abs(final_accel)))
        final_abs = float(np.mean(np.abs(final_action)))
        planar_speed = abs(float(info.get("planar_speed", 0.0)))
        roll_pitch_rate = abs(float(info.get("roll_pitch_rate", 0.0)))
        support_deficit = float(info.get("support_deficit", 0.0))
        contact_foot_speed = float(info.get("mean_contact_foot_speed", 0.0))
        height_warning_deficit = float(info.get("body_height_warning_deficit", 0.0))
        height_deficit = float(info.get("body_height_deficit", 0.0))

        reward = (
            8.0 * rewarded_yaw
            - 12.0 * wrong_yaw
            - 4.0 * underspeed
            - 5.0 * no_turn
            - 2.5 * overspeed
            - 24.0 * planar_drift
            - 34.0 * mean_drift
            - 46.0 * max_drift
            - 80.0 * drift_excess
            - 120.0 * mean_excess
            - 160.0 * max_excess
            - 4.0 * planar_speed
            - 0.08 * roll_pitch_rate
            - 0.60 * support_deficit
            - 0.90 * contact_foot_speed
            - 24.0 * height_warning_deficit
            - 60.0 * height_deficit
            - self.residual_delta_weight * residual_delta_abs
            - self.residual_accel_weight * residual_accel_abs
            - self.final_delta_weight * final_delta_abs
            - self.final_accel_weight * final_accel_abs
            - 0.20 * residual_abs
            - 0.05 * final_abs
        )
        if terminated:
            reward -= 20.0 + 0.08 * max(0, self.env.max_steps - self.env.steps)

        info["v2_residual_planar_drift"] = planar_drift
        info["v2_residual_mean_planar_drift"] = mean_drift
        info["v2_residual_max_planar_drift"] = max_drift
        info["v2_residual_rms_planar_drift"] = rms_drift
        info["v2_residual_drift_excess"] = drift_excess
        info["v2_residual_mean_drift_excess"] = mean_excess
        info["v2_residual_max_drift_excess"] = max_excess
        info["v2_residual_delta_abs"] = residual_delta_abs
        info["v2_residual_accel_abs"] = residual_accel_abs
        info["v2_residual_abs"] = residual_abs
        info["v2_residual_final_delta_abs"] = final_delta_abs
        info["v2_residual_final_accel_abs"] = final_accel_abs
        return float(reward)


def load_table(path: str | Path) -> dict[str, np.ndarray | float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    actions = np.asarray(payload.get("actions", []), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 3 or actions.shape[0] == 0:
        raise ValueError(f"{path} does not contain a non-empty Nx3 actions table.")
    hz = float(payload.get("control_hz", 0.0))
    if not hz > 0.0:
        dt = float(payload.get("dt", 0.0))
        if not dt > 0.0:
            raise ValueError(f"{path} must define control_hz or dt.")
        hz = 1.0 / dt
    return {"actions": np.clip(actions, -1.0, 1.0), "hz": hz}


def load_base_source(
    *,
    table_path: str | Path,
    params_path: str | Path | None,
) -> dict[str, Any]:
    if params_path is not None:
        primitive, params = load_params(Path(params_path))
        return {
            "kind": "params",
            "path": str(params_path),
            "primitive": primitive,
            "params": np.asarray(params, dtype=np.float64),
            "frequency_hz": float(params[0]),
        }
    table = load_table(table_path)
    return {
        "kind": "table",
        "path": str(table_path),
        **table,
    }


def sample_base_source(source: dict[str, Any], t: float) -> np.ndarray:
    if source["kind"] == "params":
        return gait_action(
            np.asarray(source["params"], dtype=np.float64),
            float(t),
            use_heading_correction=False,
        )
    return sample_table(source, float(t))


def base_source_phase(source: dict[str, Any], t: float) -> float:
    if source["kind"] == "params":
        frequency = max(1.0e-6, float(source["frequency_hz"]))
        return float(t * frequency) % 1.0
    actions = np.asarray(source["actions"], dtype=np.float32)
    hz = float(source["hz"])
    if actions.shape[0] == 0 or hz <= 0.0:
        return 0.0
    return ((float(t) * hz) / actions.shape[0]) % 1.0


def sample_table(table: dict[str, np.ndarray | float], t: float) -> np.ndarray:
    actions = np.asarray(table["actions"], dtype=np.float32)
    hz = float(table["hz"])
    duration = actions.shape[0] / hz
    phase = float(t) % duration
    index = phase * hz
    low_index = int(np.floor(index + 1.0e-9))
    low = low_index % actions.shape[0]
    high = (low + 1) % actions.shape[0]
    alpha = float(np.clip(index - low_index, 0.0, 1.0))
    return ((1.0 - alpha) * actions[low] + alpha * actions[high]).astype(np.float32)


def residual_env_metadata(env: BramV2YawResidualEnv) -> dict[str, Any]:
    return {
        "env_command_mode": YAW_RESIDUAL_COMMAND_MODE,
        "obs_dim": OBS_DIM + 8,
        "left_table": env.left_table_path,
        "right_table": env.right_table_path,
        "left_params": env.left_params_path,
        "right_params": env.right_params_path,
        "left_base_kind": str(env.left_base["kind"]),
        "right_base_kind": str(env.right_base["kind"]),
        "residual_limit": env.residual_limit,
        "target_yaw_rate": env.target_yaw_rate,
        "final_drift_limit_m": env.final_drift_limit_m,
        "mean_drift_limit_m": env.mean_drift_limit_m,
        "max_drift_limit_m": env.max_drift_limit_m,
        "slew_limit": env.slew_limit,
    }
