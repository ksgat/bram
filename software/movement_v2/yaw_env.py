from __future__ import annotations

from collections import deque
from pathlib import Path
import sys
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

GAIT_DISCOVERY_DIR = Path(__file__).resolve().parents[1] / "gait_discovery"
if str(GAIT_DISCOVERY_DIR) not in sys.path:
    sys.path.insert(0, str(GAIT_DISCOVERY_DIR))

from bram_env import BramTripodEnv


V2_PRIMITIVE_COMMAND_MODE = "movement_v2_primitives_quat4imu_action6_cmd1"
YAW_ENV_COMMAND_MODE = V2_PRIMITIVE_COMMAND_MODE
IMU_HISTORY_FRAMES = 4
ACTION_HISTORY_FRAMES = 6
IMU_FRAME_DIM = 4
SERVO_COMMAND_DIM = 3
OBS_DIM = IMU_HISTORY_FRAMES * IMU_FRAME_DIM + ACTION_HISTORY_FRAMES * SERVO_COMMAND_DIM + 1
PAPER_REWARD_WEIGHTS = {
    "progress": 30.0,
    "height": 20.0,
    "up": 5.0,
    "heading": 2.0,
    "alive": 1.0,
    "death": -1.0,
    "action": -2.0,
    "vel": -2.0,
}
HINGE_VELOCITY_NORMALIZER = (10.472 - 1.0) ** 2
ACTIVE_ALIVE_SCALE = 0.05
MIN_WALK_PROGRESS_RATE = 0.015
MIN_YAW_PROGRESS_RATE = 0.05
YAW_TARGET_RATE_PER_COMMAND = 0.24
TILT_FREE_RAD = 0.45
TILT_LIMIT_RAD = 0.80
EXCESS_TILT_WEIGHT = 10.0
TRANSLATION_FULL_CREDIT_CONE_RAD = np.deg2rad(45.0)
TRANSLATION_MAX_CREDIT_CONE_RAD = np.deg2rad(65.0)
MIN_WALK_USEFUL_SPEED = 0.025


class BramV2PrimitiveEnv(gym.Env):
    """Movement V2 primitive env with low-rate history observations.

    Observation layout:
    - last 4 IMU quaternion frames, wxyz
    - last 6 normalized servo command frames
    - yaw command scalar in [-1, 1], zero for forward/back specialists
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        xml_path: str | Path | None = None,
        frame_skip: int = 25,
        episode_seconds: float = 8.0,
        randomize_reset: bool = True,
        domain_randomization: bool = False,
        domain_randomization_strength: float = 0.15,
        primitive: str = "yaw",
        randomize_yaw_command: bool = True,
        forward_command: float = 1.0,
        yaw_command: float = 1.0,
        yaw_min_magnitude: float = 0.35,
        idle_probability: float = 0.0,
    ) -> None:
        super().__init__()
        if primitive not in ("forward", "backward", "yaw"):
            raise ValueError(f"unknown movement_v2 primitive: {primitive}")
        fixed_forward, fixed_yaw = primitive_command(
            primitive,
            forward_command=forward_command,
            yaw_command=yaw_command,
        )
        env_kwargs: dict[str, Any] = {
            "frame_skip": frame_skip,
            "episode_seconds": episode_seconds,
            "randomize_reset": randomize_reset,
            "domain_randomization": domain_randomization,
            "domain_randomization_strength": domain_randomization_strength,
            "randomize_command": False,
            "command_forward": fixed_forward,
            "command_yaw_rate": fixed_yaw,
        }
        if xml_path is not None:
            env_kwargs["xml_path"] = xml_path
        self.env = BramTripodEnv(**env_kwargs)
        self.primitive = primitive
        self.randomize_yaw_command = bool(randomize_yaw_command)
        self.fixed_forward_command = float(np.clip(fixed_forward, -1.0, 1.0))
        self.fixed_yaw_command = float(np.clip(fixed_yaw, -1.0, 1.0))
        self.yaw_min_magnitude = float(np.clip(yaw_min_magnitude, 0.0, 1.0))
        self.idle_probability = float(np.clip(idle_probability, 0.0, 1.0))
        self.forward_command = self.fixed_forward_command
        self.yaw_command = self.fixed_yaw_command

        self.imu_history: deque[np.ndarray] = deque(maxlen=IMU_HISTORY_FRAMES)
        self.action_history: deque[np.ndarray] = deque(maxlen=ACTION_HISTORY_FRAMES)
        self.previous_action = np.zeros(SERVO_COMMAND_DIM, dtype=np.float32)
        self.previous_action_delta = np.zeros(SERVO_COMMAND_DIM, dtype=np.float32)
        self.translation_path_length = 0.0
        self.translation_broad_distance = 0.0
        self.yaw_drift_integral = 0.0
        self.yaw_max_drift = 0.0

        self.action_space = self.env.action_space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBS_DIM,),
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

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        options = {} if options is None else dict(options)
        super().reset(seed=seed)
        self.forward_command, self.yaw_command = self._reset_command(options)
        obs, info = self.env.reset(
            seed=seed,
            options={
                **options,
                "forward_command": self.forward_command,
                "yaw_rate_command": self.yaw_command,
            },
        )
        del obs
        self.previous_action[:] = 0.0
        self.previous_action_delta[:] = 0.0
        self.translation_path_length = 0.0
        self.translation_broad_distance = 0.0
        self.yaw_drift_integral = 0.0
        self.yaw_max_drift = 0.0
        self._reset_history()
        info = dict(info)
        info["forward_command"] = self.forward_command
        info["yaw_rate_command"] = self.yaw_command
        info["env_command_mode"] = YAW_ENV_COMMAND_MODE
        return self._get_obs(), info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        previous = self.previous_action.copy()
        previous_delta = self.previous_action_delta.copy()
        action_delta = action - previous
        _, _, terminated, truncated, info = self.env.step(action)
        self.previous_action[:] = action
        self.previous_action_delta[:] = action_delta
        self._append_history(action)
        reward = self._primitive_reward(
            info,
            action,
            previous,
            previous_delta,
            terminated,
        )
        info = dict(info)
        info["v2_primitive_reward"] = reward
        info["v2_forward_command"] = self.forward_command
        info["v2_yaw_command"] = self.yaw_command
        info["env_command_mode"] = YAW_ENV_COMMAND_MODE
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self) -> None:
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    def _reset_command(self, options: dict[str, Any]) -> tuple[float, float]:
        if self.primitive in ("forward", "backward"):
            return self.fixed_forward_command, 0.0
        if "yaw_cmd" in options:
            return 0.0, float(np.clip(options["yaw_cmd"], -1.0, 1.0))
        if "yaw_rate_command" in options:
            return 0.0, float(np.clip(options["yaw_rate_command"], -1.0, 1.0))
        if "command_yaw_rate" in options:
            return 0.0, float(np.clip(options["command_yaw_rate"], -1.0, 1.0))
        if not self.randomize_yaw_command:
            return 0.0, self.fixed_yaw_command
        if float(self.np_random.uniform()) < self.idle_probability:
            return 0.0, 0.0
        magnitude = float(self.np_random.uniform(self.yaw_min_magnitude, 1.0))
        sign = -1.0 if float(self.np_random.uniform()) < 0.5 else 1.0
        return 0.0, sign * magnitude

    def _reset_history(self) -> None:
        self.imu_history.clear()
        self.action_history.clear()
        imu = self._imu_frame()
        zero_action = np.zeros(SERVO_COMMAND_DIM, dtype=np.float32)
        for _ in range(IMU_HISTORY_FRAMES):
            self.imu_history.append(imu.copy())
        for _ in range(ACTION_HISTORY_FRAMES):
            self.action_history.append(zero_action.copy())

    def _append_history(self, action: np.ndarray) -> None:
        self.imu_history.append(self._imu_frame())
        self.action_history.append(action.astype(np.float32).copy())

    def _imu_frame(self) -> np.ndarray:
        quat = np.asarray(self.env.data.qpos[3:7], dtype=np.float32)
        norm = float(np.linalg.norm(quat))
        if norm > 1e-6:
            quat = quat / norm
        return quat.astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(
            [
                np.concatenate(list(self.imu_history), axis=0),
                np.concatenate(list(self.action_history), axis=0),
                np.array([self.command_scalar()], dtype=np.float32),
            ]
        ).astype(np.float32)

    def command_scalar(self) -> float:
        if self.primitive == "yaw":
            return self.yaw_command
        return 0.0

    def _primitive_reward(
        self,
        info: dict[str, Any],
        action: np.ndarray,
        previous_action: np.ndarray,
        previous_action_delta: np.ndarray,
        terminated: bool,
    ) -> float:
        if self.primitive == "yaw":
            return self._yaw_reward(
                info,
                action,
                previous_action,
                previous_action_delta,
                terminated,
            )
        return self._walk_reward(info, action, previous_action, terminated)

    def _walk_reward(
        self,
        info: dict[str, Any],
        action: np.ndarray,
        previous_action: np.ndarray,
        terminated: bool,
    ) -> float:
        translation = self._translation_metrics(info)
        useful_speed = translation["useful_speed"]
        wrong_speed = translation["wrong_speed"]
        direction_miss_speed = translation["direction_miss_speed"]
        height = float(info.get("height", 0.0))
        upright = float(info.get("upright", 0.0))
        yaw_rate = abs(float(info.get("yaw_rate", 0.0)))
        action_delta = float(np.mean(np.square(action - previous_action)))
        hinge_velocity = normalized_hinge_velocity(self.env._hinge_velocity_cost())
        height_reward = min(1.0, max(0.0, height / 0.08))
        tilt_quality, excessive_tilt = tilt_quality_from_upright(upright)
        alive_reward = PAPER_REWARD_WEIGHTS["alive"] * (
            0.0 if terminated else ACTIVE_ALIVE_SCALE
        )
        stall_penalty = PAPER_REWARD_WEIGHTS["progress"] * max(
            0.0,
            MIN_WALK_USEFUL_SPEED * abs(self.forward_command) - useful_speed,
        )
        straightness_penalty = 2.0 * max(
            0.0,
            0.70 - translation["straightness"],
        )
        direction_penalty = 1.50 * max(
            0.0,
            translation["direction_error_rad"] - TRANSLATION_MAX_CREDIT_CONE_RAD,
        )
        reward = (
            PAPER_REWARD_WEIGHTS["progress"] * useful_speed
            # The paper uses positive height/up rewards. Centering them at zero
            # keeps those weights without making standing still the easy optimum.
            # BRAM needs moderate body rocking, so "up" is a corridor instead of
            # a penalty for every bit of tilt.
            + PAPER_REWARD_WEIGHTS["height"] * (height_reward - 1.0)
            + PAPER_REWARD_WEIGHTS["up"] * (tilt_quality - 1.0)
            + alive_reward
            - 25.0 * wrong_speed
            - stall_penalty
            - EXCESS_TILT_WEIGHT * excessive_tilt * excessive_tilt
            - 5.0 * direction_miss_speed
            - 1.25 * yaw_rate
            - straightness_penalty
            - direction_penalty
            + PAPER_REWARD_WEIGHTS["action"] * action_delta
            + PAPER_REWARD_WEIGHTS["vel"] * hinge_velocity
        )
        if terminated:
            reward += PAPER_REWARD_WEIGHTS["death"]
            reward -= 20.0 + 0.08 * max(0, self.env.max_steps - self.env.steps)
        return float(reward)

    def _translation_metrics(self, info: dict[str, Any]) -> dict[str, float]:
        command_sign = 1.0 if self.forward_command >= 0.0 else -1.0
        line_velocity = float(info.get("line_velocity", 0.0))
        lateral_velocity = float(info.get("cross_track_velocity", 0.0))
        forward_speed = command_sign * line_velocity
        planar_speed = float(np.hypot(forward_speed, lateral_velocity))
        if forward_speed > 1.0e-6:
            velocity_direction_error = abs(float(np.arctan2(lateral_velocity, forward_speed)))
        else:
            velocity_direction_error = np.pi
        direction_gate = translation_cone_gate(velocity_direction_error)
        useful_speed = max(0.0, planar_speed * direction_gate)
        wrong_speed = max(0.0, -forward_speed)
        direction_miss_speed = max(0.0, planar_speed - useful_speed)

        self.translation_path_length += planar_speed * self.dt
        self.translation_broad_distance += useful_speed * self.dt
        x_distance = float(info.get("x_distance", 0.0))
        y_distance = float(info.get("y_distance", 0.0))
        net_distance = float(np.hypot(x_distance, y_distance))
        direction_error = translation_displacement_error(
            x_distance,
            y_distance,
            command_sign,
        )
        net_direction_gate = translation_cone_gate(direction_error)
        straightness = (
            min(1.0, net_distance / self.translation_path_length)
            if self.translation_path_length > 1.0e-6
            else 1.0
        )
        path_waste = max(0.0, self.translation_path_length - net_distance)
        path_angle = float(np.arctan2(y_distance, x_distance)) if net_distance > 1.0e-6 else 0.0
        primary_distance = net_distance * net_direction_gate
        metrics = {
            "useful_speed": useful_speed,
            "wrong_speed": wrong_speed,
            "direction_miss_speed": direction_miss_speed,
            "velocity_direction_error_rad": velocity_direction_error,
            "velocity_direction_gate": direction_gate,
            "net_distance": net_distance,
            "path_length": self.translation_path_length,
            "path_waste": path_waste,
            "straightness": straightness,
            "direction_error_rad": direction_error,
            "direction_gate": net_direction_gate,
            "path_angle_rad": path_angle,
            "primary_distance": primary_distance,
            "broad_distance": self.translation_broad_distance,
        }
        for key, value in metrics.items():
            info[f"v2_translation_{key}"] = float(value)
        return metrics

    def _yaw_reward(
        self,
        info: dict[str, Any],
        action: np.ndarray,
        previous_action: np.ndarray,
        previous_action_delta: np.ndarray,
        terminated: bool,
    ) -> float:
        yaw_weight = abs(self.yaw_command)
        yaw_progress = float(info.get("yaw_progress", 0.0))
        planar_speed = float(info.get("planar_speed", 0.0))
        x_distance = float(info.get("x_distance", 0.0))
        y_distance = float(info.get("y_distance", 0.0))
        planar_drift = float(np.hypot(x_distance, y_distance))
        self.yaw_drift_integral += planar_drift * self.dt
        self.yaw_max_drift = max(self.yaw_max_drift, planar_drift)

        action_step = action - previous_action
        action_accel_step = action_step - previous_action_delta
        action_delta = float(np.mean(np.square(action_step)))
        action_delta_abs = float(np.mean(np.abs(action_step)))
        action_accel_abs = float(np.mean(np.abs(action_accel_step)))
        abs_action = float(np.mean(np.abs(action)))
        hinge_velocity = normalized_hinge_velocity(self.env._hinge_velocity_cost())
        height = float(info.get("height", 0.0))
        upright = float(info.get("upright", 0.0))
        height_reward = min(1.0, max(0.0, height / 0.08))
        tilt_quality, excessive_tilt = tilt_quality_from_upright(upright)
        yaw_rate_error = abs(float(info.get("yaw_error", 0.0)))
        roll_pitch_rate = abs(float(info.get("roll_pitch_rate", 0.0)))
        support_deficit = float(info.get("support_deficit", 0.0))
        contact_foot_speed = float(info.get("mean_contact_foot_speed", 0.0))
        height_warning_deficit = float(info.get("body_height_warning_deficit", 0.0))
        height_deficit = float(info.get("body_height_deficit", 0.0))
        mean_planar_drift = self.yaw_drift_integral / max(
            self.dt,
            self.env.steps * self.dt,
        )
        alive_reward = PAPER_REWARD_WEIGHTS["alive"] * (
            0.0 if terminated else ACTIVE_ALIVE_SCALE
        )
        info["v2_yaw_mean_planar_drift"] = float(mean_planar_drift)
        info["v2_yaw_max_planar_drift"] = float(self.yaw_max_drift)
        info["v2_yaw_action_delta_abs"] = float(action_delta_abs)
        info["v2_yaw_action_accel_abs"] = float(action_accel_abs)
        info["v2_yaw_abs_action"] = float(abs_action)

        if yaw_weight < 0.05:
            reward = (
                PAPER_REWARD_WEIGHTS["height"] * (height_reward - 1.0)
                + PAPER_REWARD_WEIGHTS["up"] * (tilt_quality - 1.0)
                + alive_reward
                - EXCESS_TILT_WEIGHT * excessive_tilt * excessive_tilt
                - 8.0 * planar_speed
                - 3.0 * abs(float(info.get("yaw_rate", 0.0)))
                + PAPER_REWARD_WEIGHTS["action"] * action_delta
                + PAPER_REWARD_WEIGHTS["vel"] * hinge_velocity
                - 2.0 * action_delta_abs
                - 3.0 * action_accel_abs
                - 0.35 * abs_action
            )
        else:
            target_yaw_rate = YAW_TARGET_RATE_PER_COMMAND * yaw_weight
            useful_yaw = max(0.0, yaw_progress)
            wrong_yaw = max(0.0, -yaw_progress)
            rewarded_yaw = min(useful_yaw, 1.15 * target_yaw_rate)
            yaw_rate_miss = abs(yaw_progress - target_yaw_rate)
            yaw_alignment = 1.0 if yaw_progress > 0.0 else 0.0
            stall_penalty = PAPER_REWARD_WEIGHTS["progress"] * max(
                0.0,
                max(MIN_YAW_PROGRESS_RATE, 0.45 * target_yaw_rate) - useful_yaw,
            )
            overspin_penalty = PAPER_REWARD_WEIGHTS["heading"] * max(
                0.0,
                useful_yaw - 1.35 * target_yaw_rate,
            )
            reward = (
                PAPER_REWARD_WEIGHTS["progress"] * rewarded_yaw
                + PAPER_REWARD_WEIGHTS["height"] * (height_reward - 1.0)
                + PAPER_REWARD_WEIGHTS["up"] * (tilt_quality - 1.0)
                + PAPER_REWARD_WEIGHTS["heading"] * (yaw_alignment - 1.0)
                + alive_reward
                - 36.0 * wrong_yaw
                - stall_penalty
                - overspin_penalty
                - EXCESS_TILT_WEIGHT * excessive_tilt * excessive_tilt
                - 2.50 * yaw_rate_miss
                - 0.35 * yaw_rate_error
                - 9.0 * planar_speed
                - 10.0 * planar_drift
                - 3.0 * mean_planar_drift
                - 2.0 * self.yaw_max_drift
                - 0.08 * roll_pitch_rate
                - 0.60 * support_deficit
                - 0.90 * contact_foot_speed
                - 24.0 * height_warning_deficit
                - 60.0 * height_deficit
                + PAPER_REWARD_WEIGHTS["action"] * action_delta
                + PAPER_REWARD_WEIGHTS["vel"] * hinge_velocity
                - 2.75 * action_delta_abs
                - 4.00 * action_accel_abs
                - 0.35 * abs_action
            )

        if terminated:
            reward += PAPER_REWARD_WEIGHTS["death"]
            reward -= 20.0 + 0.08 * max(0, self.env.max_steps - self.env.steps)
        return float(reward)


BramYawPrimitiveEnv = BramV2PrimitiveEnv


def primitive_command(
    primitive: str,
    *,
    forward_command: float,
    yaw_command: float,
) -> tuple[float, float]:
    if primitive == "forward":
        return abs(float(forward_command)), 0.0
    if primitive == "backward":
        return -abs(float(forward_command)), 0.0
    if primitive == "yaw":
        return 0.0, float(yaw_command)
    raise ValueError(f"unknown movement_v2 primitive: {primitive}")


def normalized_hinge_velocity(raw_mean_square_velocity: float) -> float:
    return float(raw_mean_square_velocity) / HINGE_VELOCITY_NORMALIZER


def tilt_quality_from_upright(upright: float) -> tuple[float, float]:
    tilt_rad = float(np.arccos(np.clip(upright, -1.0, 1.0)))
    if tilt_rad <= TILT_FREE_RAD:
        return 1.0, 0.0
    excessive_tilt = max(0.0, tilt_rad - TILT_LIMIT_RAD)
    span = max(1.0e-6, TILT_LIMIT_RAD - TILT_FREE_RAD)
    quality = 1.0 - smoothstep((tilt_rad - TILT_FREE_RAD) / span)
    return float(np.clip(quality, 0.0, 1.0)), excessive_tilt


def translation_cone_gate(direction_error_rad: float) -> float:
    direction_error_rad = abs(float(direction_error_rad))
    if direction_error_rad <= TRANSLATION_FULL_CREDIT_CONE_RAD:
        return 1.0
    if direction_error_rad >= TRANSLATION_MAX_CREDIT_CONE_RAD:
        return 0.0
    alpha = (
        (direction_error_rad - TRANSLATION_FULL_CREDIT_CONE_RAD)
        / (TRANSLATION_MAX_CREDIT_CONE_RAD - TRANSLATION_FULL_CREDIT_CONE_RAD)
    )
    return 1.0 - smoothstep(alpha)


def translation_displacement_error(
    x_distance: float,
    y_distance: float,
    command_sign: float,
) -> float:
    aligned = float(command_sign) * float(x_distance)
    lateral = float(y_distance)
    if abs(aligned) < 1.0e-8 and abs(lateral) < 1.0e-8:
        return 0.0
    if aligned <= 0.0:
        return np.pi
    return abs(float(np.arctan2(lateral, aligned)))


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)
