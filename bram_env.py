from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

DEFAULT_XML_PATH = Path(__file__).with_name("bram.xml")
STANDARD_GRAVITY = 9.81
KG_CM_TO_NM = 0.0980665
SERVO_TORQUE_SPEC = (
    (5.0, 35.0 * KG_CM_TO_NM),
    (7.4, 46.0 * KG_CM_TO_NM),
    (8.4, 51.0 * KG_CM_TO_NM),
)
SERVO_SPEED_SPEC = (
    (5.0, np.pi / 3.0 / 0.13),
    (7.4, np.pi / 3.0 / 0.11),
    (8.4, np.pi / 3.0 / 0.10),
)
MAX_SERVO_TORQUE_NM = 2.5
RATED_SERVO_TORQUE_NM_AT_8V4 = SERVO_TORQUE_SPEC[-1][1]
MAX_SERVO_SPEED_RAD_S = SERVO_SPEED_SPEC[-1][1]


@dataclass
class DomainParams:
    active: bool = False
    initial_voltage: float = 7.8
    final_voltage: float = 7.4
    control_latency_steps: int = 0
    imu_latency_steps: int = 0
    floor_friction: float = 1.2
    foot_friction: float = 1.7


class BramTripodEnv(gym.Env):
    """Low-sensor MuJoCo env with sim-to-real randomization.

    The policy observes only signals that can exist on the robot:
    6DOF IMU-derived values plus recent servo commands. Rewards still use
    privileged simulator displacement because rewards are not deployed.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        xml_path: str | Path = DEFAULT_XML_PATH,
        frame_skip: int = 10,
        episode_seconds: float = 8.0,
        randomize_reset: bool = True,
        domain_randomization: bool | None = None,
        randomize_command: bool | None = None,
        command_angle: float | None = None,
    ) -> None:
        super().__init__()
        self.xml_path = Path(xml_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.dt = self.model.opt.timestep * self.frame_skip
        self.max_steps = max(1, int(episode_seconds / self.dt))
        self.randomize_reset = randomize_reset
        self.domain_randomization = (
            randomize_reset if domain_randomization is None else domain_randomization
        )
        self.randomize_command = (
            randomize_reset if randomize_command is None else randomize_command
        )
        self.fixed_command_angle = command_angle

        self.chassis_id = self._body_id("chassis")
        self.leg_body_ids = [
            self._body_id("leg_front"),
            self._body_id("leg_back_left"),
            self._body_id("leg_back_right"),
        ]
        self.floor_id = self._geom_id("floor")
        self.foot_geom_ids = [
            self._geom_id("front_rubber_tip"),
            self._geom_id("back_left_rubber_tip"),
            self._geom_id("back_right_rubber_tip"),
        ]
        self.leg_geom_ids = [
            self._geom_id("front_leg_tube"),
            self._geom_id("back_left_leg_tube"),
            self._geom_id("back_right_leg_tube"),
        ]
        self.hinge_joint_ids = [
            self._joint_id("hip_front"),
            self._joint_id("hip_back_left"),
            self._joint_id("hip_back_right"),
        ]
        self.hinge_qpos_addresses = [
            int(self.model.jnt_qposadr[joint_id]) for joint_id in self.hinge_joint_ids
        ]

        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self.ctrl_center = ((self.ctrl_low + self.ctrl_high) * 0.5).astype(np.float32)
        self.ctrl_half_range = ((self.ctrl_high - self.ctrl_low) * 0.5).astype(
            np.float32
        )

        self.base_gravity = self.model.opt.gravity.copy()
        self.base_body_mass = self.model.body_mass.copy()
        self.base_body_inertia = self.model.body_inertia.copy()
        self.base_body_pos = self.model.body_pos.copy()
        self.base_geom_friction = self.model.geom_friction.copy()
        self.base_geom_size = self.model.geom_size.copy()
        self.base_actuator_forcerange = self.model.actuator_forcerange.copy()
        self.base_actuator_gainprm = self.model.actuator_gainprm.copy()
        self.base_actuator_biasprm = self.model.actuator_biasprm.copy()
        self.base_dof_damping = self.model.dof_damping.copy()

        self.sensor_slices = self._sensor_slices()
        self.domain = DomainParams()
        self.steps = 0
        self.start_x = 0.0
        self.start_y = 0.0
        self.command_distance = 0.0
        self.cross_track_error = 0.0
        self.command_angle = 0.0
        self.command_direction_world = np.array([1.0, 0.0], dtype=np.float32)
        self.command_direction_body = np.array([1.0, 0.0], dtype=np.float32)

        self.policy_action = np.zeros(self.model.nu, dtype=np.float32)
        self.previous_policy_action = np.zeros(self.model.nu, dtype=np.float32)
        self.delayed_action = np.zeros(self.model.nu, dtype=np.float32)
        self.commanded_target_rad = self.ctrl_center.copy()
        self.applied_target_rad = self.ctrl_center.copy()
        self.tracking_error_rad = np.zeros(self.model.nu, dtype=np.float32)
        self.action_queue: deque[np.ndarray] = deque()

        self.servo_zero_offset = np.zeros(self.model.nu, dtype=np.float32)
        self.servo_gain = np.ones(self.model.nu, dtype=np.float32)
        self.servo_strength = np.ones(self.model.nu, dtype=np.float32)
        self.servo_speed = np.full(
            self.model.nu, MAX_SERVO_SPEED_RAD_S, dtype=np.float32
        )
        self.servo_deadband = np.zeros(self.model.nu, dtype=np.float32)
        self.servo_quantization = np.zeros(self.model.nu, dtype=np.float32)
        self.servo_time_constant = np.full(self.model.nu, 0.02, dtype=np.float32)

        self.gyro_bias = np.zeros(3, dtype=np.float32)
        self.accel_bias_g = np.zeros(3, dtype=np.float32)
        self.gyro_noise_std = 0.0
        self.accel_noise_std_g = 0.0
        self.gravity_noise_std = 0.0
        self.gyro_bias_walk_std = 0.0
        self.accel_bias_walk_std_g = 0.0
        self.gravity_filter_alpha = 0.08
        self.gravity_estimate = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.measured_gyro = np.zeros(3, dtype=np.float32)
        self.measured_accel_g = np.zeros(3, dtype=np.float32)
        self.measured_gravity = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.imu_buffer: deque[tuple[np.ndarray, np.ndarray, np.ndarray]] = deque()

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.model.nu,),
            dtype=np.float32,
        )
        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._restore_model_parameters()

        options = {} if options is None else options
        randomize = (
            self.randomize_reset
            if not options
            else options.get("randomize", self.randomize_reset)
        )
        use_domain_randomization = self.domain_randomization and randomize
        self._sample_domain(use_domain_randomization)
        self._apply_domain_to_model()
        self._reset_command(options)

        mujoco.mj_resetData(self.model, self.data)
        self._reset_servo_state()
        self._set_initial_pose(randomize)
        if randomize:
            self._randomize_joint_angles()

        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.start_x = self._x_position()
        self.start_y = self._y_position()
        self.command_distance = 0.0
        self.cross_track_error = 0.0
        self._update_command_observation()
        self._reset_imu_state()
        return self._get_obs(), self._reset_info()

    def _reset_command(self, options: dict[str, Any]) -> None:
        if "command_direction" in options:
            direction = np.asarray(options["command_direction"], dtype=np.float32)[:2]
            direction = normalize(direction)
            if float(np.linalg.norm(direction)) < 1e-6:
                direction = np.array([1.0, 0.0], dtype=np.float32)
            angle = float(np.arctan2(direction[1], direction[0]))
        elif "command_angle" in options:
            angle = float(options["command_angle"])
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        elif self.fixed_command_angle is not None:
            angle = float(self.fixed_command_angle)
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        elif self.randomize_command:
            angle = float(self.np_random.uniform(-np.pi, np.pi))
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        else:
            angle = 0.0
            direction = np.array([1.0, 0.0], dtype=np.float32)

        self.command_angle = wrap_angle(angle)
        self.command_direction_world[:] = normalize(direction).astype(np.float32)
        self.command_direction_body[:] = self.command_direction_world

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.previous_policy_action[:] = self.policy_action
        self.policy_action[:] = action
        self.delayed_action[:] = self._latency_action(action)

        self._update_servo_targets()
        self._apply_voltage_sag()

        x_before = self._x_position()
        y_before = self._y_position()
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        x_after = self._x_position()
        y_after = self._y_position()

        self.steps += 1
        self._update_imu_state()
        self._update_command_observation()
        obs = self._get_obs()

        forward_velocity = (x_after - x_before) / self.dt
        lateral_velocity = (y_after - y_before) / self.dt
        world_velocity = np.array([forward_velocity, lateral_velocity], dtype=np.float32)
        command_velocity = float(np.dot(world_velocity, self.command_direction_world))
        off_axis_velocity = float(
            -self.command_direction_world[1] * world_velocity[0]
            + self.command_direction_world[0] * world_velocity[1]
        )
        displacement_world = np.array(
            [x_after - self.start_x, y_after - self.start_y], dtype=np.float32
        )
        self.command_distance = float(
            np.dot(displacement_world, self.command_direction_world)
        )
        self.cross_track_error = float(
            -self.command_direction_world[1] * displacement_world[0]
            + self.command_direction_world[0] * displacement_world[1]
        )
        height = float(self.data.xpos[self.chassis_id, 2])
        upright = self._upright()
        action_delta = action - self.previous_policy_action

        forward_reward = 5.0 * float(np.clip(command_velocity, -1.5, 2.5))
        alive_reward = 0.005 * max(0.0, upright)
        backward_penalty = 2.0 * max(0.0, -command_velocity)
        upright_penalty = 0.06 * max(0.0, 0.35 - upright)
        sideways_penalty = 0.08 * abs(off_axis_velocity)
        line_penalty = 0.35 * abs(self.cross_track_error)
        ctrl_cost = 0.006 * float(np.mean(np.square(action)))
        smoothness_cost = 0.008 * float(np.mean(np.square(action_delta)))
        reward = (
            forward_reward
            + alive_reward
            - backward_penalty
            - upright_penalty
            - sideways_penalty
            - line_penalty
            - ctrl_cost
            - smoothness_cost
        )

        terminated = bool(
            height < 0.025 or upright < 0.10 or not np.isfinite(obs).all()
        )
        truncated = self.steps >= self.max_steps
        info = {
            "x_position": x_after,
            "x_distance": x_after - self.start_x,
            "y_distance": y_after - self.start_y,
            "command_distance": self.command_distance,
            "cross_track_error": self.cross_track_error,
            "command_angle": self.command_angle,
            "command_x_world": float(self.command_direction_world[0]),
            "command_y_world": float(self.command_direction_world[1]),
            "command_x_body": float(self.command_direction_body[0]),
            "command_y_body": float(self.command_direction_body[1]),
            "forward_velocity": forward_velocity,
            "lateral_velocity": lateral_velocity,
            "command_velocity": command_velocity,
            "off_axis_velocity": off_axis_velocity,
            "height": height,
            "upright": upright,
            "voltage": self._current_voltage(),
            "servo_torque_limit_nm": float(np.mean(self.model.actuator_forcerange[:, 1])),
            "floor_friction": self.domain.floor_friction,
            "foot_friction": self.domain.foot_friction,
            "control_latency_steps": self.domain.control_latency_steps,
            "imu_latency_steps": self.domain.imu_latency_steps,
            "forward_reward": forward_reward,
            "alive_reward": alive_reward,
            "backward_penalty": backward_penalty,
            "upright_penalty": upright_penalty,
            "sideways_penalty": sideways_penalty,
            "line_penalty": line_penalty,
            "ctrl_cost": ctrl_cost,
            "smoothness_cost": smoothness_cost,
        }
        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        command_delta = self.policy_action - self.previous_policy_action
        return np.concatenate(
            [
                self.measured_gyro,
                self.measured_accel_g,
                self.measured_gravity,
                self.command_direction_body,
                self.policy_action,
                command_delta,
            ]
        ).astype(np.float32)

    def _sample_domain(self, active: bool) -> None:
        self.domain = DomainParams(active=active)
        self.servo_zero_offset[:] = 0
        self.servo_gain[:] = 1
        self.servo_strength[:] = 1
        self.servo_speed[:] = MAX_SERVO_SPEED_RAD_S
        self.servo_deadband[:] = 0
        self.servo_quantization[:] = 0
        self.servo_time_constant[:] = 0.02
        self.gyro_bias[:] = 0
        self.accel_bias_g[:] = 0
        self.gyro_noise_std = 0.0
        self.accel_noise_std_g = 0.0
        self.gravity_noise_std = 0.0
        self.gyro_bias_walk_std = 0.0
        self.accel_bias_walk_std_g = 0.0
        self.gravity_filter_alpha = 0.15

        if not active:
            return

        self.domain.initial_voltage = float(self.np_random.uniform(7.2, 8.4))
        drop = float(self.np_random.uniform(0.05, 0.75))
        self.domain.final_voltage = max(6.2, self.domain.initial_voltage - drop)
        self.domain.control_latency_steps = int(self.np_random.integers(0, 4))
        self.domain.imu_latency_steps = int(self.np_random.integers(0, 4))
        self.domain.floor_friction = float(self.np_random.uniform(0.55, 2.1))
        self.domain.foot_friction = float(self.np_random.uniform(0.75, 2.4))

        self.servo_zero_offset[:] = self.np_random.normal(0.0, 0.035, self.model.nu)
        self.servo_gain[:] = self.np_random.uniform(0.92, 1.08, self.model.nu)
        self.servo_strength[:] = self.np_random.uniform(0.50, 1.00, self.model.nu)
        self.servo_speed[:] = MAX_SERVO_SPEED_RAD_S * self.np_random.uniform(
            0.65, 1.00, self.model.nu
        )
        self.servo_deadband[:] = self.np_random.uniform(0.003, 0.020, self.model.nu)
        self.servo_quantization[:] = self.np_random.uniform(
            0.0008, 0.006, self.model.nu
        )
        self.servo_time_constant[:] = self.np_random.uniform(
            0.018, 0.075, self.model.nu
        )

        self.gyro_bias[:] = self.np_random.normal(0.0, 0.035, 3)
        self.accel_bias_g[:] = self.np_random.normal(0.0, 0.025, 3)
        self.gyro_noise_std = float(self.np_random.uniform(0.006, 0.035))
        self.accel_noise_std_g = float(self.np_random.uniform(0.010, 0.055))
        self.gravity_noise_std = float(self.np_random.uniform(0.006, 0.030))
        self.gyro_bias_walk_std = float(self.np_random.uniform(0.0002, 0.0015))
        self.accel_bias_walk_std_g = float(self.np_random.uniform(0.0001, 0.0008))
        self.gravity_filter_alpha = float(self.np_random.uniform(0.035, 0.12))

    def _apply_domain_to_model(self) -> None:
        if not self.domain.active:
            return

        floor_torsional = float(self.np_random.uniform(0.01, 0.12))
        floor_rolling = float(self.np_random.uniform(0.0005, 0.02))
        self.model.geom_friction[self.floor_id] = [
            self.domain.floor_friction,
            floor_torsional,
            floor_rolling,
        ]
        for geom_id in self.foot_geom_ids:
            torsional = float(self.np_random.uniform(0.03, 0.16))
            rolling = float(self.np_random.uniform(0.003, 0.035))
            self.model.geom_friction[geom_id] = [
                self.domain.foot_friction,
                torsional,
                rolling,
            ]
            self.model.geom_size[geom_id, 0] *= float(
                self.np_random.uniform(0.80, 1.25)
            )
        for geom_id in self.leg_geom_ids:
            self.model.geom_friction[geom_id] = [
                float(self.np_random.uniform(0.7, 1.8)),
                float(self.np_random.uniform(0.015, 0.08)),
                float(self.np_random.uniform(0.002, 0.018)),
            ]

        slope = float(self.np_random.uniform(0.0, np.deg2rad(5.0)))
        heading = float(self.np_random.uniform(-np.pi, np.pi))
        self.model.opt.gravity[:] = STANDARD_GRAVITY * np.array(
            [
                np.sin(slope) * np.cos(heading),
                np.sin(slope) * np.sin(heading),
                -np.cos(slope),
            ]
        )

        chassis_scale = float(self.np_random.uniform(0.85, 1.22))
        self._scale_body_mass(self.chassis_id, chassis_scale)
        for body_id in self.leg_body_ids:
            self._scale_body_mass(body_id, float(self.np_random.uniform(0.80, 1.35)))
            self.model.body_pos[body_id, :2] += self.np_random.normal(0.0, 0.0018, 2)
            self.model.body_pos[body_id, 2] += float(self.np_random.normal(0.0, 0.0008))

        for joint_id in self.hinge_joint_ids:
            dof_addr = int(self.model.jnt_dofadr[joint_id])
            self.model.dof_damping[dof_addr] *= float(self.np_random.uniform(0.6, 1.7))

    def _restore_model_parameters(self) -> None:
        self.model.opt.gravity[:] = self.base_gravity
        self.model.body_mass[:] = self.base_body_mass
        self.model.body_inertia[:] = self.base_body_inertia
        self.model.body_pos[:] = self.base_body_pos
        self.model.geom_friction[:] = self.base_geom_friction
        self.model.geom_size[:] = self.base_geom_size
        self.model.actuator_forcerange[:] = self.base_actuator_forcerange
        self.model.actuator_gainprm[:] = self.base_actuator_gainprm
        self.model.actuator_biasprm[:] = self.base_actuator_biasprm
        self.model.dof_damping[:] = self.base_dof_damping

    def _reset_servo_state(self) -> None:
        self.policy_action[:] = 0
        self.previous_policy_action[:] = 0
        self.delayed_action[:] = 0
        self.commanded_target_rad[:] = self.ctrl_center
        self.applied_target_rad[:] = self.ctrl_center
        self.tracking_error_rad[:] = 0
        self.data.ctrl[:] = self.ctrl_center
        self.action_queue.clear()

    def _set_initial_pose(self, randomize: bool) -> None:
        if not randomize:
            return
        self.data.qpos[0:3] += self.np_random.normal(
            [0.0, 0.0, 0.0], [0.004, 0.004, 0.002]
        )
        roll = float(self.np_random.uniform(-0.04, 0.04))
        pitch = float(self.np_random.uniform(-0.04, 0.04))
        yaw = float(self.np_random.uniform(-0.18, 0.18))
        self.data.qpos[3:7] = quat_from_euler(roll, pitch, yaw)

    def _randomize_joint_angles(self) -> None:
        for qpos_addr in self.hinge_qpos_addresses:
            self.data.qpos[qpos_addr] = self.np_random.uniform(-0.05, 0.05)

    def _latency_action(self, action: np.ndarray) -> np.ndarray:
        self.action_queue.append(action.copy())
        if len(self.action_queue) > self.domain.control_latency_steps:
            return self.action_queue.popleft()
        return np.zeros_like(action)

    def _update_servo_targets(self) -> None:
        desired = self._scale_action(self.delayed_action)
        desired = self.ctrl_center + self.servo_gain * (desired - self.ctrl_center)
        desired = desired + self.servo_zero_offset
        desired = np.clip(desired, self.ctrl_low, self.ctrl_high)

        close = np.abs(desired - self.commanded_target_rad) < self.servo_deadband
        desired = np.where(close, self.commanded_target_rad, desired)
        quantized = self.servo_quantization > 0
        desired[quantized] = (
            np.round(desired[quantized] / self.servo_quantization[quantized])
            * self.servo_quantization[quantized]
        )
        self.commanded_target_rad[:] = np.clip(desired, self.ctrl_low, self.ctrl_high)

        alpha = self.dt / (self.servo_time_constant + self.dt)
        lagged = self.applied_target_rad + alpha * (
            self.commanded_target_rad - self.applied_target_rad
        )
        max_delta = self.servo_speed * self._voltage_speed_scale() * self.dt
        self.applied_target_rad += np.clip(
            lagged - self.applied_target_rad,
            -max_delta,
            max_delta,
        )
        self.tracking_error_rad = (
            0.92 * self.tracking_error_rad
            + self.np_random.normal(
                0.0, 0.004 if self.domain.active else 0.0, self.model.nu
            )
        ).astype(np.float32)
        self.data.ctrl[:] = np.clip(
            self.applied_target_rad + self.tracking_error_rad,
            self.ctrl_low,
            self.ctrl_high,
        )

    def _apply_voltage_sag(self) -> None:
        force_limit = self._servo_torque_limit() * self.servo_strength
        self.model.actuator_forcerange[:, 0] = -force_limit
        self.model.actuator_forcerange[:, 1] = force_limit

    def _current_voltage(self) -> float:
        if self.max_steps <= 1:
            return self.domain.final_voltage
        progress = min(1.0, self.steps / (self.max_steps - 1))
        return self.domain.initial_voltage + progress * (
            self.domain.final_voltage - self.domain.initial_voltage
        )

    def _voltage_strength_scale(self) -> float:
        return self._servo_torque_limit() / MAX_SERVO_TORQUE_NM

    def _voltage_speed_scale(self) -> float:
        voltage = self._current_voltage()
        return self._interp_servo_spec(voltage, SERVO_SPEED_SPEC) / MAX_SERVO_SPEED_RAD_S

    def _servo_torque_limit(self) -> float:
        voltage = self._current_voltage()
        rated_torque = self._interp_servo_spec(voltage, SERVO_TORQUE_SPEC)
        weak_scale = MAX_SERVO_TORQUE_NM / RATED_SERVO_TORQUE_NM_AT_8V4
        return rated_torque * weak_scale

    def _interp_servo_spec(
        self, voltage: float, spec: tuple[tuple[float, float], ...]
    ) -> float:
        voltages = [point[0] for point in spec]
        values = [point[1] for point in spec]
        return float(np.interp(voltage, voltages, values))

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        return self.ctrl_center + self.ctrl_half_range * action

    def _reset_imu_state(self) -> None:
        self.gravity_estimate = self._true_gravity_body().astype(np.float32)
        sample = self._make_imu_sample(reset=True)
        self.imu_buffer.clear()
        for _ in range(self.domain.imu_latency_steps + 1):
            self.imu_buffer.append(tuple(value.copy() for value in sample))
        self.measured_gyro, self.measured_accel_g, self.measured_gravity = (
            value.copy() for value in self.imu_buffer[0]
        )

    def _update_imu_state(self) -> None:
        sample = self._make_imu_sample(reset=False)
        self.imu_buffer.append(tuple(value.copy() for value in sample))
        while len(self.imu_buffer) > self.domain.imu_latency_steps + 1:
            self.imu_buffer.popleft()
        self.measured_gyro, self.measured_accel_g, self.measured_gravity = (
            value.copy() for value in self.imu_buffer[0]
        )

    def _make_imu_sample(
        self, reset: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not reset:
            self.gyro_bias += self.np_random.normal(0.0, self.gyro_bias_walk_std, 3)
            self.accel_bias_g += self.np_random.normal(
                0.0, self.accel_bias_walk_std_g, 3
            )

        gyro = self._sensor("gyro", 3)
        accel_g = self._sensor("accel", 3) / STANDARD_GRAVITY
        gyro = (
            gyro + self.gyro_bias + self.np_random.normal(0.0, self.gyro_noise_std, 3)
        )
        accel_g = (
            accel_g
            + self.accel_bias_g
            + self.np_random.normal(0.0, self.accel_noise_std_g, 3)
        )

        true_gravity = self._true_gravity_body()
        accel_disturbance = np.clip(accel_g - true_gravity, -2.0, 2.0)
        gravity_measurement = true_gravity + 0.15 * accel_disturbance
        gravity_measurement += self.np_random.normal(0.0, self.gravity_noise_std, 3)
        gravity_measurement = normalize(gravity_measurement)
        self.gravity_estimate = normalize(
            (1.0 - self.gravity_filter_alpha) * self.gravity_estimate
            + self.gravity_filter_alpha * gravity_measurement
        ).astype(np.float32)
        return (
            gyro.astype(np.float32),
            accel_g.astype(np.float32),
            self.gravity_estimate.astype(np.float32),
        )

    def _sensor(self, name: str, width: int) -> np.ndarray:
        sensor_slice = self.sensor_slices.get(name)
        if sensor_slice is None:
            return np.zeros(width, dtype=np.float32)
        values = self.data.sensordata[sensor_slice]
        if values.size != width:
            padded = np.zeros(width, dtype=np.float32)
            padded[: min(width, values.size)] = values[:width]
            return padded
        return values.astype(np.float32)

    def _sensor_slices(self) -> dict[str, slice]:
        slices = {}
        for sensor_id in range(self.model.nsensor):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
            if name is None:
                continue
            start = int(self.model.sensor_adr[sensor_id])
            dim = int(self.model.sensor_dim[sensor_id])
            slices[name] = slice(start, start + dim)
        return slices

    def _true_gravity_body(self) -> np.ndarray:
        xmat = self.data.xmat[self.chassis_id].reshape(3, 3)
        return normalize(xmat.T @ self._gravity_up_world())

    def _gravity_up_world(self) -> np.ndarray:
        return normalize(-self.model.opt.gravity.copy())

    def _scale_body_mass(self, body_id: int, scale: float) -> None:
        self.model.body_mass[body_id] = self.base_body_mass[body_id] * scale
        self.model.body_inertia[body_id] = self.base_body_inertia[body_id] * scale

    def _upright(self) -> float:
        xmat = self.data.xmat[self.chassis_id].reshape(3, 3)
        body_up_world = xmat[:, 2]
        return float(np.dot(body_up_world, self._gravity_up_world()))

    def _x_position(self) -> float:
        return float(self.data.xpos[self.chassis_id, 0])

    def _y_position(self) -> float:
        return float(self.data.xpos[self.chassis_id, 1])

    def _world_velocity_to_body(self, x_velocity: float, y_velocity: float) -> np.ndarray:
        xmat = self.data.xmat[self.chassis_id].reshape(3, 3)
        body_x_world = xmat[:, 0].copy()
        body_y_world = xmat[:, 1].copy()
        body_x_world[2] = 0.0
        body_y_world[2] = 0.0
        body_x_world = normalize(body_x_world)
        body_y_world = normalize(body_y_world)
        world_velocity = np.array([x_velocity, y_velocity, 0.0], dtype=np.float64)
        return np.array(
            [
                np.dot(world_velocity, body_x_world),
                np.dot(world_velocity, body_y_world),
            ],
            dtype=np.float32,
        )

    def _update_command_observation(self) -> None:
        self.command_direction_body[:] = self._world_direction_to_body(
            self.command_direction_world
        )

    def _world_direction_to_body(self, direction_world: np.ndarray) -> np.ndarray:
        xmat = self.data.xmat[self.chassis_id].reshape(3, 3)
        body_x_world = xmat[:, 0].copy()
        body_y_world = xmat[:, 1].copy()
        body_x_world[2] = 0.0
        body_y_world[2] = 0.0
        body_x_world = normalize(body_x_world)
        body_y_world = normalize(body_y_world)
        direction = normalize(np.asarray(direction_world, dtype=np.float64)[:2])
        direction_3d = np.array([direction[0], direction[1], 0.0], dtype=np.float64)
        return np.array(
            [
                np.dot(direction_3d, body_x_world),
                np.dot(direction_3d, body_y_world),
            ],
            dtype=np.float32,
        )

    def _reset_info(self) -> dict[str, float | int | bool]:
        return {
            "domain_randomization": self.domain.active,
            "voltage": self._current_voltage(),
            "command_angle": self.command_angle,
            "command_x_world": float(self.command_direction_world[0]),
            "command_y_world": float(self.command_direction_world[1]),
            "command_x_body": float(self.command_direction_body[0]),
            "command_y_body": float(self.command_direction_body[1]),
            "floor_friction": self.domain.floor_friction,
            "foot_friction": self.domain.foot_friction,
            "control_latency_steps": self.domain.control_latency_steps,
            "imu_latency_steps": self.domain.imu_latency_steps,
        }

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Could not find body named {name!r}.")
        return int(body_id)

    def _geom_id(self, name: str) -> int:
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise ValueError(f"Could not find geom named {name!r}.")
        return int(geom_id)

    def _joint_id(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Could not find joint named {name!r}.")
        return int(joint_id)


def normalize(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm < 1e-8:
        return np.zeros_like(value)
    return value / norm


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )
