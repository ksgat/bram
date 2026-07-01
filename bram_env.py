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
MAX_SERVO_TORQUE_NM = 3.0
RATED_SERVO_TORQUE_NM_AT_8V4 = SERVO_TORQUE_SPEC[-1][1]
MAX_SERVO_SPEED_RAD_S = SERVO_SPEED_SPEC[-1][1]
MAX_FORWARD_COMMAND_MPS = 0.15
MAX_YAW_COMMAND_RAD_S = 1.2
GAIT_PHASE_HZ = 1.4
LINEAR_TRACKING_SIGMA_MPS = 0.070
YAW_TRACKING_SIGMA_RAD_S = 0.50
HEADING_TRACKING_SIGMA_RAD = 0.70
MIN_CHASSIS_CENTER_HEIGHT_M = 0.008
BATTERY_SIDE_DOWN_UPRIGHT_LIMIT = -0.25
BATTERY_SIDE_WARNING_UPRIGHT = -0.10
ENV_COMMAND_MODE = "forward_yaw_heading_v10"


@dataclass
class DomainParams:
    active: bool = False
    strength: float = 0.0
    initial_voltage: float = 7.8
    final_voltage: float = 7.4
    control_latency_steps: int = 0
    imu_latency_steps: int = 0
    floor_friction: float = 1.2
    foot_friction: float = 1.7


class BramTripodEnv(gym.Env):
    """Low-sensor MuJoCo env with sim-to-real randomization."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        xml_path: str | Path = DEFAULT_XML_PATH,
        frame_skip: int = 10,
        episode_seconds: float = 8.0,
        randomize_reset: bool = True,
        domain_randomization: bool | None = None,
        domain_randomization_strength: float = 0.45,
        randomize_command: bool | None = None,
        command_curriculum: bool = True,
        command_curriculum_level: float = 1.0,
        command_forward: float = 1.0,
        command_yaw_rate: float = 0.0,
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
        self.domain_randomization_strength = float(
            np.clip(domain_randomization_strength, 0.0, 1.0)
        )
        self.randomize_command = (
            False if randomize_command is None else randomize_command
        )
        self.command_curriculum = command_curriculum
        self.command_curriculum_level = float(
            np.clip(command_curriculum_level, 0.0, 1.0)
        )
        self.fixed_command_forward = float(command_forward)
        self.fixed_command_yaw_rate = float(command_yaw_rate)

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
        self.foot_geom_index = {
            geom_id: index for index, geom_id in enumerate(self.foot_geom_ids)
        }
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
        self.hinge_dof_addresses = [
            int(self.model.jnt_dofadr[joint_id]) for joint_id in self.hinge_joint_ids
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
        self.start_heading = 0.0
        self.desired_heading = 0.0
        self.heading_error = 0.0
        self.cross_track_error = 0.0
        self.command_distance = 0.0
        self.line_distance = 0.0
        self.yaw_distance = 0.0
        self.command_quality_ema = 0.0
        self.gait_phase = 0.0
        self.forward_command = 1.0
        self.yaw_rate_command = 0.0
        self.command = np.array([1.0, 0.0], dtype=np.float32)

        self.policy_action = np.zeros(self.model.nu, dtype=np.float32)
        self.previous_policy_action = np.zeros(self.model.nu, dtype=np.float32)
        self.previous_action_delta = np.zeros(self.model.nu, dtype=np.float32)
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
        self.start_heading = self._heading()
        self.desired_heading = self.start_heading
        self.heading_error = 0.0
        self.cross_track_error = 0.0
        self.command_distance = 0.0
        self.line_distance = 0.0
        self.yaw_distance = 0.0
        self.command_quality_ema = 0.0
        self.gait_phase = float(self.np_random.uniform()) if randomize else 0.0
        self._reset_imu_state()
        return self._get_obs(), self._reset_info()

    def _reset_command(self, options: dict[str, Any]) -> None:
        if self.randomize_command:
            forward, yaw_rate = self._sample_random_command()
        else:
            forward = self.fixed_command_forward
            yaw_rate = self.fixed_command_yaw_rate

        forward = float(
            options.get("forward_command", options.get("command_forward", forward))
        )
        yaw_rate = float(
            options.get("yaw_rate_command", options.get("command_yaw_rate", yaw_rate))
        )
        self.forward_command = float(np.clip(forward, -1.0, 1.0))
        self.yaw_rate_command = float(np.clip(yaw_rate, -1.0, 1.0))
        self.command[:] = [self.forward_command, self.yaw_rate_command]

    def set_command_curriculum_level(self, level: float) -> None:
        self.command_curriculum_level = float(np.clip(level, 0.0, 1.0))

    def _sample_random_command(self) -> tuple[float, float]:
        if not self.command_curriculum:
            return self._sample_full_command()

        level = self.command_curriculum_level
        mode = float(self.np_random.uniform())
        if level < 0.20:
            if mode < 0.10:
                return 0.0, 0.0
            return float(self.np_random.uniform(0.50, 1.0)), 0.0

        if level < 0.40:
            if mode < 0.08:
                return 0.0, 0.0
            if mode < 0.72:
                return float(self.np_random.uniform(0.45, 1.0)), 0.0
            return -float(self.np_random.uniform(0.35, 0.85)), 0.0

        if level < 0.62:
            if mode < 0.08:
                return 0.0, 0.0
            if mode < 0.66:
                return self._signed_command_magnitude(0.40, 1.0), 0.0
            return 0.0, self._signed_command_magnitude(0.35, 0.85)

        if level < 0.82:
            if mode < 0.06:
                return 0.0, 0.0
            if mode < 0.48:
                return self._signed_command_magnitude(0.35, 1.0), 0.0
            if mode < 0.76:
                return 0.0, self._signed_command_magnitude(0.30, 0.85)
            return (
                self._signed_command_magnitude(0.35, 0.90),
                self._signed_command_magnitude(0.20, 0.55),
            )

        return self._sample_full_command()

    def _sample_full_command(self) -> tuple[float, float]:
        mode = float(self.np_random.uniform())
        if mode < 0.05:
            return 0.0, 0.0

        if mode < 0.40:
            forward = self._signed_command_magnitude(0.35, 1.0)
            return forward, 0.0

        if mode < 0.65:
            yaw_rate = self._signed_command_magnitude(0.35, 1.0)
            return 0.0, yaw_rate

        forward = self._signed_command_magnitude(0.35, 1.0)
        yaw_rate = self._signed_command_magnitude(0.25, 0.80)
        return forward, yaw_rate

    def _signed_command_magnitude(self, low: float, high: float) -> float:
        magnitude = float(self.np_random.uniform(low, high))
        sign = -1.0 if float(self.np_random.uniform()) < 0.5 else 1.0
        return sign * magnitude

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        action_delta = action - self.policy_action
        action_accel = action_delta - self.previous_action_delta
        self.previous_policy_action[:] = self.policy_action
        self.policy_action[:] = action
        self.previous_action_delta[:] = action_delta
        self.delayed_action[:] = self._latency_action(action)

        self._update_servo_targets()
        self._apply_voltage_sag()

        x_before = self._x_position()
        y_before = self._y_position()
        foot_positions_before = self._foot_positions()
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        x_after = self._x_position()
        y_after = self._y_position()
        foot_positions_after = self._foot_positions()
        foot_horizontal_speeds = np.linalg.norm(
            (foot_positions_after[:, :2] - foot_positions_before[:, :2]) / self.dt,
            axis=1,
        )

        self.steps += 1
        self.gait_phase = (self.gait_phase + GAIT_PHASE_HZ * self.dt) % 1.0
        self._update_imu_state()

        world_x_velocity = (x_after - x_before) / self.dt
        world_y_velocity = (y_after - y_before) / self.dt
        body_velocity = self._world_velocity_to_body(world_x_velocity, world_y_velocity)
        body_forward_velocity = float(body_velocity[0])
        body_lateral_velocity = float(body_velocity[1])
        yaw_rate = float(self._sensor("gyro", 3)[2])
        desired_forward_velocity = self.forward_command * MAX_FORWARD_COMMAND_MPS
        desired_yaw_rate = self.yaw_rate_command * MAX_YAW_COMMAND_RAD_S

        previous_desired_heading = self.desired_heading
        reward_heading = wrap_angle(
            previous_desired_heading + 0.5 * desired_yaw_rate * self.dt
        )
        self.desired_heading = wrap_angle(
            previous_desired_heading + desired_yaw_rate * self.dt
        )
        current_heading = self._heading()
        self.heading_error = wrap_angle(current_heading - self.desired_heading)

        desired_axis = np.array(
            [np.cos(reward_heading), np.sin(reward_heading)], dtype=np.float64
        )
        lateral_axis = np.array([-desired_axis[1], desired_axis[0]], dtype=np.float64)
        world_velocity = np.array(
            [world_x_velocity, world_y_velocity], dtype=np.float64
        )
        world_displacement = np.array(
            [x_after - self.start_x, y_after - self.start_y], dtype=np.float64
        )
        line_velocity = float(np.dot(world_velocity, desired_axis))
        cross_track_velocity = float(np.dot(world_velocity, lateral_axis))
        self.cross_track_error = float(np.dot(world_displacement, lateral_axis))
        planar_speed = float(np.linalg.norm(world_velocity))

        forward_progress = self.forward_command * line_velocity
        yaw_progress = self.yaw_rate_command * yaw_rate
        forward_error = line_velocity - desired_forward_velocity
        yaw_error = yaw_rate - desired_yaw_rate
        forward_weight = abs(self.forward_command)
        yaw_weight = abs(self.yaw_rate_command)
        straight_weight = forward_weight * max(0.0, 1.0 - yaw_weight)
        rotate_in_place_weight = yaw_weight * max(0.0, 1.0 - forward_weight)
        self.line_distance += forward_progress * self.dt
        self.yaw_distance += yaw_progress * self.dt
        command_progress_rate = forward_progress + 0.10 * yaw_progress
        self.command_distance += command_progress_rate * self.dt
        obs = self._get_obs()
        height = float(self.data.xpos[self.chassis_id, 2])
        upright = self._upright()
        foot_contact_mask = self._foot_contact_mask()
        foot_contact_count = int(np.count_nonzero(foot_contact_mask))
        contact_foot_speeds = foot_horizontal_speeds[foot_contact_mask]
        mean_contact_foot_speed = (
            float(np.mean(contact_foot_speeds)) if contact_foot_speeds.size else 0.0
        )
        max_contact_foot_speed = (
            float(np.max(contact_foot_speeds)) if contact_foot_speeds.size else 0.0
        )
        below_body_limit = height < MIN_CHASSIS_CENTER_HEIGHT_M
        battery_side_down = upright < BATTERY_SIDE_DOWN_UPRIGHT_LIMIT
        terminated = bool(
            below_body_limit or battery_side_down or not np.isfinite(obs).all()
        )

        episode_fraction = min(1.0, self.steps / self.max_steps)
        progress_pressure = smoothstep((episode_fraction - 0.05) / 0.25)
        coaching_ramp = smoothstep((episode_fraction - 0.14) / 0.50)
        polish_ramp = smoothstep((episode_fraction - 0.45) / 0.40)
        soft_penalty_scale = 0.25 + 0.75 * coaching_ramp
        regularizer_scale = 0.06 + 0.94 * coaching_ramp
        battery_orientation_quality = smoothstep(
            (upright - BATTERY_SIDE_DOWN_UPRIGHT_LIMIT)
            / (BATTERY_SIDE_WARNING_UPRIGHT - BATTERY_SIDE_DOWN_UPRIGHT_LIMIT)
        )
        alive_quality = 0.0 if terminated else battery_orientation_quality
        commanded_speed = abs(desired_forward_velocity)
        commanded_yaw_speed = abs(desired_yaw_rate)
        command_activity = float(np.clip(forward_weight + yaw_weight, 0.0, 1.0))
        idle_weight = 1.0 - command_activity
        mixed_weight = forward_weight * yaw_weight
        rewarded_forward_progress = min(
            max(0.0, forward_progress),
            max(0.035, 1.40 * commanded_speed),
        )
        rewarded_yaw_progress = min(
            max(0.0, float(yaw_progress)),
            max(0.25, 1.35 * commanded_yaw_speed),
        )
        linear_error = line_velocity - desired_forward_velocity
        cross_track_weight = 0.35 + 0.65 * max(0.0, 1.0 - yaw_weight)
        linear_tracking_error = (
            linear_error * linear_error
            + np.square(cross_track_weight * cross_track_velocity)
        )
        linear_tracking_reward = float(
            np.exp(-linear_tracking_error / np.square(LINEAR_TRACKING_SIGMA_MPS))
        )
        yaw_tracking_reward = tracking_exp(yaw_error, YAW_TRACKING_SIGMA_RAD_S)
        heading_tracking_reward = tracking_exp(
            self.heading_error,
            HEADING_TRACKING_SIGMA_RAD,
        )
        straight_weight = forward_weight * max(0.0, 1.0 - yaw_weight)
        heading_gate = 1.0 - 0.55 * straight_weight * (
            1.0 - heading_tracking_reward
        )
        yaw_still_gate = 1.0 - 0.65 * straight_weight * (1.0 - yaw_tracking_reward)
        gated_linear_tracking_reward = (
            linear_tracking_reward * heading_gate * yaw_still_gate
        )
        coupled_tracking_reward = (
            gated_linear_tracking_reward
            * (
                yaw_tracking_reward
                if yaw_weight > 0.05 or straight_weight > 0.05
                else 1.0
            )
        )
        idle_tracking_reward = idle_weight * tracking_exp(planar_speed, 0.035) * (
            tracking_exp(yaw_rate, 0.35)
        )
        forward_motion_quality = (
            smoothstep(forward_progress / max(0.020, 0.30 * commanded_speed))
            if commanded_speed > 1e-6
            else 0.0
        )
        yaw_motion_quality = (
            smoothstep(yaw_progress / max(0.15, 0.30 * commanded_yaw_speed))
            if commanded_yaw_speed > 1e-6
            else 0.0
        )
        command_quality = (
            (
                forward_weight * gated_linear_tracking_reward
                + yaw_weight * yaw_tracking_reward
            )
            / max(1e-6, forward_weight + yaw_weight)
            if command_activity > 0.0
            else idle_tracking_reward
        )
        self.command_quality_ema = 0.93 * self.command_quality_ema + 0.07 * float(
            command_quality
        )
        crawl_polish_gate = (
            alive_quality
            * command_activity
            * smoothstep((self.command_quality_ema - 0.20) / 0.45)
            * smoothstep((episode_fraction - 0.10) / 0.25)
        )
        support_deficit = max(0, 2 - foot_contact_count)
        slip_excess = np.maximum(0.0, contact_foot_speeds - 0.020)
        foot_slip_cost = (
            crawl_polish_gate
            * (
                0.08 * float(np.mean(slip_excess))
                + 0.18 * float(np.mean(np.square(slip_excess)))
            )
            if slip_excess.size
            else 0.0
        )
        support_deficit_cost = crawl_polish_gate * 0.035 * support_deficit
        crawl_effort_cost = crawl_polish_gate * (
            0.009 * float(np.mean(np.abs(action_delta)))
            + 0.012 * float(np.mean(np.abs(action_accel)))
        )

        progress_reward = alive_quality * (
            10.5 * rewarded_forward_progress * (0.55 + 0.45 * heading_gate)
            + 1.10 * yaw_weight * rewarded_yaw_progress
        )
        forward_reward = (
            forward_weight
            * alive_quality
            * (
                0.54 * forward_motion_quality
                + 0.18 * gated_linear_tracking_reward
            )
        )
        yaw_reward = (
            yaw_weight
            * alive_quality
            * (0.46 * yaw_motion_quality + 0.16 * yaw_tracking_reward)
        )
        command_tracking_reward = alive_quality * (
            0.16 * forward_weight * gated_linear_tracking_reward
            + 0.14 * yaw_weight * yaw_tracking_reward
            + 0.10 * mixed_weight * coupled_tracking_reward
            + 0.16 * idle_tracking_reward
        )
        command_motion_reward = alive_quality * (
            0.06 * forward_weight * forward_motion_quality
            + 0.06 * yaw_weight * yaw_motion_quality
        )
        forward_stall = (
            forward_weight
            * (1.0 - forward_motion_quality)
            * smoothstep(forward_weight / 0.20)
        )
        yaw_stall = (
            yaw_weight
            * (1.0 - yaw_motion_quality)
            * smoothstep(yaw_weight / 0.20)
        )
        command_stall_penalty = alive_quality * (
            progress_pressure * (0.20 * forward_stall + 0.16 * yaw_stall)
        )
        keep_going_reward = 0.0
        alive_reward = 0.0
        command_sign_penalty_scale = 0.70 + 0.30 * soft_penalty_scale
        wrong_way_penalty = command_sign_penalty_scale * (
            9.0 * forward_weight * max(0.0, -forward_progress)
            + 1.25 * yaw_weight * max(0.0, -yaw_progress)
        )
        overspeed_penalty = (
            soft_penalty_scale
            * forward_weight
            * 0.35
            * max(0.0, forward_progress - max(0.05, 1.35 * commanded_speed))
        )
        upright_penalty = (
            soft_penalty_scale
            * 0.25
            * max(0.0, BATTERY_SIDE_WARNING_UPRIGHT - upright)
        )
        sideways_penalty = (
            soft_penalty_scale
            * straight_weight
            * (
                0.10 * abs(cross_track_velocity)
                + 0.12 * min(abs(self.cross_track_error), 0.75)
            )
        )
        translation_penalty = (
            soft_penalty_scale * rotate_in_place_weight * 0.10 * planar_speed
        )
        idle_motion_cost = alive_quality * idle_weight * (
            0.30 * planar_speed + 0.12 * abs(yaw_rate)
        )
        forward_tracking_cost = (
            soft_penalty_scale * forward_weight * 0.025 * abs(forward_error)
        )
        yaw_tracking_cost = (
            soft_penalty_scale * (0.02 + 0.18 * straight_weight + 0.08 * yaw_weight)
        ) * abs(yaw_error)
        heading_tracking_cost = (
            soft_penalty_scale * (0.01 + 0.20 * straight_weight + 0.06 * yaw_weight)
        ) * min(abs(self.heading_error), np.pi * 0.5)
        roll_pitch_rate = float(np.linalg.norm(self._sensor("gyro", 3)[:2]))
        angular_rate_cost = (
            soft_penalty_scale * 0.006 + polish_ramp * 0.020
        ) * roll_pitch_rate
        airtime_cost = (soft_penalty_scale * 0.010 + polish_ramp * 0.018) * max(
            0, 2 - foot_contact_count
        )
        ctrl_cost = regularizer_scale * 0.0018 * float(np.mean(np.square(action)))
        smoothness_cost = (regularizer_scale * 0.007 + polish_ramp * 0.017) * float(
            np.mean(np.square(action_delta))
        )
        jerk_cost = (regularizer_scale * 0.008 + polish_ramp * 0.018) * float(
            np.mean(np.square(action_accel))
        )
        hinge_velocity_cost = (
            regularizer_scale * 0.0006 + polish_ramp * 0.0012
        ) * self._hinge_velocity_cost()
        remaining_steps = max(0, self.max_steps - self.steps)
        termination_penalty = 35.0 + 0.16 * remaining_steps if terminated else 0.0
        reward = (
            progress_reward
            + forward_reward
            + yaw_reward
            + command_tracking_reward
            + command_motion_reward
            + keep_going_reward
            + alive_reward
            - wrong_way_penalty
            - command_stall_penalty
            - overspeed_penalty
            - upright_penalty
            - sideways_penalty
            - translation_penalty
            - idle_motion_cost
            - forward_tracking_cost
            - yaw_tracking_cost
            - heading_tracking_cost
            - angular_rate_cost
            - foot_slip_cost
            - support_deficit_cost
            - crawl_effort_cost
            - airtime_cost
            - ctrl_cost
            - smoothness_cost
            - jerk_cost
            - hinge_velocity_cost
            - termination_penalty
        )

        truncated = self.steps >= self.max_steps
        info = {
            "x_position": x_after,
            "x_distance": x_after - self.start_x,
            "y_distance": y_after - self.start_y,
            "command_distance": self.command_distance,
            "line_distance": self.line_distance,
            "yaw_distance": self.yaw_distance,
            "forward_command": self.forward_command,
            "yaw_rate_command": self.yaw_rate_command,
            "gait_phase": self.gait_phase,
            "desired_forward_velocity": desired_forward_velocity,
            "desired_yaw_rate": desired_yaw_rate,
            "forward_velocity": world_x_velocity,
            "lateral_velocity": world_y_velocity,
            "world_x_velocity": world_x_velocity,
            "world_y_velocity": world_y_velocity,
            "line_velocity": line_velocity,
            "cross_track_velocity": cross_track_velocity,
            "cross_track_error": self.cross_track_error,
            "body_forward_velocity": body_forward_velocity,
            "body_lateral_velocity": body_lateral_velocity,
            "yaw_rate": yaw_rate,
            "heading": current_heading,
            "desired_heading": self.desired_heading,
            "heading_error": self.heading_error,
            "forward_progress": forward_progress,
            "rewarded_forward_progress": rewarded_forward_progress,
            "yaw_progress": yaw_progress,
            "rewarded_yaw_progress": rewarded_yaw_progress,
            "command_progress_rate": command_progress_rate,
            "forward_error": forward_error,
            "yaw_error": yaw_error,
            "linear_tracking_reward": linear_tracking_reward,
            "yaw_tracking_reward": yaw_tracking_reward,
            "heading_tracking_reward": heading_tracking_reward,
            "gated_linear_tracking_reward": gated_linear_tracking_reward,
            "coupled_tracking_reward": coupled_tracking_reward,
            "heading_gate": heading_gate,
            "yaw_still_gate": yaw_still_gate,
            "idle_tracking_reward": idle_tracking_reward,
            "forward_motion_quality": forward_motion_quality,
            "yaw_motion_quality": yaw_motion_quality,
            "command_quality": command_quality,
            "command_quality_ema": self.command_quality_ema,
            "crawl_polish_gate": crawl_polish_gate,
            "planar_speed": planar_speed,
            "height": height,
            "upright": upright,
            "below_body_limit": below_body_limit,
            "battery_side_down": battery_side_down,
            "foot_contact_count": foot_contact_count,
            "mean_contact_foot_speed": mean_contact_foot_speed,
            "max_contact_foot_speed": max_contact_foot_speed,
            "support_deficit": support_deficit,
            "remaining_steps": remaining_steps,
            "episode_fraction": episode_fraction,
            "progress_pressure": progress_pressure,
            "coaching_ramp": coaching_ramp,
            "polish_ramp": polish_ramp,
            "soft_penalty_scale": soft_penalty_scale,
            "command_sign_penalty_scale": command_sign_penalty_scale,
            "regularizer_scale": regularizer_scale,
            "alive_quality": alive_quality,
            "battery_orientation_quality": battery_orientation_quality,
            "body_height_limit": MIN_CHASSIS_CENTER_HEIGHT_M,
            "battery_side_down_upright_limit": BATTERY_SIDE_DOWN_UPRIGHT_LIMIT,
            "roll_pitch_rate": roll_pitch_rate,
            "voltage": self._current_voltage(),
            "servo_torque_limit_nm": float(
                np.mean(self.model.actuator_forcerange[:, 1])
            ),
            "domain_randomization_strength": self.domain.strength,
            "floor_friction": self.domain.floor_friction,
            "foot_friction": self.domain.foot_friction,
            "control_latency_steps": self.domain.control_latency_steps,
            "imu_latency_steps": self.domain.imu_latency_steps,
            "forward_reward": forward_reward,
            "yaw_reward": yaw_reward,
            "progress_reward": progress_reward,
            "command_tracking_reward": command_tracking_reward,
            "command_motion_reward": command_motion_reward,
            "keep_going_reward": keep_going_reward,
            "alive_reward": alive_reward,
            "wrong_way_penalty": wrong_way_penalty,
            "command_stall_penalty": command_stall_penalty,
            "overspeed_penalty": overspeed_penalty,
            "upright_penalty": upright_penalty,
            "sideways_penalty": sideways_penalty,
            "translation_penalty": translation_penalty,
            "idle_motion_cost": idle_motion_cost,
            "forward_tracking_cost": forward_tracking_cost,
            "yaw_tracking_cost": yaw_tracking_cost,
            "heading_tracking_cost": heading_tracking_cost,
            "angular_rate_cost": angular_rate_cost,
            "foot_slip_cost": foot_slip_cost,
            "support_deficit_cost": support_deficit_cost,
            "crawl_effort_cost": crawl_effort_cost,
            "airtime_cost": airtime_cost,
            "ctrl_cost": ctrl_cost,
            "smoothness_cost": smoothness_cost,
            "jerk_cost": jerk_cost,
            "hinge_velocity_cost": hinge_velocity_cost,
            "termination_penalty": termination_penalty,
        }
        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        command_delta = self.policy_action - self.previous_policy_action
        heading_features = np.array(
            [np.sin(self.heading_error), np.cos(self.heading_error)], dtype=np.float32
        )
        phase_features = np.array(
            [
                np.sin(2.0 * np.pi * self.gait_phase),
                np.cos(2.0 * np.pi * self.gait_phase),
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            [
                self.measured_gyro,
                self.measured_accel_g,
                self.measured_gravity,
                self.command,
                phase_features,
                heading_features,
                self.policy_action,
                command_delta,
            ]
        ).astype(np.float32)

    def _sample_domain(self, active: bool) -> None:
        strength = self.domain_randomization_strength if active else 0.0
        self.domain = DomainParams(active=active, strength=strength)
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

        max_latency_steps = int(np.ceil(2.0 * strength))
        self.domain.initial_voltage = self._uniform_around(7.8, 7.4, 8.4, strength)
        drop = float(self.np_random.uniform(0.0, 0.45 * strength))
        self.domain.final_voltage = max(6.8, self.domain.initial_voltage - drop)
        self.domain.control_latency_steps = int(
            self.np_random.integers(0, max_latency_steps + 1)
        )
        self.domain.imu_latency_steps = int(
            self.np_random.integers(0, max_latency_steps + 1)
        )
        self.domain.floor_friction = self._uniform_around(1.2, 0.80, 1.80, strength)
        self.domain.foot_friction = self._uniform_around(1.7, 1.00, 2.20, strength)

        self.servo_zero_offset[:] = self.np_random.normal(
            0.0, 0.025 * strength, self.model.nu
        )
        self.servo_gain[:] = self.np_random.uniform(
            1.0 - 0.05 * strength, 1.0 + 0.05 * strength, self.model.nu
        )
        self.servo_strength[:] = self.np_random.uniform(
            1.0 - 0.30 * strength, 1.0, self.model.nu
        )
        self.servo_speed[:] = MAX_SERVO_SPEED_RAD_S * self.np_random.uniform(
            1.0 - 0.25 * strength, 1.00, self.model.nu
        )
        self.servo_deadband[:] = self.np_random.uniform(
            0.0, 0.012 * strength, self.model.nu
        )
        self.servo_quantization[:] = self.np_random.uniform(
            0.0, 0.004 * strength, self.model.nu
        )
        self.servo_time_constant[:] = self.np_random.uniform(
            0.02 - 0.005 * strength, 0.02 + 0.035 * strength, self.model.nu
        )

        self.gyro_bias[:] = self.np_random.normal(0.0, 0.020 * strength, 3)
        self.accel_bias_g[:] = self.np_random.normal(0.0, 0.015 * strength, 3)
        self.gyro_noise_std = float(self.np_random.uniform(0.0, 0.020 * strength))
        self.accel_noise_std_g = float(self.np_random.uniform(0.0, 0.035 * strength))
        self.gravity_noise_std = float(self.np_random.uniform(0.0, 0.020 * strength))
        self.gyro_bias_walk_std = float(self.np_random.uniform(0.0, 0.0008 * strength))
        self.accel_bias_walk_std_g = float(
            self.np_random.uniform(0.0, 0.00045 * strength)
        )
        self.gravity_filter_alpha = self._uniform_around(0.10, 0.05, 0.14, strength)

    def _apply_domain_to_model(self) -> None:
        if not self.domain.active:
            return

        strength = self.domain.strength
        floor_torsional = self._uniform_around(0.04, 0.01, 0.12, strength)
        floor_rolling = self._uniform_around(0.006, 0.0005, 0.02, strength)
        self.model.geom_friction[self.floor_id] = [
            self.domain.floor_friction,
            floor_torsional,
            floor_rolling,
        ]
        for geom_id in self.foot_geom_ids:
            torsional = self._uniform_around(0.08, 0.03, 0.16, strength)
            rolling = self._uniform_around(0.010, 0.003, 0.025, strength)
            self.model.geom_friction[geom_id] = [
                self.domain.foot_friction,
                torsional,
                rolling,
            ]
            self.model.geom_size[geom_id, 0] *= float(
                self.np_random.uniform(1.0 - 0.10 * strength, 1.0 + 0.15 * strength)
            )
        for geom_id in self.leg_geom_ids:
            self.model.geom_friction[geom_id] = [
                self._uniform_around(1.15, 0.85, 1.55, strength),
                self._uniform_around(0.03, 0.015, 0.06, strength),
                self._uniform_around(0.006, 0.002, 0.014, strength),
            ]

        slope = float(self.np_random.uniform(0.0, np.deg2rad(3.0) * strength))
        heading = float(self.np_random.uniform(-np.pi, np.pi))
        self.model.opt.gravity[:] = STANDARD_GRAVITY * np.array(
            [
                np.sin(slope) * np.cos(heading),
                np.sin(slope) * np.sin(heading),
                -np.cos(slope),
            ]
        )

        chassis_scale = float(
            self.np_random.uniform(1.0 - 0.10 * strength, 1.0 + 0.15 * strength)
        )
        self._scale_body_mass(self.chassis_id, chassis_scale)
        for body_id in self.leg_body_ids:
            self._scale_body_mass(
                body_id,
                float(
                    self.np_random.uniform(1.0 - 0.10 * strength, 1.0 + 0.20 * strength)
                ),
            )
            self.model.body_pos[body_id, :2] += self.np_random.normal(
                0.0, 0.0010 * strength, 2
            )
            self.model.body_pos[body_id, 2] += float(
                self.np_random.normal(0.0, 0.0005 * strength)
            )

        for joint_id in self.hinge_joint_ids:
            dof_addr = int(self.model.jnt_dofadr[joint_id])
            self.model.dof_damping[dof_addr] *= float(
                self.np_random.uniform(1.0 - 0.20 * strength, 1.0 + 0.30 * strength)
            )

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
        self.previous_action_delta[:] = 0
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
        return (
            self._interp_servo_spec(voltage, SERVO_SPEED_SPEC) / MAX_SERVO_SPEED_RAD_S
        )

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

    def _uniform_around(
        self,
        nominal: float,
        low: float,
        high: float,
        strength: float,
    ) -> float:
        scaled_low = nominal + (low - nominal) * strength
        scaled_high = nominal + (high - nominal) * strength
        return float(self.np_random.uniform(scaled_low, scaled_high))

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

    def _heading(self) -> float:
        xmat = self.data.xmat[self.chassis_id].reshape(3, 3)
        body_x_world = xmat[:, 0]
        return float(np.arctan2(body_x_world[1], body_x_world[0]))

    def _world_velocity_to_body(
        self, x_velocity: float, y_velocity: float
    ) -> np.ndarray:
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

    def _hinge_velocity_cost(self) -> float:
        velocities = self.data.qvel[self.hinge_dof_addresses]
        return float(np.mean(np.square(velocities)))

    def _foot_positions(self) -> np.ndarray:
        return self.data.geom_xpos[self.foot_geom_ids].copy()

    def _foot_contact_mask(self) -> np.ndarray:
        contact_mask = np.zeros(len(self.foot_geom_ids), dtype=bool)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            geom_1 = int(contact.geom1)
            geom_2 = int(contact.geom2)
            if geom_1 == self.floor_id and geom_2 in self.foot_geom_index:
                contact_mask[self.foot_geom_index[geom_2]] = True
            elif geom_2 == self.floor_id and geom_1 in self.foot_geom_index:
                contact_mask[self.foot_geom_index[geom_1]] = True
        return contact_mask

    def _foot_contact_count(self) -> int:
        return int(np.count_nonzero(self._foot_contact_mask()))

    def _reset_info(self) -> dict[str, float | int | bool]:
        return {
            "domain_randomization": self.domain.active,
            "domain_randomization_strength": self.domain.strength,
            "voltage": self._current_voltage(),
            "forward_command": self.forward_command,
            "yaw_rate_command": self.yaw_rate_command,
            "gait_phase": self.gait_phase,
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


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def tracking_exp(error: float, sigma: float) -> float:
    sigma = max(float(sigma), 1e-6)
    return float(np.exp(-np.square(float(error) / sigma)))


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
