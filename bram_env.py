from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


DEFAULT_XML_PATH = Path(__file__).with_name("bram.xml")
STANDARD_GRAVITY = 9.81


class BramTripodEnv(gym.Env):
    """Low-sensor MuJoCo env for the Bram tripod.

    Policy observations are intentionally deploy-realistic:
    IMU readings plus the last commanded servo targets. Training reward still
    uses simulator-only forward displacement because the physical robot does
    not need to observe reward at runtime.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        xml_path: str | Path = DEFAULT_XML_PATH,
        frame_skip: int = 10,
        episode_seconds: float = 8.0,
        randomize_reset: bool = True,
    ) -> None:
        super().__init__()
        self.xml_path = Path(xml_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.dt = self.model.opt.timestep * self.frame_skip
        self.max_steps = max(1, int(episode_seconds / self.dt))
        self.randomize_reset = randomize_reset

        self.chassis_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis"
        )
        if self.chassis_id < 0:
            raise ValueError("Could not find body named 'chassis' in the MuJoCo model.")

        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self.ctrl_center = ((self.ctrl_low + self.ctrl_high) * 0.5).astype(np.float32)
        self.ctrl_half_range = ((self.ctrl_high - self.ctrl_low) * 0.5).astype(np.float32)

        self.servo_targets = np.zeros(self.model.nu, dtype=np.float32)
        self.previous_servo_targets = np.zeros(self.model.nu, dtype=np.float32)
        self.steps = 0
        self.start_x = 0.0

        self.sensor_slices = self._sensor_slices()
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
        mujoco.mj_resetData(self.model, self.data)
        self.servo_targets[:] = 0
        self.previous_servo_targets[:] = 0
        self.data.ctrl[:] = self.ctrl_center

        randomize = self.randomize_reset if options is None else options.get(
            "randomize", self.randomize_reset
        )
        if randomize:
            self._randomize_joint_angles()

        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.start_x = self._x_position()
        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.previous_servo_targets[:] = self.servo_targets
        self.servo_targets[:] = action
        self.data.ctrl[:] = self._scale_action(action)

        x_before = self._x_position()
        y_before = self._y_position()
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        x_after = self._x_position()
        y_after = self._y_position()

        self.steps += 1
        obs = self._get_obs()
        forward_velocity = (x_after - x_before) / self.dt
        lateral_velocity = (y_after - y_before) / self.dt
        height = float(self.data.xpos[self.chassis_id, 2])
        upright = self._upright()
        action_delta = action - self.previous_servo_targets

        forward_reward = forward_velocity
        alive_reward = 0.04 * max(0.0, upright)
        upright_penalty = 0.10 * max(0.0, 0.65 - upright)
        sideways_penalty = 0.03 * abs(lateral_velocity)
        ctrl_cost = 0.015 * float(np.mean(np.square(action)))
        smoothness_cost = 0.020 * float(np.mean(np.square(action_delta)))
        reward = (
            forward_reward
            + alive_reward
            - upright_penalty
            - sideways_penalty
            - ctrl_cost
            - smoothness_cost
        )

        terminated = bool(
            height < 0.025
            or upright < 0.10
            or not np.isfinite(obs).all()
        )
        truncated = self.steps >= self.max_steps
        info = {
            "x_position": x_after,
            "x_distance": x_after - self.start_x,
            "forward_velocity": forward_velocity,
            "lateral_velocity": lateral_velocity,
            "height": height,
            "upright": upright,
            "forward_reward": forward_reward,
            "alive_reward": alive_reward,
            "upright_penalty": upright_penalty,
            "sideways_penalty": sideways_penalty,
            "ctrl_cost": ctrl_cost,
            "smoothness_cost": smoothness_cost,
        }
        return obs, float(reward), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        gyro = self._sensor("gyro", 3)
        accel = self._sensor("accel", 3) / STANDARD_GRAVITY
        orientation = self._sensor("orientation", 4)
        servo_delta = self.servo_targets - self.previous_servo_targets
        return np.concatenate(
            [
                gyro.astype(np.float32),
                accel.astype(np.float32),
                orientation.astype(np.float32),
                self.servo_targets.astype(np.float32),
                servo_delta.astype(np.float32),
            ]
        ).astype(np.float32)

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
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id
            )
            if name is None:
                continue
            start = int(self.model.sensor_adr[sensor_id])
            dim = int(self.model.sensor_dim[sensor_id])
            slices[name] = slice(start, start + dim)
        return slices

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        return self.ctrl_center + self.ctrl_half_range * action

    def _randomize_joint_angles(self) -> None:
        for joint_id in range(self.model.njnt):
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE:
                qpos_addr = self.model.jnt_qposadr[joint_id]
                self.data.qpos[qpos_addr] = self.np_random.uniform(-0.04, 0.04)

    def _upright(self) -> float:
        xmat = self.data.xmat[self.chassis_id].reshape(3, 3)
        return float(xmat[2, 2])

    def _x_position(self) -> float:
        return float(self.data.xpos[self.chassis_id, 0])

    def _y_position(self) -> float:
        return float(self.data.xpos[self.chassis_id, 1])
