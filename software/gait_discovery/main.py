from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mujoco
import numpy as np

from bram_env import BramTripodEnv


DEFAULT_XML_PATH = Path(__file__).with_name("bram.xml")


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    print_model_summary(model, args.xml)

    if args.env_check:
        run_env_check(args.xml, args.seconds)
        return

    if args.headless:
        run_headless(model, data, args)
    else:
        run_viewer(model, data, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Bram tripod MuJoCo model.")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML_PATH)
    parser.add_argument("--headless", action="store_true", help="Run without the viewer.")
    parser.add_argument(
        "--env-check",
        action="store_true",
        help="Step the Gymnasium environment with random actions.",
    )
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--no-gait", action="store_true", help="Leave actuator controls at 0.")
    parser.add_argument("--pause", action="store_true", help="Open the viewer without stepping physics.")
    parser.add_argument(
        "--fake-walk",
        action="store_true",
        help="Animate hinge qpos directly in the viewer without stepping physics.",
    )
    parser.add_argument("--amplitude", type=float, default=0.35)
    parser.add_argument("--frequency", type=float, default=0.7)
    return parser.parse_args()


def print_model_summary(model: mujoco.MjModel, xml_path: Path) -> None:
    print(f"Loaded {xml_path}")
    print(
        f"nq={model.nq} nv={model.nv} nu={model.nu} "
        f"nsensor={model.nsensor} timestep={model.opt.timestep:g}"
    )
    print("joints:    " + ", ".join(names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)))
    print(
        "actuators: "
        + ", ".join(names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu))
    )
    print(
        "sensors:   "
        + ", ".join(names(model, mujoco.mjtObj.mjOBJ_SENSOR, model.nsensor))
    )


def names(model: mujoco.MjModel, obj_type: mujoco.mjtObj, count: int) -> list[str]:
    return [
        mujoco.mj_id2name(model, obj_type, index) or f"unnamed_{index}"
        for index in range(count)
    ]


def run_headless(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    args: argparse.Namespace,
) -> None:
    steps = max(1, int(args.seconds / model.opt.timestep))
    start_x = chassis_x(model, data)
    for step in range(steps):
        elapsed = step * model.opt.timestep
        if not args.no_gait:
            apply_tripod_gait(model, data, elapsed, args.amplitude, args.frequency)
        mujoco.mj_step(model, data)

    end_x = chassis_x(model, data)
    print(
        f"Simulated {steps} steps / {steps * model.opt.timestep:.2f}s. "
        f"chassis_x {start_x:.4f} -> {end_x:.4f}"
    )


def run_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    args: argparse.Namespace,
) -> None:
    import mujoco.viewer

    started = time.monotonic()
    mujoco.mj_forward(model, data)
    fake_walk_joints = fake_walk_joint_addresses(model) if args.fake_walk else []
    if fake_walk_joints:
        print("fake-walk joints: " + ", ".join(name for name, _ in fake_walk_joints))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        configure_viewer_visuals(viewer)
        while viewer.is_running():
            elapsed = time.monotonic() - started
            if args.seconds > 0 and elapsed >= args.seconds:
                break

            step_started = time.monotonic()
            if args.fake_walk:
                apply_fake_walk(model, data, fake_walk_joints, elapsed, args.amplitude, args.frequency)
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(max(0.0, (1.0 / 60.0) - (time.monotonic() - step_started)))
                continue

            if args.pause:
                viewer.sync()
                time.sleep(0.02)
                continue

            if not args.no_gait:
                apply_tripod_gait(model, data, elapsed, args.amplitude, args.frequency)
            mujoco.mj_step(model, data)
            viewer.sync()

            sleep_time = model.opt.timestep - (time.monotonic() - step_started)
            if sleep_time > 0:
                time.sleep(sleep_time)


def configure_viewer_visuals(viewer) -> None:
    for flag in [
        mujoco.mjtVisFlag.mjVIS_INERTIA,
        mujoco.mjtVisFlag.mjVIS_SCLINERTIA,
        mujoco.mjtVisFlag.mjVIS_CONTACTPOINT,
        mujoco.mjtVisFlag.mjVIS_CONTACTFORCE,
    ]:
        viewer.opt.flags[int(flag)] = 0


def apply_tripod_gait(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    elapsed: float,
    amplitude: float,
    frequency: float,
) -> None:
    phases = np.array([0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0])
    ctrl = amplitude * np.sin(2.0 * math.pi * frequency * elapsed + phases[: model.nu])
    low = model.actuator_ctrlrange[:, 0]
    high = model.actuator_ctrlrange[:, 1]
    data.ctrl[:] = np.clip(ctrl, low, high)


def fake_walk_joint_addresses(model: mujoco.MjModel) -> list[tuple[str, int]]:
    joint_names = ["hip_front", "hip_back_left", "hip_back_right"]
    joint_addresses = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Could not find joint named {joint_name!r}.")
        joint_addresses.append((joint_name, int(model.jnt_qposadr[joint_id])))
    return joint_addresses


def apply_fake_walk(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_addresses: list[tuple[str, int]],
    elapsed: float,
    amplitude: float,
    frequency: float,
) -> None:
    phases = np.array([0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0])
    pose = amplitude * np.sin(2.0 * math.pi * frequency * elapsed + phases)
    for (_, qpos_addr), value in zip(joint_addresses, pose, strict=True):
        data.qpos[qpos_addr] = float(np.clip(value, -0.75, 0.75))
    data.qvel[:] = 0
    data.ctrl[:] = pose[: model.nu]


def run_env_check(xml_path: Path, seconds: float) -> None:
    env = BramTripodEnv(xml_path=xml_path)
    obs, _ = env.reset(seed=1)
    steps = max(1, int(seconds / env.dt))
    total_reward = 0.0
    final_info = {}
    for _ in range(steps):
        obs, reward, terminated, truncated, final_info = env.step(env.action_space.sample())
        total_reward += reward
        if terminated or truncated:
            break
    print(
        f"Env check: obs_shape={obs.shape} steps={env.steps} "
        f"total_reward={total_reward:.3f} info={final_info}"
    )


def chassis_x(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    chassis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
    if chassis_id < 0:
        return 0.0
    return float(data.xpos[chassis_id, 0])


if __name__ == "__main__":
    main()
