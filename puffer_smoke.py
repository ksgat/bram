from __future__ import annotations

import numpy as np
from pufferlib.emulation import GymnasiumPufferEnv

from bram_env import BramTripodEnv


def main() -> None:
    env = GymnasiumPufferEnv(env_creator=BramTripodEnv)
    obs, _ = env.reset(seed=1)
    total_reward = 0.0

    for _ in range(32):
        action = np.asarray([env.single_action_space.sample()])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"obs_shape={np.asarray(obs).shape}")
    print(f"total_reward={total_reward:.3f}")
    print(f"last_info={info}")


if __name__ == "__main__":
    main()
