from typing import Callable

import gymnasium as gym
from gymnasium.envs.mujoco.half_cheetah_v5 import HalfCheetahEnv

env_name = "HalfCheetah-v5"
_already_logged = []


def create_half_cheetah(
    gravity: float,
    make_func: Callable[..., gym.Env],
    **kwargs,
):
    if gravity not in _already_logged:
        _already_logged.append(gravity)
        print(f"Creating {env_name}. Parameters:")
        print(f"- gravity: {gravity}")
    env: gym.Env = make_func(**kwargs)
    env_unwrapped: HalfCheetahEnv = env.unwrapped
    env_unwrapped.gravity = gravity
    return env
