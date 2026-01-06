from typing import Type

import gymnasium as gym
import jax
import jax.numpy as jnp

from rl_blox.blox.env_util import AppendHistoryWrapper
from rl_blox.blox.vec_env_util import AppendHistoryVecEnvWrapper

env_name = "MountainCar-v0"

vec_env_wrappers = []
env_wrappers = []


def create_mountain_car(
    num_sub_envs: int,
    seed: int,
    wrappers: list[Type[gym.vector.VectorWrapper | gym.Wrapper]] = [],
    is_vec: bool = True,
    vectorization_mode: str = "sync",
    goal_velocity: float = 0.1,
):
    if is_vec:
        envs = gym.make_vec(
            env_name,
            num_envs=num_sub_envs,
            vectorization_mode=vectorization_mode,
            goal_velocity=goal_velocity,
        )
    else:
        envs = gym.make(env_name, goal_velocity=goal_velocity)
    for wrapper in wrappers:
        envs = wrapper(envs)
    envs.reset(seed=seed)
    return envs


def create_mc_set(
    num_envs: int,
    num_sub_envs: int,
    seeds: list[int],
    wrappers: list[Type[gym.vector.VectorWrapper | gym.Wrapper]] = [],
    is_vec: bool = True,
    vectorization_mode: str = "sync",
    goal_velocity: float = 0.1,
) -> list[gym.vector.VectorEnv | gym.vector.VectorWrapper]:
    return [
        create_mountain_car(
            num_sub_envs,
            seeds[i],
            wrappers,
            is_vec,
            vectorization_mode,
            goal_velocity[i],
        )
        for i in range(num_envs)
    ]


def create_vectorized_mc_from_hparams(
    set_size: int,
    hparams_algorithm: dict,
    hparams_env: dict,
    key: jnp.ndarray,
    is_vec: bool = True,
    ignore_wrappers: bool = False,
    recurrent_model: bool = False,
):
    seeds = jax.random.randint(
        key,
        set_size,
        minval=1,
        maxval=1000,
    ).tolist()

    wrappers = vec_env_wrappers if is_vec else env_wrappers
    if recurrent_model:
        wrappers += [
            AppendHistoryVecEnvWrapper if is_vec else AppendHistoryWrapper
        ]

    return create_mc_set(
        set_size,
        hparams_algorithm["num_envs"],
        seeds,
        wrappers=[] if ignore_wrappers else wrappers,
        is_vec=is_vec,
        goal_velocity=hparams_env["goal_velocity"],
    )
