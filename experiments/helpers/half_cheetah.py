from typing import Type

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.envs.classic_control.mountain_car import MountainCarEnv

from experiments.hparams import (
    hparams_algorithm,
    hparams_eval,
    hparams_model,
    params_env,
)
from rl_blox.blox.env_util import AppendHistoryWrapper
from rl_blox.blox.vec_env_util import AppendHistoryVecEnvWrapper

env_name = "HalfCheetah-v5"

vec_env_wrappers = [
    gym.wrappers.vector.NormalizeObservation,
    gym.wrappers.vector.ClipAction,
]  # , gym.wrappers.vector.NormalizeReward
env_wrappers = []


def create_half_cheetah(
    num_sub_envs: int,
    seed: int,
    wrappers: list[Type[gym.vector.VectorWrapper | gym.Wrapper]] = [],
    is_vec: bool = True,
    vectorization_mode: str = "sync",
    gravity: float = 9.8,
):
    if is_vec:
        envs = gym.make_vec(
            env_name,
            num_envs=num_sub_envs,
            vectorization_mode=vectorization_mode,
            max_episode_steps=1000,
        )
        envs.unwrapped.set_attr("gravity", gravity)
    else:
        envs = gym.make(env_name, max_episode_steps=1000)
        envs.unwrapped.gravity = gravity
    for wrapper in wrappers:
        envs = wrapper(envs)
    envs.reset(seed=seed)
    return envs


def create_hc_set(
    num_envs: int,
    num_sub_envs: int,
    seeds: list[int],
    wrappers: list[Type[gym.vector.VectorWrapper | gym.Wrapper]] = [],
    is_vec: bool = True,
    vectorization_mode: str = "sync",
    gravity: float = 0.1,
) -> list[gym.vector.VectorEnv | gym.vector.VectorWrapper]:
    return [
        create_half_cheetah(
            num_sub_envs,
            seeds[i],
            wrappers,
            is_vec,
            vectorization_mode,
            gravity[i],
        )
        for i in range(num_envs)
    ]


def create_vectorized_hc_from_hparams(
    set_size: int,
    hparams_algorithm: dict,
    hparams_env: dict,
    key: jnp.ndarray,
    is_vec: bool = True,
    ignore_wrappers: bool = False,
    recurrent_model: bool = False,
):

    print("creating half cheetah")

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

    return create_hc_set(
        set_size,
        hparams_algorithm["num_envs"],
        seeds,
        wrappers=[] if ignore_wrappers else wrappers,
        is_vec=is_vec,
        gravity=hparams_env["gravity"],
    )
