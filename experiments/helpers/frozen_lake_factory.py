from typing import Type

import gymnasium as gym
import jax
import jax.numpy as jnp
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from rl_blox.blox.env_util import AppendHistoryWrapper, OneHotObservationWrapper
from rl_blox.blox.vec_env_util import (
    AppendHistoryVecEnvWrapper,
    OneHotVecObservationWrapper,
)

vec_env_wrappers = [OneHotVecObservationWrapper]
env_wrappers = [OneHotObservationWrapper]


def create_fl(
    num_sub_envs: int,
    seed: int,
    wrappers: list[Type[gym.vector.VectorWrapper]] = [],
    is_vec: bool = True,
    vectorization_mode: str = "sync",
    lake_size: int = 4,
    step_success_rate: float = 1.0,
    reward_goal: int = 1,
    reward_frozen: int = -0.025,
    reward_hole: int = -1,
) -> gym.vector.VectorEnv | gym.vector.VectorWrapper:
    random_map_gen = generate_random_map(size=lake_size, seed=seed)
    if is_vec:
        envs = gym.make_vec(
            "FrozenLake-v1",
            desc=random_map_gen,
            is_slippery=step_success_rate < 1,
            success_rate=step_success_rate,
            reward_schedule=(reward_goal, reward_frozen, reward_hole),
            num_envs=num_sub_envs,
            vectorization_mode=vectorization_mode,
        )
    else:
        envs = gym.make(
            "FrozenLake-v1",
            desc=random_map_gen,
        )
    for wrapper in wrappers:
        envs = wrapper(envs)
    return envs


def create_fl_set(
    num_envs: int,
    num_sub_envs: int,
    seeds: list[int],
    wrappers: list[Type[gym.vector.VectorWrapper]] = [],
    is_vec: bool = True,
    vectorization_mode: str = "sync",
    lake_size: int = 4,
    step_success_rate: float = 1.0,
    reward_goal: int = 1,
    reward_frozen: int = -0.025,
    reward_hole: int = -1,
) -> list[gym.vector.VectorEnv | gym.vector.VectorWrapper]:
    return [
        create_fl(
            num_sub_envs,
            seeds[i],
            wrappers,
            is_vec,
            vectorization_mode,
            lake_size,
            step_success_rate,
            reward_goal,
            reward_frozen,
            reward_hole,
        )
        for i in range(num_envs)
    ]


def create_vectorized_fl_from_hparams(
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

    return create_fl_set(
        set_size,
        hparams_algorithm["num_envs"],
        seeds,
        wrappers=[] if ignore_wrappers else wrappers,
        is_vec=is_vec,
        lake_size=hparams_env["lake_size"],
        step_success_rate=hparams_env["lake_size"],
        reward_goal=hparams_env["reward_goal"],
        reward_frozen=hparams_env["reward_frozen"],
        reward_hole=hparams_env["reward_hole"],
    )
