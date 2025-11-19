from typing import Type

import gymnasium as gym
import jax
import jax.numpy as jnp
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


def create_fl(
    num_sub_envs: int,
    seed: int,
    wrappers: list[Type[gym.vector.VectorWrapper]] = [],
    is_vec: bool = True,
    vectorization_mode: str = "sync",
    lake_size: int = 4,
) -> gym.vector.VectorEnv | gym.vector.VectorWrapper:
    if is_vec:
        envs = gym.make_vec(
            "FrozenLake-v1",
            desc=generate_random_map(size=lake_size, seed=seed),
            num_envs=num_sub_envs,
            vectorization_mode=vectorization_mode,
        )
    else:
        envs = gym.make(
            "FrozenLake-v1",
            desc=generate_random_map(size=lake_size, seed=seed),
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
) -> list[gym.vector.VectorEnv | gym.vector.VectorWrapper]:
    return [
        create_fl(
            num_sub_envs,
            seeds[i],
            wrappers,
            is_vec,
            vectorization_mode,
            lake_size,
        )
        for i in range(num_envs)
    ]


def create_vectorized_fl_from_hparams(
    hparams_algorithm: dict,
    wrappers: list[gym.vector.VectorWrapper],
    key: jnp.ndarray,
    is_vec: bool = True,
    ignore_wrappers: bool = False,
    lake_size: int = 4,
):
    seeds = jax.random.randint(
        key,
        hparams_algorithm["set_size_train"]
        + hparams_algorithm["set_size_test"],
        minval=1,
        maxval=1000,
    ).tolist()

    return create_fl_set(
        hparams_algorithm["set_size_train"]
        + hparams_algorithm["set_size_test"],
        hparams_algorithm["num_envs"],
        seeds,
        wrappers=[] if ignore_wrappers else wrappers,
        is_vec=is_vec,
        lake_size=lake_size,
    )
