from functools import partial
from typing import Callable

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from .helpers.half_cheetah_factory import create_half_cheetah
from .hparams import (
    hparams_algorithm,
    hparams_backbone,
    params_cartpole,
    params_env,
    params_frozen_lake,
    params_half_cheetah,
    params_inverted_pendulum,
    params_mountain_car,
    params_pendulum,
)

if params_env == params_half_cheetah:
    make_env = create_half_cheetah
else:
    raise Exception("Unreognized env params")


def get_env_factory(
    env_id: str,
    max_episode_steps: int = 1000,
    idx: int = -1,
    capture_video: bool = False,
    run_name: str = None,
    gamma: float = 0.99,
) -> gym.Env:
    if capture_video and idx == 0:
        env = gym.make(
            env_id, render_mode="rgb_array", max_episode_steps=max_episode_steps
        )
        if run_name is None:
            run_name = env_id
        env = gym.wrappers.RecordVideo(
            env,
            f"videos/{run_name}",
            episode_trigger=lambda i: True,
            step_trigger=lambda i: True,
        )
    else:
        env = gym.make(env_id, max_episode_steps=max_episode_steps)
    env = gym.wrappers.FlattenObservation(
        env
    )  # deal with dm_control's Dict observation space
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    # env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env


get_env_factory = partial(
    get_env_factory,
    env_id=params_env["env_name"],
    max_episode_steps=params_env["max_episode_steps"],
)

make_env = partial(make_env, make_func=get_env_factory)


def make_vec_env(
    key: jnp.ndarray,
    num_envs: int,
    params: dict,
    env_factory: Callable[..., gym.Env],
) -> gym.vector.SyncVectorEnv:
    envs = []
    for i in range(num_envs):
        envs.append(partial(env_factory, **params, idx=i))
    vec_env = gym.vector.SyncVectorEnv(envs)
    seeds = jax.random.randint(
        key,
        num_envs,
        minval=1,
        maxval=1000,
    ).tolist()
    for seed, env in zip(seeds, vec_env.envs):
        env.reset(seed=seed)
    return vec_env


make_vec_env = partial(
    make_vec_env,
    num_envs=hparams_backbone["num_envs"],
    params=params_env,
    env_factory=make_env,
)


def make_vec_env_set(
    key: jnp.ndarray,
    set_size: int,
    params: dict,
    vec_env_func: Callable[[jnp.ndarray, dict], gym.Env],
) -> list[gym.vector.SyncVectorEnv]:
    subkeys = jax.random.split(key, set_size)

    env_set = []
    for i in range(set_size):
        params_per_env = dict()
        for k, v in params.items():
            if k in ["env_name"]:
                continue
            params_per_env[k] = v[i] if type(v) == list else v
        env_set.append(vec_env_func(subkeys[i], params=params_per_env))
    return env_set


make_vec_env_set = partial(
    make_vec_env_set,
    set_size=hparams_algorithm["train_set_size"],
    params=params_env,
    vec_env_func=make_vec_env,
)
