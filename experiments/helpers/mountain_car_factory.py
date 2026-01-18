from typing import Type

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.envs.classic_control.mountain_car import MountainCarEnv

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
    goal_velocity: float = 9.8,
):
    gravity = goal_velocity / 400
    if is_vec:
        envs = gym.make_vec(
            env_name,
            num_envs=num_sub_envs,
            vectorization_mode=vectorization_mode,
            max_episode_steps=100,
        )
        envs = VecHeightRewardWrapper(envs)
        envs.unwrapped.set_attr("gravity", gravity)
    else:
        envs = gym.make(env_name, max_episode_steps=100)
        envs = HeightRewardWrapper(envs)
        envs.unwrapped.gravity = gravity
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


class HeightRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.mc: MountainCarEnv = env.unwrapped

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        height = self.mc._height(obs[0])

        # height = np.sin(3 * xs) * 0.45 + 0.55
        reward = ((height - 0.55) / 0.45 - 1) / 2

        return obs, reward, terminated, truncated, info


class VecHeightRewardWrapper(gym.vector.VectorWrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, actions):
        obs, _, terminations, truncations, infos = self.env.step(actions)

        positions = obs[:, 0]

        # _height is scalar, so vectorize it
        heights = np.array(
            [
                self.env.envs[i].unwrapped._height(positions[i])
                for i in range(self.num_envs)
            ]
        )

        rewards = ((heights - 0.55) / 0.45 - 1) / 2

        return obs, rewards, terminations, truncations, infos
