import gymnasium as gym
import numpy as np
from args import Args

from .vector_env import init_vec_envs


def init_envs_set(
    args: Args, run_name: str, set_size: int
) -> list[gym.vector.SyncVectorEnv]:
    velocities = np.random.uniform(
        args.target_velocity_min, args.target_velocity_max, set_size
    )
    print(f"Velocities: {[float(round(v, ndigits=2)) for v in velocities]}")
    return [
        init_vec_envs(args, run_name, target_velocity)
        for target_velocity in velocities
    ]


def init_train_envs_set(
    args: Args, run_name: str
) -> list[gym.vector.SyncVectorEnv]:
    return init_envs_set(args, run_name, args.train_set_size)


def init_test_envs_set(
    args: Args, run_name: str
) -> list[gym.vector.SyncVectorEnv]:
    return init_envs_set(args, run_name, args.test_set_size)


def init_env_sets(
    args: Args, run_name: str
) -> tuple[list[gym.vector.SyncVectorEnv], list[gym.vector.SyncVectorEnv]]:
    return init_train_envs_set(args, run_name), init_test_envs_set(
        args, run_name
    )
