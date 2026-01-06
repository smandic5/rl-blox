from functools import partial
from typing import Callable

import gymnasium as gym
import jax
import jax.numpy as jnp

from rl_blox.algorithm.meta.maml_ppo import train_maml_ppo
from rl_blox.algorithm.meta.rl2_ppo import train_rl2_ppo
from rl_blox.logging.checkpointer import OrbaxCheckpointer
from rl_blox.logging.logger import AIMLogger, LoggerList, StandardLogger

from .helpers.cart_pole_factory import create_vectorized_cp_from_hparams
from .helpers.frozen_lake_factory import create_vectorized_fl_from_hparams
from .helpers.mountain_car_factory import create_vectorized_mc_from_hparams
from .helpers.policy_factory import create_policies
from .helpers.task_selector_configurations import get_ts_config
from .helpers.task_selector_factory import get_task_selector
from .hparams import (
    MAML_NAME,
    RL2_NAME,
    algorithm_to_use,
    checkpoint_path,
    hparams_algorithm,
    hparams_eval,
    hparams_model,
    params_cartpole,
    params_env,
    params_frozen_lake,
    params_mountain_car,
    save_frequency,
)

# ------------------- Env

create_vec_fl_env = partial(
    create_vectorized_fl_from_hparams,
    set_size=hparams_algorithm["set_size_train"],
    hparams_algorithm=hparams_algorithm,
    hparams_env=params_frozen_lake,
)
create_fl_env = partial(
    create_vectorized_fl_from_hparams,
    set_size=hparams_algorithm["set_size_train"],
    hparams_algorithm=hparams_algorithm,
    hparams_env=params_frozen_lake,
    is_vec=False,
)
create_vec_mc_env = partial(
    create_vectorized_mc_from_hparams,
    set_size=hparams_algorithm["set_size_train"],
    hparams_algorithm=hparams_algorithm,
    hparams_env=params_mountain_car,
)
create_mc_env = partial(
    create_vectorized_mc_from_hparams,
    set_size=hparams_algorithm["set_size_train"],
    hparams_algorithm=hparams_algorithm,
    hparams_env=params_mountain_car,
    is_vec=False,
)
create_vec_cp_env = partial(
    create_vectorized_cp_from_hparams,
    set_size=hparams_algorithm["set_size_train"],
    hparams_algorithm=hparams_algorithm,
    hparams_env=params_cartpole,
)
create_cp_env = partial(
    create_vectorized_cp_from_hparams,
    set_size=hparams_algorithm["set_size_train"],
    hparams_algorithm=hparams_algorithm,
    hparams_env=params_cartpole,
    is_vec=False,
)
if params_env == params_frozen_lake:
    create_vec_env = create_vec_fl_env
    create_env = create_fl_env
elif params_env == params_mountain_car:
    create_vec_env = create_vec_mc_env
    create_env = create_mc_env
elif params_env == params_cartpole:
    create_vec_env = create_vec_cp_env
    create_env = create_cp_env
else:
    raise Exception("Unreognized env params")


def get_num_features_actions(envs: gym.vector.VectorEnv) -> tuple[int, int]:
    assert type(envs.single_action_space) == gym.spaces.Discrete
    actions = int(envs.single_action_space.n)
    features = int(envs.single_observation_space.shape[0])
    return features, actions


# ------------------- Algorithm


def get_train_func(algorithm_name: str) -> Callable:
    if RL2_NAME in algorithm_name:
        train_func = train_rl2_ppo
    elif MAML_NAME in algorithm_name:
        train_func = train_maml_ppo
    else:
        raise Exception(f"Unknown algorithm: {algorithm_name}")
    return train_func


get_train_func = partial(get_train_func, algorithm_name=algorithm_to_use)

# ------------------- Policies

get_policies = partial(
    create_policies,
    is_recurrent=algorithm_to_use == RL2_NAME,
    hparams_model=hparams_model,
)

# ------------------- Task Selector

get_task_selector_config = partial(
    get_ts_config, max_reward=None, min_reward=None
)

# ------------------- Logger


def get_logger(algorithm_name: str):
    logger = LoggerList(
        [
            AIMLogger(),
            # StandardLogger(verbose=1),
            OrbaxCheckpointer(checkpoint_path),
        ]
    )
    logger.define_checkpoint_frequency(
        algorithm_name + "_ACTOR", save_frequency
    )
    logger.define_checkpoint_frequency(
        algorithm_name + "_CRITIC", save_frequency
    )
    return logger


# ------------------- Complete training setup


def prepare_training(hparams_task_selector, name, seed):
    prep_key = jax.random.key(seed)
    prep_key, subkey = jax.random.split(prep_key)

    vec_env_set = create_vec_env(key=subkey)
    env_set = create_env(key=subkey, ignore_wrappers=True)
    features, actions = get_num_features_actions(vec_env_set[0])

    prep_key, subkey = jax.random.split(prep_key)
    actor, critic, optimizer_actor, optimizer_critic = get_policies(
        features=features, actions=actions, key=subkey
    )

    prep_key, subkey = jax.random.split(prep_key)
    task_selector = get_task_selector(
        hparams_task_selector,
        hparams_algorithm,
        subkey,
        actor,
        env_set,
    )

    logger = get_logger(name)
    logger.define_experiment(
        env_name=params_env["env_name"],
        algorithm_name=name,
        hparams=hparams_model | hparams_algorithm | params_env | {"seed": seed},
    )

    train_func = get_train_func()

    return (
        prep_key,
        vec_env_set,
        actor,
        critic,
        optimizer_actor,
        optimizer_critic,
        task_selector,
        logger,
        train_func,
    )
