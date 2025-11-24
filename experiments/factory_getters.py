from functools import partial
from typing import Callable

import gymnasium as gym

from rl_blox.algorithm.meta.maml_ppo import train_maml_ppo
from rl_blox.algorithm.meta.rl2_ppo import train_rl2_ppo
from rl_blox.logging.checkpointer import OrbaxCheckpointer
from rl_blox.logging.logger import AIMLogger, LoggerList, StandardLogger

from .helpers.frozen_lake_factory import create_vectorized_fl_from_hparams
from .helpers.policy_factory import create_policies
from .helpers.task_selector_configurations import get_ts_config
from .hparams import (
    MAML_NAME,
    RL2_NAME,
    algorithm_to_use,
    checkpoint_path,
    hparams_algorithm,
    hparams_eval,
    hparams_model,
    params_frozen_lake,
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

create_vec_env = create_vec_fl_env
create_env = create_fl_env


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

max_reward = (
    (
        1 / (params_frozen_lake["lake_size"] - 1) ** 2
    )  # max potential reward per step (not realistic)
    * hparams_algorithm["batch_size"]
    * hparams_algorithm["num_envs"]  # max steps made
    * 0.3  # mulitplier
)
get_task_selector_config = partial(get_ts_config, max_reward=max_reward)

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


get_logger = partial(get_logger, algorithm_name=algorithm_to_use)
