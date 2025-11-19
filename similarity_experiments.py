from typing import Any, Callable

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from rl_blox.algorithm.meta.maml_ppo import train_maml_ppo
from rl_blox.algorithm.meta.rl2_ppo import train_rl2_ppo
from rl_blox.blox.env_util import AppendHistoryWrapper
from rl_blox.blox.vec_env_util import AppendHistoryVecEnvWrapper
from rl_blox.logging.checkpointer import OrbaxCheckpointer
from rl_blox.logging.logger import AIMLogger, LoggerList, StandardLogger
from sven_master.hparams import (
    MAML_NAME,
    RL2_NAME,
    algorithm_to_use,
    checkpoint_path,
    create_env,
    create_vec_env,
    hparams_algorithm,
    hparams_model,
    params_env,
    save_frequency,
    selector_configurations,
)
from sven_master.policy_factory import create_policies
from sven_master.task_selector_factory import create_task_selector

jax.config.update("jax_platforms", "cpu")


def experiments_train():
    for i, hparams_selector in enumerate(selector_configurations):
        algorithm_name = algorithm_to_use + f"_{i}"
        print(f"Training started for {algorithm_name}")
        test_task_selector(hparams_selector, algorithm_name)


def test_task_selector(hparams_task_selector: dict, name: str):
    # init key
    prep_key = jax.random.key(hparams_algorithm["seed"])
    prep_key, subkey = jax.random.split(prep_key)

    # init env
    vec_env_set = create_vec_env(key=subkey)
    env_set = create_env(key=subkey, ignore_wrappers=True)
    features, actions = get_num_features_actions(vec_env_set[0])

    # init models
    prep_key, subkey = jax.random.split(prep_key)
    actor, critic, optimizer_actor, optimizer_critic = create_policies(
        name == RL2_NAME, features, actions, hparams_model, subkey
    )

    # init task selector
    prep_key, subkey = jax.random.split(prep_key)
    task_selector = get_task_selector(
        hparams_task_selector,
        subkey,
        actor,
        env_set[: hparams_algorithm["set_size_train"]],
    )

    # init logger
    logger = get_logger(name)
    logger.define_experiment(
        env_name=params_env["env_name"],
        algorithm_name=name,
        hparams=hparams_model | hparams_algorithm | params_env,
    )

    # train
    train_func = get_train_func(name)
    actor, critic, optimizer_actor, optimizer_critic = train_func(
        vec_env_set[: hparams_algorithm["set_size_train"]],
        task_selector,
        actor,
        critic,
        optimizer_actor,
        optimizer_critic,
        iterations=hparams_algorithm["iterations"],
        epochs=hparams_algorithm["epochs"],
        logger=logger,
        batch_size=hparams_algorithm["batch_size"],
        seed=hparams_algorithm["seed"],
        agent_name=name,
    )


######### Helpers


def get_task_selector(hparams_task_selector, key, actor, envs):
    task_selector = create_task_selector(
        hparams_task_selector["selector_class"],
        np.arange(hparams_algorithm["set_size_train"]),
        key,
        envs,
        [nnx.clone(actor) for _ in range(hparams_algorithm["set_size_train"])],
        hparams_task_selector["prefer_similar"],
        hparams_task_selector["choose_from_last_pick"],
    )

    return task_selector


def get_num_features_actions(envs: gym.vector.VectorEnv) -> tuple[int, int]:
    assert type(envs.single_action_space) == gym.spaces.Discrete
    actions = int(envs.single_action_space.n)
    features = int(envs.single_observation_space.shape[0])
    return features, actions


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


def get_train_func(algorithm_name: str) -> Callable:
    if RL2_NAME in algorithm_name:
        train_func = train_rl2_ppo
    elif MAML_NAME in algorithm_name:
        train_func = train_maml_ppo
    else:
        raise Exception(f"Unknown algorithm: {algorithm_name}")
    return train_func


if __name__ == "__main__":
    experiments_train()
