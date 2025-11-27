import sys

import jax

from experiments.factory_getters import (
    create_env,
    create_vec_env,
    get_logger,
    get_num_features_actions,
    get_policies,
    get_task_selector_config,
    get_train_func,
)
from experiments.helpers.task_selector_factory import get_task_selector
from experiments.hparams import (
    algorithm_to_use,
    hparams_algorithm,
    hparams_model,
    params_env,
)

jax.config.update("jax_platforms", "cpu")


def train_with_task_selector(hparams_task_selector: dict, name: str):
    # init key
    prep_key = jax.random.key(hparams_algorithm["seed"])
    prep_key, subkey = jax.random.split(prep_key)

    # init env
    vec_env_set = create_vec_env(key=subkey)
    env_set = create_env(key=subkey, ignore_wrappers=True)
    features, actions = get_num_features_actions(vec_env_set[0])

    # init models
    prep_key, subkey = jax.random.split(prep_key)
    actor, critic, optimizer_actor, optimizer_critic = get_policies(
        features=features, actions=actions, key=subkey
    )

    # init task selector
    prep_key, subkey = jax.random.split(prep_key)
    task_selector = get_task_selector(
        hparams_task_selector,
        hparams_algorithm,
        subkey,
        actor,
        env_set,
    )

    # init logger
    logger = get_logger()
    logger.define_experiment(
        env_name=params_env["env_name"],
        algorithm_name=name,
        hparams=hparams_model | hparams_algorithm | params_env,
    )

    # train
    train_func = get_train_func()
    actor, critic, optimizer_actor, optimizer_critic = train_func(
        vec_env_set,
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


def experiments_train(task_selector_index: int):
    hparams_selector = get_task_selector_config(task_selector_index)
    algorithm_name = algorithm_to_use + f"_{task_selector_index}"
    print(f"Training started for {algorithm_name}")
    train_with_task_selector(hparams_selector, algorithm_name)


if __name__ == "__main__":
    index = int(sys.argv[1])
    experiments_train(index)
