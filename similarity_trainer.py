import sys

import jax

from experiments.factory_getters import (
    get_task_selector_config,
    prepare_training,
)
from experiments.hparams import (
    algorithm_to_use,
    hparams_algorithm,
    hparams_model,
    params_env,
)

jax.config.update("jax_platforms", "cpu")


def train_with_task_selector(hparams_task_selector: dict, name: str, seed: int):
    (
        prep_key,
        vec_env_set,
        actor,
        critic,
        optimizer_actor,
        optimizer_critic,
        task_selector,
        logger,
        train_func,
    ) = prepare_training(hparams_task_selector, name, seed)

    # train
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
        seed=seed,
        agent_name=name,
    )


def experiments_train(task_selector_index: int):
    hparams_selector = get_task_selector_config(task_selector_index)
    for seed in range(5):
        algorithm_name = algorithm_to_use + f"_{task_selector_index}_{seed}"
        print(f"Training started for {algorithm_name}")
        train_with_task_selector(hparams_selector, algorithm_name, seed)


if __name__ == "__main__":
    index = int(sys.argv[1])
    if len(sys.argv) >= 3:
        index += int(sys.argv[2])
    experiments_train(index)
