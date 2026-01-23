import sys

import jax

from experiments.factory_getters import (
    get_task_selector_config,
    prepare_training,
)
from experiments.hparams import (
    algorithm_to_use,
    hparams_algorithm,
    hparams_backbone,
    hparams_eval,
    hparams_model,
    params_env,
)
from rl_blox.algorithm.ppo import train_ppo

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

    envs = vec_env_set[0]

    # train
    actor_clone, critic_clone, optimizer_actor, optimizer_critic = train_ppo(
        envs,
        actor,
        critic,
        optimizer_actor,
        optimizer_critic,
        iterations=hparams_eval["iterations"],
        epochs=hparams_backbone["backbone_epochs"],
        rollout_length=hparams_backbone["rollout_length"],
        batch_size=hparams_backbone["batch_size"],
        seed=1,
        logger=logger,
    )

    envs = vec_env_set[0].close()


def experiments_train():
    hparams_selector = get_task_selector_config(0)
    algorithm_name = algorithm_to_use + f"_{-1}_{0}"
    print(f"Training started for {algorithm_name}")
    train_with_task_selector(hparams_selector, algorithm_name, 0)


if __name__ == "__main__":
    experiments_train()
