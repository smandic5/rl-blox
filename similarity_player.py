import sys

import jax

from experiments.factory_getters import (
    get_task_selector_config,
    prepare_training,
)
from experiments.hparams import (
    algorithm_to_use,
    hparams_algorithm,
    hparams_eval,
    hparams_model,
    params_env,
)
from rl_blox.algorithm.ppo import train_ppo

jax.config.update("jax_platforms", "cpu")


def train_with_task_selector(hparams_task_selector: dict, name: str, seed: int):

    import gymnasium as gym
    from gymnasium.utils import play

    from experiments.factory_getters import create_env

    prep_key = jax.random.key(1)
    prep_key, subkey = jax.random.split(prep_key)

    env = create_env(key=subkey, ignore_wrappers=True)[5]
    env = play.play(env, zoom=1, keys_to_action={"2": 2, "1": 0}, noop=1)


def experiments_train():
    hparams_selector = get_task_selector_config(0)
    algorithm_name = algorithm_to_use + f"_{-1}_{0}"
    print(f"Training started for {algorithm_name}")
    train_with_task_selector(hparams_selector, algorithm_name, 0)


if __name__ == "__main__":
    experiments_train()
