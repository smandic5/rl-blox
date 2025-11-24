import os

import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from experiments.similarity_trainer import (
    create_policies,
    get_logger,
    get_num_features_actions,
    get_task_selector,
)
from rl_blox.algorithm.meta.maml_ppo import train_maml_ppo
from rl_blox.algorithm.ppo import train_ppo
from rl_blox.blox.adaptation_metrics import (
    area_ratio,
    asymptotic_performance,
    jumpstart,
    time_to_treshold,
    total_reward,
)
from rl_blox.blox.env_util import OneHotObservationWrapper
from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.function_approximator.policy_head import SoftmaxPolicy
from rl_blox.blox.multitask import (
    HardTaskPrioritizationTaskSelector,
    UniformTaskSelector,
)
from rl_blox.blox.vec_env_util import OneHotVecObservationWrapper
from rl_blox.logging.logger import (  # AIMLogger,
    LoggerList,
    MemoryLogger,
    StandardLogger,
)

from .hparams import (
    MAML_NAME,
    RL2_NAME,
    algorithm_to_use,
    checkpoint_path,
    create_env,
    create_vec_env,
    hparams_algorithm,
    hparams_eval,
    hparams_model,
    params_env,
    save_frequency,
    selector_configurations,
)

jax.config.update("jax_platforms", "cpu")

saves_actor = [
    "FrozenLake-v1_MAML_PPO_0_1763383127.9423096_MAML_PPO_0_ACTOR_step_000000250_epoch_251",
    "FrozenLake-v1_MAML_PPO_0_1763383127.9423096_MAML_PPO_0_ACTOR_step_000000500_epoch_501",
    "FrozenLake-v1_MAML_PPO_0_1763383127.9423096_MAML_PPO_0_ACTOR_step_000000750_epoch_751",
    "FrozenLake-v1_MAML_PPO_1_1763389410.1435137_MAML_PPO_1_ACTOR_step_000000250_epoch_251",
    "FrozenLake-v1_MAML_PPO_1_1763389410.1435137_MAML_PPO_1_ACTOR_step_000000500_epoch_501",
    "FrozenLake-v1_MAML_PPO_1_1763389410.1435137_MAML_PPO_1_ACTOR_step_000000750_epoch_751",
    "FrozenLake-v1_MAML_PPO_2_1763396758.197202_MAML_PPO_2_ACTOR_step_000000250_epoch_251",
]
saves_critic = [
    "FrozenLake-v1_MAML_PPO_0_1763383127.9423096_MAML_PPO_0_CRITIC_step_000000250_epoch_251",
    "FrozenLake-v1_MAML_PPO_0_1763383127.9423096_MAML_PPO_0_CRITIC_step_000000500_epoch_501",
    "FrozenLake-v1_MAML_PPO_0_1763383127.9423096_MAML_PPO_0_CRITIC_step_000000750_epoch_751",
    "FrozenLake-v1_MAML_PPO_1_1763389410.1435137_MAML_PPO_1_CRITIC_step_000000250_epoch_251",
    "FrozenLake-v1_MAML_PPO_1_1763389410.1435137_MAML_PPO_1_CRITIC_step_000000500_epoch_501",
    "FrozenLake-v1_MAML_PPO_1_1763389410.1435137_MAML_PPO_1_CRITIC_step_000000750_epoch_751",
    "FrozenLake-v1_MAML_PPO_2_1763396758.197202_MAML_PPO_2_CRITIC_step_000000250_epoch_251",
]


def experiments_eval():
    for i, (a, c) in enumerate(zip(saves_actor, saves_critic)):
        eval_task_selector(a[:24] + f"_{i}", a, c)


def eval_task_selector(
    name: str, checkpoint_path_actor: str, checkpoint_path_critic: str
):
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
    load_weights(actor, checkpoint_path_actor)
    load_weights(critic, checkpoint_path_critic)

    for i in range(hparams_algorithm["set_size_test"]):

        # init logger
        memory_logger = MemoryLogger()
        logger = LoggerList(
            [
                memory_logger,
                # AIMLogger(),
                # StandardLogger(verbose=1),
            ]
        )
        logger.define_experiment(
            env_name=params_env["env_name"],
            algorithm_name=name + f"_eval_{i}",
            hparams=hparams_model
            | hparams_algorithm
            | hparams_eval
            | params_env,
        )

        # Adaptation
        actor_clone = nnx.clone(actor)
        critic_clone = nnx.clone(critic)
        optimizer_actor = nnx.Optimizer(
            actor_clone,
            optax.adam(hparams_model["actor_learning_rate"]),
            wrt=nnx.Param,
        )
        optimizer_critic = nnx.Optimizer(
            critic_clone,
            optax.adam(hparams_model["critic_learning_rate"]),
            wrt=nnx.Param,
        )
        envs = vec_env_set[hparams_algorithm["set_size_train"] + i]

        actor_clone, critic_clone, optimizer_actor, optimizer_critic = (
            train_ppo(
                envs,
                actor_clone,
                critic_clone,
                optimizer_actor,
                optimizer_critic,
                iterations=hparams_eval["iterations"],
                epochs=hparams_eval["epochs"],
                logger=logger,
                batch_size=hparams_algorithm["batch_size"],
            )
        )

        x, y = memory_logger.get_stat("return")

        print(f"{name} on task {i}")
        print(f"Jumpstart: {jumpstart(y, hparams_algorithm["batch_size"])}")
        print(
            f"Asymptotic Performance: {asymptotic_performance(y, hparams_algorithm["batch_size"])}"
        )
        print(f"Total Reward: {total_reward(y)}")

        del logger


def load_weights(model, name):
    checkpointer = ocp.StandardCheckpointer()
    path = f"{os.path.abspath(checkpoint_path)}\\{name}"
    print(os.listdir(path))
    state = checkpointer.restore(path, target=nnx.state(model))
    nnx.update(model, state)


if __name__ == "__main__":
    experiments_eval()
