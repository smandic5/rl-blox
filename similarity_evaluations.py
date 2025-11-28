import os

import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx

from experiments.checkpoint_loader import (
    get_all_checkpoint_str,
    load_models,
    parse_checkpoint_str,
)
from experiments.factory_getters import create_vec_env, get_num_features_actions
from experiments.hparams import (
    hparams_algorithm,
    hparams_eval,
    hparams_model,
    params_env,
)
from rl_blox.algorithm.ppo import train_ppo
from rl_blox.blox.adaptation_metrics import (
    area_ratio,
    asymptotic_performance,
    jumpstart,
    time_to_treshold,
    total_reward,
)
from rl_blox.blox.vec_env_util import OneHotVecObservationWrapper
from rl_blox.logging.logger import (
    AIMLogger,
    LoggerList,
    MemoryLogger,
    StandardLogger,
)

jax.config.update("jax_platforms", "cpu")


def experiments_eval():
    for s in get_all_checkpoint_str():
        parsed_return = parse_checkpoint_str(s)
        if not parsed_return[4]:  # skip critic
            continue
        eval_task_selector(s, *parsed_return)


def eval_task_selector(
    dir_name: str,
    env: str,
    algo: str,
    ts: int,
    seed: int,
    is_actor: bool,
    step: int,
    epoch: int,
    alg_name: str,
):
    # init key
    prep_key = jax.random.key(seed + hparams_eval["seed_change_by"])
    prep_key, subkey = jax.random.split(prep_key)

    # init env
    vec_env_set = create_vec_env(key=subkey)
    features, actions = get_num_features_actions(vec_env_set[0])

    # init models
    prep_key, subkey = jax.random.split(prep_key)
    actor, critic = load_models(prep_key, features, actions, dir_name)

    for i in range(hparams_eval["set_size_test"]):

        memory_logger = MemoryLogger()
        logger = adapt(alg_name, vec_env_set, actor, critic, i, memory_logger)

        x, y = memory_logger.get_stat("return")
        js, ap, tr = calculate_eval_metrics(x, y, alg_name, i, epoch)

        del logger


def adapt(alg_name, vec_env_set, actor, critic, task_i, memory_logger):
    logger = LoggerList(
        [
            memory_logger,
            AIMLogger(),
            # StandardLogger(verbose=1),
        ]
    )
    logger.define_experiment(
        env_name=params_env["env_name"],
        algorithm_name=alg_name + f"_eval_{task_i}",
        hparams=hparams_model | hparams_algorithm | hparams_eval | params_env,
    )

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
    envs = vec_env_set[task_i]

    actor_clone, critic_clone, optimizer_actor, optimizer_critic = train_ppo(
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

    return logger


def calculate_eval_metrics(x, y, name, task_i, epoch):
    js = jumpstart(y, hparams_algorithm["batch_size"])
    ap = asymptotic_performance(y, hparams_algorithm["batch_size"])
    tr = total_reward(y)

    print(f"{name} on task {task_i} from epoch {epoch}")
    print(f"Jumpstart: {js}")
    print(f"Asymptotic Performance: {ap}")
    print(f"Total Reward: {tr}")

    return js, ap, tr


if __name__ == "__main__":
    experiments_eval()
