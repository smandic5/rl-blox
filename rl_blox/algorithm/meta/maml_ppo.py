import os
from collections import namedtuple
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import psutil
import tensorflow_probability.substrates.jax.distributions as dist
from flax import nnx
from tqdm.rich import trange

from ...blox.function_approximator.policy_head import StochasticPolicyBase
from ...blox.gae import compute_gae
from ...blox.multitask import TaskSelector, WeightedTaskSelector
from ...logging.logger import LoggerBase, MemoryLogger
from ..ppo import collect_trajectories, ppo_loss, train_ppo, update_ppo

_ADAPTING_METRICS = ["average_return", "average_success"]
_ADAPTED_METRICS = ["return", "success"]


def log_return_success_loss(
    logger: LoggerBase,
    logger_results: MemoryError,
    iteration: int,
    suffix: str = "while_adapting",
) -> tuple[float, float]:
    res = []
    adapting = suffix == "while_adapting"
    for metric_name in _ADAPTING_METRICS if adapting else _ADAPTED_METRICS:
        x, y = logger_results.get_stat(metric_name)
        value = y[-1] if metric_name == "loss" else jnp.average(y)
        logger.record_stat(
            f"{"average_" if not adapting else ""}{metric_name}_{suffix}",
            value,
            step=iteration,
        )
        res.append(value)
    return res


def inner_loop(
    envs: gym.vector.VectorEnv,
    actor: StochasticPolicyBase,
    critic: nnx.Module,
    optimizer_actor: nnx.Optimizer,
    optimizer_critic: nnx.Optimizer,
    key: jnp.ndarray,
    current_iteration: int,
    batch_size: int = 64,
    epochs: int = 1,
    logger: LoggerBase | None = None,
) -> tuple[float, float]:
    # adapt model
    logger_adapting = logger if logger is None else MemoryLogger()
    key, subkey = jax.random.split(key)
    actor, critic, optimizer_actor, optimizer_critic = train_ppo(
        envs,
        actor,
        critic,
        optimizer_actor,
        optimizer_critic,
        epochs,
        batch_size=batch_size,
        key=subkey,
        logger=logger_adapting,
        progress_bar=False,
    )
    if logger is not None:
        log_return_success_loss(logger, logger_adapting, current_iteration)

    # collect trajectory with adapted model
    logger_adapted = logger if logger is None else MemoryLogger()
    key, subkey = jax.random.split(key)
    (
        observation,
        action,
        reward,
        terminated,
        next_value,
        _,
        _,
    ) = collect_trajectories(
        envs,
        actor,
        critic,
        subkey,
        batch_size,
        logger_adapted,
        reach_batch_size=True,
    )
    if logger is not None:
        ret, success = log_return_success_loss(
            logger, logger_adapted, current_iteration, "after_adapting"
        )

    # calc loss
    advs, returns = compute_gae(
        reward, critic(observation).flatten(), next_value, terminated
    )
    logp = actor.log_probability(observation, action)
    adapted_loss_val = ppo_loss(
        actor, critic, logp, observation, action, advs, returns
    )

    return adapted_loss_val, (ret, success)


loss_grad_fn = nnx.value_and_grad(inner_loop, argnums=(1, 2), has_aux=True)


def print_memory_usage(i=None):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss  # Resident Set Size in bytes
    print(f"{i};{' '*(30 - len(i))} Memory usage: {mem / (1024 ** 2):.2f} MB")


def train_maml_ppo(
    env_set: list[gym.vector.VectorEnv],
    task_selector: TaskSelector,
    actor: StochasticPolicyBase,
    critic: nnx.Module,
    optimizer_actor: nnx.Optimizer,
    optimizer_critic: nnx.Optimizer,
    iterations: int = 1000,
    batch_size: int = 64,
    epochs: int = 1,
    seed: int = 1,
    inner_actor_lr: float = 0.001,
    inner_critic_lr: float = 0.001,
    logger: LoggerBase | None = None,
    progress_bar: bool = True,
    agent_name: str = "MAML_PPO",
) -> tuple[StochasticPolicyBase, nnx.Module, nnx.Optimizer, nnx.Optimizer]:
    key = jax.random.key(seed)
    for i, envs in enumerate(env_set):
        envs.reset(seed=seed)
        env_set[i] = gym.wrappers.vector.RecordEpisodeStatistics(envs)

    is_weighted_ts = isinstance(task_selector, WeightedTaskSelector)
    optimizer_actor_inner = nnx.Optimizer(
        actor, optax.rprop(inner_actor_lr / 10), wrt=nnx.Param
    )
    optimizer_critic_inner = nnx.Optimizer(
        critic, optax.rprop(inner_critic_lr / 10), wrt=nnx.Param
    )

    for iteration in trange(iterations, disable=not progress_bar):
        print_memory_usage(f"Iteration: {iteration}")
        task_id = task_selector.select()
        envs = env_set[task_id]
        envs.reset()
        if logger is not None:
            logger.record_stat("chosen_environment", task_id, step=iteration)
            if is_weighted_ts:
                for i, w in enumerate(task_selector.weights):
                    logger.record_stat(f"env_{i}_prob", w, step=iteration)

        actor_clone = nnx.clone(actor)
        critic_clone = nnx.clone(critic)

        (meta_loss, (reward, success_rate)), (grad_actor, grad_critic) = (
            loss_grad_fn(
                envs,
                actor_clone,
                critic_clone,
                optimizer_actor_inner,
                optimizer_critic_inner,
                key,
                iteration,
                batch_size=batch_size,
                epochs=epochs,
                logger=logger,
            )
        )

        optimizer_actor.update(actor, grad_actor)
        optimizer_critic.update(critic, grad_critic)

        task_selector.feedback(reward=success_rate, policy=actor_clone)
        if logger is not None:
            logger.record_stat("meta_loss", meta_loss, step=iteration)
            logger.record_stat(
                f"env_{task_id}_meta_loss", meta_loss, step=iteration
            )
            logger.record_stat(f"env_{task_id}_reward", reward, step=iteration)
            logger.record_stat(
                f"env_{task_id}_success_rate", success_rate, step=iteration
            )
            logger.record_epoch(f"{agent_name}_ACTOR", actor, step=iteration)
            logger.record_epoch(f"{agent_name}_CRITIC", critic, step=iteration)

    return actor, critic, optimizer_actor, optimizer_critic
