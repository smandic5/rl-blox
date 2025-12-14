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
        for metric_name in ["average_return", "average_success", "loss"]:
            x, y = logger_adapting.get_stat(metric_name)
            value = y[-1] if metric_name == "loss" else jnp.average(y)
            logger.record_stat(
                f"{metric_name}_while_adapting",
                value,
                step=current_iteration,
            )

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
        total_episodes = len(logger_adapted.get_stat("return")[0]) + batch_size
        logger.record_stat(
            "average_return_after_adapting",
            jnp.sum(reward).item() / total_episodes,
            step=current_iteration,
        )
        _, success = logger_adapted.get_stat("success")
        success_rate = sum(success) / len(success)
        logger.record_stat(
            "average_success_after_adapting",
            success_rate,
            step=current_iteration,
        )

    # calc loss
    advs, returns = compute_gae(
        reward, critic(observation).flatten(), next_value, terminated
    )
    logp = actor.log_probability(observation, action)
    adapted_loss_val = ppo_loss(
        actor, critic, logp, observation, action, advs, returns
    )

    return adapted_loss_val, (jnp.average(reward).item(), success_rate)


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
        actor, optax.rprop(inner_actor_lr), wrt=nnx.Param
    )
    optimizer_critic_inner = nnx.Optimizer(
        critic, optax.rprop(inner_critic_lr), wrt=nnx.Param
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
