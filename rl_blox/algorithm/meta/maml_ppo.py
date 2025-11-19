from collections import namedtuple
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_probability.substrates.jax.distributions as dist
from flax import nnx
from tqdm.rich import trange

from ...blox.function_approximator.policy_head import StochasticPolicyBase
from ...blox.gae import compute_gae
from ...blox.multitask import TaskSelector
from ...logging.logger import LoggerBase, MemoryLogger
from ..ppo import collect_trajectories, ppo_loss, update_ppo


def inner_loop(
    envs: gym.vector.VectorEnv,
    actor: StochasticPolicyBase,
    critic: nnx.Module,
    key: jnp.ndarray,
    current_iteration: int,
    batch_size: int = 64,
    epochs: int = 1,
    inner_actor_lr: float = 0.001,
    inner_critic_lr: float = 0.001,
    logger: LoggerBase | None = None,
) -> tuple[float, float]:
    for i in range(epochs):
        # collect trajectory
        logger_adapting = logger if logger is None else MemoryLogger()
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
            envs, actor, critic, subkey, batch_size, logger_adapting
        )

        # adapt model
        optimizer_actor = nnx.Optimizer(
            actor, optax.rprop(inner_actor_lr), wrt=nnx.Param
        )
        optimizer_critic = nnx.Optimizer(
            critic, optax.rprop(inner_critic_lr), wrt=nnx.Param
        )
        loss_val = update_ppo(
            actor,
            critic,
            optimizer_actor,
            optimizer_critic,
            observation,
            action,
            reward,
            terminated,
            next_value,
            1,
        )
    total_episodes = len(logger_adapting.get_stat("return")[0]) + batch_size
    logger.record_stat(
        "reward_while_adapting",
        jnp.sum(reward).item() / total_episodes,
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
        envs, actor, critic, subkey, batch_size, logger_adapted
    )
    total_episodes = len(logger_adapted.get_stat("return")[0]) + batch_size
    logger.record_stat(
        "reward_after_adapting",
        jnp.sum(reward).item() / total_episodes,
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

    return adapted_loss_val, jnp.sum(reward).item()


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

    for iteration in trange(iterations, disable=not progress_bar):
        task_id = task_selector.select()
        envs = env_set[task_id]
        envs.reset()

        actor_clone = nnx.clone(actor)
        critic_clone = nnx.clone(critic)

        loss_grad_fn = nnx.value_and_grad(
            inner_loop, argnums=(1, 2), has_aux=True
        )
        (meta_loss, reward), (grad_actor, grad_critic) = loss_grad_fn(
            envs,
            actor_clone,
            critic_clone,
            key,
            iteration,
            batch_size=batch_size,
            epochs=epochs,
            inner_actor_lr=inner_actor_lr,
            inner_critic_lr=inner_critic_lr,
            logger=logger,
        )

        optimizer_actor.update(actor, grad_actor)
        optimizer_critic.update(critic, grad_critic)

        task_selector.feedback(reward=reward, policy=actor_clone)
        if logger is not None:
            logger.record_stat("meta_loss", meta_loss, step=iteration)
            logger.record_epoch(f"{agent_name}_ACTOR", actor, step=iteration)
            logger.record_epoch(f"{agent_name}_CRITIC", critic, step=iteration)

    return actor, critic, optimizer_actor, optimizer_critic
