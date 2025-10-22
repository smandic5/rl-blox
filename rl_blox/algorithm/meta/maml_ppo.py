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
from ...logging.logger import LoggerBase
from ..ppo import collect_trajectories, ppo_loss, update_ppo


def inner_loop(
    envs: gym.vector.VectorEnv,
    actor: StochasticPolicyBase,
    critic: nnx.Module,
    key: jnp.ndarray,
    batch_size: int = 64,
    inner_actor_lr: float = 0.001,
    inner_critic_lr: float = 0.001,
    logger: LoggerBase | None = None,
):
    update_ppo_jitted = nnx.jit(update_ppo, static_argnames="epochs")

    # collect trajectory
    key, subkey = jax.random.split(key)
    (
        observation,
        action,
        reward,
        terminated,
        next_value,
        _,
        _,
    ) = collect_trajectories(envs, actor, critic, subkey, batch_size, logger)

    # adapt model
    optimizer_actor = nnx.Optimizer(
        actor, optax.rprop(inner_actor_lr), wrt=nnx.Param
    )
    optimizer_critic = nnx.Optimizer(
        critic, optax.rprop(inner_critic_lr), wrt=nnx.Param
    )
    loss_val = update_ppo_jitted(
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

    # collect trajectory with adapted model
    key, subkey = jax.random.split(key)
    (
        observation,
        action,
        reward,
        terminated,
        next_value,
        _,
        _,
    ) = collect_trajectories(envs, actor, critic, subkey, batch_size, logger)

    # calc loss
    advs, returns = compute_gae(
        reward, critic(observation).flatten(), next_value, terminated
    )
    logp = actor.log_probability(observation, action)
    adapted_loss_val = ppo_loss(
        actor, critic, logp, observation, action, advs, returns
    )

    return adapted_loss_val


def train_maml_ppo(
    envs: gym.vector.VectorEnv,
    actor: StochasticPolicyBase,
    critic: nnx.Module,
    optimizer_actor: nnx.Optimizer,
    optimizer_critic: nnx.Optimizer,
    iterations: int = 1000,
    batch_size: int = 64,
    seed: int = 1,
    inner_actor_lr: float = 0.001,
    inner_critic_lr: float = 0.001,
    logger: LoggerBase | None = None,
    progress_bar: bool = True,
) -> tuple[StochasticPolicyBase, nnx.Module, nnx.Optimizer, nnx.Optimizer]:
    # init vars
    key = jax.random.key(seed)
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)
    assert (
        envs.metadata["autoreset_mode"] == gym.vector.AutoresetMode.SAME_STEP
    ), "Vectorized Env has to be instantiated with the SAME_STEP autoreset mode."

    # loop
    for iteration in trange(iterations, disable=not progress_bar):
        # sample env
        # TODO sampling with vec envs

        # clone base model
        actor_clone = nnx.clone(actor)
        critic_clone = nnx.clone(critic)

        # inner loop
        loss_grad_fn = nnx.value_and_grad(inner_loop, argnums=(1, 2))
        meta_loss, (grad_actor, grad_critic) = loss_grad_fn(
            envs,
            actor_clone,
            critic_clone,
            key,
            batch_size,
            inner_actor_lr,
            inner_critic_lr,
            logger,
        )

        # update meta models
        optimizer_actor.update(actor, grad_actor)
        optimizer_critic.update(critic, grad_critic)

        if logger is not None:
            logger.record_stat("meta_loss", meta_loss, step=iteration)

    return actor, critic, optimizer_actor, optimizer_critic
