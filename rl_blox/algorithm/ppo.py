import os
from collections import namedtuple
from functools import partial
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import psutil
from flax import nnx
from tqdm.rich import trange

from ..blox.function_approximator.policy_head import StochasticPolicyBase
from ..blox.gae import compute_gae
from ..blox.vec_env_util import TrajectoryCollector
from ..logging.logger import LoggerBase, LoggerList, MemoryLogger


def collect_trajectories(
    envs: gym.vector.VectorEnv,
    actor: StochasticPolicyBase,
    critic: nnx.Module,
    key: jnp.ndarray,
    batch_size: int = 64,
    logger: LoggerBase | None = None,
    last_observation=None,
    global_step: int = 0,
    reach_batch_size: bool = True,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    Any,
    int,
]:
    """
    Run and collect trajectories until at least `batch_size` steps are gathered.

    Parameters
    ----------
    envs : gym.vector.VectorEnv
        The vectorized environment to interact with.
    actor : StochasticPolicyBase
        The actor network.
    critic : nnx.Module
        The critic network.
    key : jnp.ndarray
        Random key.
    batch_size : int, optional
        Minimum number of steps to collect.
    logger : LoggerBase, optional
        Experiment Logger.
    last_observation : Any, optional
        Last observation produced by the environment. Used for running
        an environment over multiple calls of this function.
    global_step : int, optional
        Global step count

    Returns
    -------
    observation : jnp.ndarray
        Array of observations.
    action : jnp.ndarray
        Actions taken per step.
    reward : jnp.ndarray
        Array of rewards per step.
    terminated : jnp.ndarray
        Flags indicating episode termination per step.
    next_value : jnp.ndarray
        Array of predicted values for next steps per step.
    last_observation
        Last observation produced by the environment. Used for running
        an environment over multiple calls of this function.
    global_step : int, optional
        Global step count
    """

    trajectory_collector = TrajectoryCollector(
        envs.num_envs,
        batch_size * 2,
        envs.single_observation_space.shape,
        envs.single_action_space.shape,
        save_hidden_states=False,
    )
    obs = envs.reset()[0] if last_observation is None else last_observation

    subkeys = jax.random.split(key, batch_size)
    for i in range(batch_size * 2):
        action = actor.sample(obs, subkeys[i])
        value = critic(obs)
        next_obs, reward, terminated, truncated, info = envs.step(
            np.asarray(action)
        )
        trajectory_collector.update(
            obs,
            action,
            value,
            reward,
            terminated,
            truncated,
        )

        if "episode" in info.keys():
            finished_reward_len_obs = [
                (index, r, l)
                for index, (r, l, f) in enumerate(
                    zip(
                        info["episode"]["r"],
                        info["episode"]["l"],
                        info["_episode"],
                        strict=True,
                    )
                )
                if f
            ]
            for i, (index, r, l) in enumerate(finished_reward_len_obs):
                global_step += int(l)
                if logger is not None:
                    logger.record_stat("return", float(r), step=global_step)
                    logger.record_stat(
                        "success", reward[index] == 1.0, step=global_step
                    )
                    logger.start_new_episode()

        obs = next_obs
        reached_size = (
            trajectory_collector.current_batch_size()
            >= batch_size * envs.num_envs
        )
        if reached_size or (not reach_batch_size and i >= batch_size):
            break

    value = critic(obs)
    batch = trajectory_collector.get_batch(
        value, batch_size * envs.num_envs if reach_batch_size else None
    )

    return namedtuple(
        "PPO_Trajectory",
        [
            "observation",
            "action",
            "reward",
            "terminated",
            "next_value",
            "last_observation",
            "global_step",
        ],
    )(
        batch[0],
        batch[1],
        batch[2],
        batch[3],
        # batch[4], value is not returned
        batch[5],
        obs,
        global_step,
    )


def ppo_loss(
    actor: StochasticPolicyBase,
    critic: nnx.Module,
    old_logps: jnp.ndarray,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
    clip: float = 0.2,
) -> jnp.ndarray:
    """
    Calculate the PPO loss.

    Parameters
    ----------
    actor : StochasticPolicyBase
        The actor network.
    critic : nnx.Module
        The critic network.
    old_logps : jnp.ndarray
        Log probabilities of actions calculated during rollout.
    observations : jnp.ndarray
        Batch of observations.
    actions : jnp.ndarray
        Actions taken in each observation.
    advantages : jnp.ndarray
        Estimated advantages for each action.
    returns : jnp.ndarray
        Computed returns.
    clip : float, optional
        Clipping range for the PPO objective.

    Returns
    -------
    loss : jnp.ndarray
        The computed PPO loss for the batch.
    """
    logps = actor.log_probability(observations, actions)
    ratios = jnp.exp(logps - old_logps)
    surrogate1 = ratios * advantages
    surrogate2 = jnp.clip(ratios, 1 - clip, 1 + clip) * advantages
    policy_loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))

    values = critic(observations)
    value_loss = jnp.mean((returns - values) ** 2)

    return (
        policy_loss
        + 0.5 * value_loss
        - 0.01 * actor.entropy(observations).mean()
    )


loss_grad_fn = nnx.value_and_grad(ppo_loss, argnums=(0, 1))


@partial(nnx.jit, static_argnames="epochs")
def update_ppo(
    actor: StochasticPolicyBase,
    critic: nnx.Module,
    optimizer_actor: nnx.Optimizer,
    optimizer_critic: nnx.Optimizer,
    observation: jnp.ndarray,
    action: jnp.ndarray,
    reward: jnp.ndarray,
    terminated: jnp.ndarray,
    next_value: jnp.ndarray,
    epochs: int = 1,
) -> jnp.ndarray:
    """Updates the PPO agent.

    Arguments
    ---------
    actor : StochasticPolicyBase
        The actor network
    critic : nnx.Module
        The critic network
    observation : jnp.ndarray
        Array of observations.
    action : jnp.ndarray
        Actions taken per step.
    reward : jnp.ndarray
        Array of rewards per step.
    terminated : jnp.ndarray
        Flags indicating episode termination per step.
    next_value : jnp.ndarray
        Array of predicted next_values per step.
    epochs : int, optional
        Number of training epochs.

    Returns
    -------
    loss_val : jnp.ndarray
        Calculated loss.
    """
    advs, returns = compute_gae(
        reward, critic(observation).flatten(), next_value, terminated
    )
    logp = actor.log_probability(observation, action)

    for _ in range(epochs):
        (loss_val), (grad_actor, grad_critic) = loss_grad_fn(
            actor, critic, logp, observation, action, advs, returns
        )
        optimizer_actor.update(actor, grad_actor)
        optimizer_critic.update(critic, grad_critic)

    return loss_val


def train_ppo(
    envs: gym.vector.VectorEnv,
    actor: StochasticPolicyBase,
    critic: nnx.Module,
    optimizer_actor: nnx.Optimizer,
    optimizer_critic: nnx.Optimizer,
    iterations: int = 3000,
    epochs: int = 1,
    batch_size: int = 64,
    seed: int = 1,
    key: jnp.ndarray = None,
    logger: LoggerBase | None = None,
    progress_bar: bool = True,
) -> tuple[StochasticPolicyBase, nnx.Module, nnx.Optimizer, nnx.Optimizer]:
    """
    Train a PPO agent.

    Parameters
    ----------
    envs : gym.vector.VectorEnv
        The vectorized training environment.
    actor : StochasticPolicyBase
        The actor network.
    critic : nnx.Module
        The critic network.
    optimizer_actor : nnx.Optimizer
        Optimizer for the actor network.
    optimizer_critic : nnx.Optimizer
        Optimizer for the critic network.
    iterations : int, optional
        Number of training iterations.
    epochs : int, optional
        Number of training epochs per iteration.
    batch_size : int, optional
        Batch size per update.
    seed : int, optional
        Random seed for reproducibility.
    logger : LoggerBase, optional
        Experiment Logger.
    progress_bar : bool, optional
        Display a progress bar during training.

    Returns
    -------
    actor : StochasticPolicyBase
        Trained actor network.
    critic : nnx.Module
        Trained critic network.
    optimizer_actor : nnx.Optimizer
        Updated actor optimizer.
    optimizer_critic : nnx.Optimizer
        Updated critic optimizer.
    """
    if key == None:
        key = jax.random.key(seed)
        envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)

    last_observation, _ = envs.reset(seed=seed) if key == None else envs.reset()
    if logger is not None:
        logger.start_new_episode()

    global_step = 0
    list_logger = None
    for iteration in trange(iterations, disable=not progress_bar):
        if logger != None:
            mem_logger = MemoryLogger()
            list_logger = LoggerList([mem_logger, logger])
        key, subkey = jax.random.split(key)
        (
            observation,
            action,
            reward,
            terminated,
            next_value,
            last_observation,
            global_step,
        ) = collect_trajectories(
            envs,
            actor,
            critic,
            subkey,
            batch_size,
            list_logger,
            last_observation,
            global_step,
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
            epochs,
        )

        if logger is not None:
            logger.record_stat("loss", loss_val.item(), step=iteration)
            for metric_name in ["return", "success"]:
                x, y = mem_logger.get_stat(metric_name)
                logger.record_stat(
                    f"average_{metric_name}", jnp.average(y), step=iteration
                )

    return actor, critic, optimizer_actor, optimizer_critic
