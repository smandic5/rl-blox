from collections import namedtuple
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.distributions as dist
from flax import nnx
from tqdm.rich import trange

from ...blox.function_approximator.recurrent_policy_head import (
    StochasticRecurrentPolicyBase,
)
from ...blox.function_approximator.rnn import RNN
from ...blox.gae import compute_gae
from ...blox.multitask import TaskSelector
from ...blox.vec_env_util import TrajectoryCollector
from ...logging.logger import LoggerBase


def collect_trajectories(
    envs: gym.wrappers.vector.RecordEpisodeStatistics,
    actor: StochasticRecurrentPolicyBase,
    critic: RNN,
    hidden_state_actor: jnp.ndarray,
    hidden_state_critic: jnp.ndarray,
    key: jnp.ndarray,
    batch_size: int = 64,
    logger: LoggerBase | None = None,
    last_observation=None,
    global_step: int = 0,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
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
    actor : StochasticRecurrentPolicyBase
        The actor network.
    critic : RNN
        The critic network.
    hidden_state_actor : array
        Actor's hidden state.
    hidden_state_critic : array
        Critic's hidden state.
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
    value : jnp.ndarray
        Array of predicted values for steps.
    next_value : jnp.ndarray
        Array of predicted values for next steps per step.
    hidden_state_actor : array
        Actor's hidden states used in batch.
    hidden_state_critic : array
        Critic's hidden states used in batch.
    new_hidden_state_actor : array
        Actor's new hidden state.
    new_hidden_state_critic : array
        Critic's new hidden state.
    last_observation
        Last observation produced by the environment. Used for running
        an environment over multiple calls of this function.
    global_step : int, optional
        Global step count
    """

    @nnx.jit
    def sample(
        policy: StochasticRecurrentPolicyBase, observation, hidden_state, subkey
    ):
        return policy.sample(observation, hidden_state, subkey)

    @nnx.jit
    def value_fn(value_rnn: RNN, observation, hidden_state):
        value, next_hidden_state = value_rnn(observation, hidden_state)
        return value.flatten(), next_hidden_state

    trajectory_collector = TrajectoryCollector(
        envs.num_envs,
        batch_size,
        envs.single_observation_space.shape,
        envs.single_action_space.shape,
        hidden_state_actor.shape[1:],
        hidden_state_critic.shape[1:],
    )

    obs = envs.reset()[0] if last_observation is None else last_observation

    accumulated_return = 0.0
    for step in range(batch_size):
        key, subkey = jax.random.split(key)
        action, new_hidden_state_actor = sample(
            actor, obs, hidden_state_actor, subkey
        )
        value, new_hidden_state_critic = value_fn(
            critic, obs, hidden_state_critic
        )
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
            hidden_state_actor,
            hidden_state_critic,
        )

        if "episode" in info.keys():
            finished_reward_len_obs = [
                (r, l)
                for r, l, f in zip(
                    info["episode"]["r"],
                    info["episode"]["l"],
                    info["_episode"],
                )
                if f
            ]
            for i, (r, l) in enumerate(finished_reward_len_obs):
                accumulated_return += r
                if logger is not None:
                    logger.record_stat("return", r, step=global_step + step)

        obs = next_obs
        hidden_state_critic = new_hidden_state_critic
        hidden_state_actor = new_hidden_state_actor

    if logger is not None:
        accumulated_return += jnp.sum(envs.episode_returns).item()
        logger.record_stat(
            "return_per_epoch",
            accumulated_return,
            step=global_step + batch_size,
        )

    value, new_hidden_state_critic = value_fn(critic, obs, hidden_state_critic)
    batch = trajectory_collector.get_batch(value)

    return namedtuple(
        "PPO_Trajectory",
        [
            "observation",
            "action",
            "reward",
            "terminated",
            "value",
            "next_value",
            "hidden_state_actor",
            "hidden_state_critic",
            "new_hidden_state_actor",
            "new_hidden_state_critic",
            "last_observation",
            "global_step",
        ],
    )(
        batch[0],
        batch[1],
        batch[2],
        batch[3],
        batch[4],
        batch[5],
        batch[6],
        batch[7],
        hidden_state_actor,
        hidden_state_critic,
        obs,
        global_step + batch_size,
    )


def ppo_loss(
    actor: StochasticRecurrentPolicyBase,
    critic: RNN,
    old_logps: jnp.ndarray,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
    hidden_state_actor: jnp.ndarray,
    hidden_state_critic: jnp.ndarray,
    clip: float = 0.2,
) -> jnp.ndarray:
    """
    Calculate the PPO loss.

    Parameters
    ----------
    actor : StochasticRecurrentPolicyBase
        The actor network.
    critic : RNN
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
    hidden_state_actor : array
        Actor's hidden state.
    hidden_state_critic : array
        Critic's hidden state.
    clip : float, optional
        Clipping range for the PPO objective.

    Returns
    -------
    loss : jnp.ndarray
        The computed PPO loss for the batch.
    """
    logps = actor.log_probability(observations, hidden_state_actor, actions)[0]
    ratios = jnp.exp(logps - old_logps)
    surrogate1 = ratios * advantages
    surrogate2 = jnp.clip(ratios, 1 - clip, 1 + clip) * advantages
    policy_loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))

    values = critic(observations, hidden_state_critic)[0]
    value_loss = jnp.mean((returns - values) ** 2)

    return (
        policy_loss
        + 0.5 * value_loss
        - 0.01 * jnp.mean(actor.entropy(observations, hidden_state_actor)[0])
    )


def update_ppo(
    actor: StochasticRecurrentPolicyBase,
    critic: RNN,
    optimizer_actor: nnx.Optimizer,
    optimizer_critic: nnx.Optimizer,
    observation: jnp.ndarray,
    action: jnp.ndarray,
    reward: jnp.ndarray,
    terminated: jnp.ndarray,
    value: jnp.ndarray,
    next_value: jnp.ndarray,
    hidden_state_actor: jnp.ndarray,
    hidden_state_critic: jnp.ndarray,
    epochs: int = 1,
) -> jnp.ndarray:
    """
    Updates the PPO agent

    Args:
        actor : StochasticRecurrentPolicyBase
            The actor network
        critic : RNN
            The critic network
        observation : jnp.ndarray
            Array of observations.
        action : jnp.ndarray
            Actions taken per step.
        reward : jnp.ndarray
            Array of rewards per step.
        terminated : jnp.ndarray
            Flags indicating episode termination per step.
        value : jnp.ndarray
            Array of predicted values per step.
        next_value : jnp.ndarray
            Array of predicted next_values per step.
        hidden_state_actor : array
            Actor's hidden state.
        hidden_state_critic : array
            Critic's hidden state.
        epochs : int, optional
            Number of training epochs.

    Returns:
    - loss_val : jnp.ndarray
        Calculated loss.
    """
    advs, returns = compute_gae(reward, value, next_value, terminated)
    logp = actor.log_probability(observation, hidden_state_actor, action)[0]
    loss_grad_fn = nnx.value_and_grad(ppo_loss, argnums=(0, 1))

    for _ in range(epochs):
        loss_val, (grad_actor, grad_critic) = loss_grad_fn(
            actor,
            critic,
            logp,
            observation,
            action,
            advs,
            returns,
            hidden_state_actor,
            hidden_state_critic,
        )
        optimizer_actor.update(actor, grad_actor)
        optimizer_critic.update(critic, grad_critic)

    return loss_val


def train_rl2_ppo(
    env_set: list[gym.vector.VectorEnv],
    task_selector: TaskSelector,
    actor: StochasticRecurrentPolicyBase,
    critic: RNN,
    optimizer_actor: nnx.Optimizer,
    optimizer_critic: nnx.Optimizer,
    iterations: int = 3000,
    epochs: int = 1,
    batch_size: int = 64,
    seed: int = 1,
    logger: LoggerBase | None = None,
    progress_bar: bool = True,
) -> tuple[StochasticRecurrentPolicyBase, RNN, nnx.Optimizer, nnx.Optimizer]:
    """
    Train a PPO agent.

    Parameters
    ----------
    env_set : list[gym.vector.VectorEnv]
        Set of vectorized envbironments.
    task_selector : TaskSelector
        Selector for vectorized environments.
    actor : StochasticRecurrentPolicyBase
        The actor network.
    critic : RNN
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
    - actor : StochasticRecurrentPolicyBase
        Trained actor network.
    - critic : RNN
        Trained critic network.
    - optimizer_actor : nnx.Optimizer
        Updated actor optimizer.
    - optimizer_critic : nnx.Optimizer
        Updated critic optimizer.
    """
    key = jax.random.key(seed)

    for env in env_set:
        env.reset(seed=seed)
        env = gym.wrappers.vector.RecordEpisodeStatistics(env)

    if logger is not None:
        logger.start_new_episode()

    update_ppo_jitted = nnx.jit(update_ppo, static_argnames="epochs")

    global_step = 0
    hidden_state_actor = None
    hidden_state_critic = None
    last_observation = None
    last_task_id = -1
    for iteration in trange(iterations, disable=not progress_bar):
        task_id = task_selector.select()
        envs = env_set[task_id]
        if task_id != last_task_id:
            hidden_state_actor = actor.init_hidden_state(envs.num_envs)
            hidden_state_critic = critic.init_hidden_state(envs.num_envs)
            last_observation = None

        key, subkey = jax.random.split(key)
        (
            observation,
            action,
            reward,
            terminated,
            value,
            next_value,
            hidden_states_actor,
            hidden_states_critic,
            hidden_state_actor,
            hidden_state_critic,
            last_observation,
            global_step,
        ) = collect_trajectories(
            envs,
            actor,
            critic,
            hidden_state_actor,
            hidden_state_critic,
            subkey,
            batch_size,
            logger,
            last_observation,
            global_step,
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
            value,
            next_value,
            hidden_states_actor,
            hidden_states_critic,
            epochs,
        )

        task_selector.feedback(reward=reward.sum(), policy=nnx.clone(actor))
        if logger is not None:
            logger.record_stat("loss", loss_val, step=iteration)
            logger.record_epoch("RL2_PPO_ACTOR", actor, step=iteration)
            logger.record_epoch("RL2_PPO_CRITIC", critic, step=iteration)

    return actor, critic, optimizer_actor, optimizer_critic
