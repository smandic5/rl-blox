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
from ...logging.logger import LoggerBase


def one_hot(arr: jnp.ndarray, max_options: int) -> jnp.ndarray:
    res = jnp.eye(max_options)[arr]
    return res.reshape(list(arr.shape) + [max_options])


def create_observation(
    observation: jnp.ndarray,
    last_action: jnp.ndarray,
    last_reward: jnp.ndarray,
    last_done: jnp.ndarray,
    max_action_options: int,
) -> jnp.ndarray:
    return jnp.concatenate(
        [
            observation,
            one_hot(last_action, max_action_options),
            last_reward.reshape((-1, 1)),
            last_done.reshape((-1, 1)),
        ],
        axis=1,
    )


def collect_trajectories(
    envs: gym.vector.VectorEnv,
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
    def value(value_fn: RNN, observation, hidden_state):
        value, next_hidden_state = value_fn(observation, hidden_state)
        return value.flatten(), next_hidden_state

    def add_to_batch(batch, value):
        return (
            jnp.array(value[None, ...])
            if batch == None
            else jnp.concat([batch, value[None, ...]], axis=0)
        )

    (
        observations,
        actions,
        rewards,
        terminated_arr,
        next_values,
        hidden_states_actor,
        hidden_states_critic,
    ) = (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    obs = (
        create_observation(
            envs.reset()[0],
            envs.action_space.sample(),
            jnp.zeros(envs.num_envs),
            jnp.ones(envs.num_envs),
            envs.single_action_space.n,
        )
        if last_observation is None
        else last_observation
    )

    for _ in range(batch_size):
        key, subkey = jax.random.split(key)
        action, new_hidden_state_actor = sample(
            actor, obs, hidden_state_actor, subkey
        )
        next_obs, reward, terminated, truncated, info = envs.step(
            np.asarray(action)
        )
        next_obs = create_observation(
            next_obs,
            action,
            reward,
            jnp.logical_or(terminated, truncated),
            envs.single_action_space.n,
        )

        observations = add_to_batch(observations, obs)
        actions = add_to_batch(actions, action)
        hidden_states_actor = add_to_batch(
            hidden_states_actor, hidden_state_actor
        )
        rewards = add_to_batch(rewards, reward)
        terminated_arr = add_to_batch(terminated_arr, terminated)

        obs = jnp.copy(next_obs)
        if "episode" in info.keys():
            finished_reward_len_obs = [
                (r, l, o)
                for r, l, o, f in zip(
                    info["episode"]["r"],
                    info["episode"]["l"],
                    info["final_obs"],
                    info["_episode"],
                )
                if f
            ]
            for i, (r, l, o) in enumerate(finished_reward_len_obs):
                global_step += int(l)
                obs = obs.at[i, : envs.single_observation_space.shape[0]].set(o)
                if logger is not None:
                    # TODO figure out what to do with logging
                    pass

        next_value, new_hidden_state_critic = value(
            critic, obs, hidden_state_critic
        )
        next_values = add_to_batch(next_values, next_value)
        hidden_states_critic = add_to_batch(
            hidden_states_critic, hidden_state_critic
        )

        obs = next_obs
        hidden_state_critic = new_hidden_state_critic
        hidden_state_actor = new_hidden_state_actor

    def reshape_batch(batch):
        return jnp.permute_dims(batch, (1, 0)).flatten()

    def reshape_obs_batch(observations: jnp.ndarray):
        return jnp.swapaxes(observations, 0, 1).reshape(
            -1, observations.shape[-1]
        )

    def reshape_hidden_batch(hidden_state: jnp.ndarray):
        return jnp.swapaxes(hidden_state, 0, 1).reshape(
            (-1, hidden_state.shape[-2], hidden_state.shape[-1])
        )

    return namedtuple(
        "PPO_Trajectory",
        [
            "observation",
            "action",
            "reward",
            "terminated",
            "next_value",
            "hidden_state_actor",
            "hidden_state_critic",
            "new_hidden_state_actor",
            "new_hidden_state_critic",
            "last_observation",
            "global_step",
        ],
    )(
        reshape_obs_batch(observations),
        reshape_batch(actions),
        reshape_batch(rewards),
        reshape_batch(terminated_arr),
        reshape_batch(next_values),
        reshape_hidden_batch(hidden_states_actor),
        reshape_hidden_batch(hidden_states_critic),
        hidden_state_actor,
        hidden_state_critic,
        obs,
        global_step,
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
    advs, returns = compute_gae(
        reward,
        critic(observation, hidden_state_critic)[0].flatten(),
        next_value,
        terminated,
    )
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
    envs: gym.vector.VectorEnv,
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
    envs : gym.vector.VectorEnv
        The vectorized training environment.
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
    envs.reset(seed=seed)
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)
    assert (
        envs.metadata["autoreset_mode"] == gym.vector.AutoresetMode.SAME_STEP
    ), "Vectorized Env has to be instantiated with the SAME_STEP autoreset mode."
    # TODO envs -> task set

    if logger is not None:
        logger.start_new_episode()

    update_ppo_jitted = nnx.jit(update_ppo, static_argnames="epochs")

    global_step = 0
    hidden_state_actor = None
    hidden_state_critic = None
    last_observation = None
    for iteration in trange(iterations, disable=not progress_bar):
        # TODO sample task

        if True:  # TODO if new task
            hidden_state_actor = actor.init_hidden_state(envs.num_envs)
            hidden_state_critic = critic.init_hidden_state(envs.num_envs)

        key, subkey = jax.random.split(key)
        (
            observation,
            action,
            reward,
            terminated,
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
            next_value,
            hidden_states_actor,
            hidden_states_critic,
            epochs,
        )

        if logger is not None:
            logger.record_stat("loss", loss_val, step=iteration)

    return actor, critic, optimizer_actor, optimizer_critic
