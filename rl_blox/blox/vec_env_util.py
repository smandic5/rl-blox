from collections import namedtuple
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


class TrajectoryCollector:
    def __init__(
        self,
        num_envs,
        max_steps,
        obs_shape,
        action_shape,
        save_hidden_states=False,
        hidden_actor_shape=(),
        hidden_critic_shape=(),
    ):
        self.ptr = 0
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        self.observations = jnp.zeros((num_envs, max_steps, *obs_shape))
        self.actions = jnp.zeros((num_envs, max_steps, *action_shape))
        self.values = jnp.zeros((num_envs, max_steps + 1))
        self.rewards = jnp.zeros((num_envs, max_steps))
        self.terminated = jnp.zeros((num_envs, max_steps))

        self.save_hidden_states = save_hidden_states
        if save_hidden_states:
            self.hidden_actor_shape = hidden_actor_shape
            self.hidden_critic_shape = hidden_critic_shape
            self.hidden_actor = jnp.zeros(
                (num_envs, max_steps, *hidden_actor_shape)
            )
            self.hidden_critic = jnp.zeros(
                (num_envs, max_steps, *hidden_critic_shape)
            )
        else:
            self.hidden_actor = None
            self.hidden_critic = None
            self.hidden_actor_shape = None
            self.hidden_critic_shape = None

        self.invalid = jnp.zeros((num_envs, max_steps + 1), dtype=bool)

    def update(
        self,
        obs,
        action,
        value,
        reward,
        terminated,
        truncated,
        h_actor=None,
        h_critic=None,
    ):
        self.observations = self.observations.at[:, self.ptr].set(obs)
        self.actions = self.actions.at[:, self.ptr].set(action)
        self.values = self.values.at[:, self.ptr].set(value.flatten())
        self.rewards = self.rewards.at[:, self.ptr].set(reward)
        self.terminated = self.terminated.at[:, self.ptr].set(terminated)

        if self.save_hidden_states:
            self.hidden_actor = self.hidden_actor.at[:, self.ptr].set(h_actor)
            self.hidden_critic = self.hidden_critic.at[:, self.ptr].set(
                h_critic
            )

        self.ptr += 1
        self.invalid = self.invalid.at[:, self.ptr].set(
            jnp.logical_or(terminated, truncated)
        )

    def get_batch(self, next_values: jnp.ndarray) -> tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ]:
        self.values = self.values.at[:, -1].set(next_values.flatten())
        valid = ~self.invalid[:, :-1].flatten()

        if self.save_hidden_states:
            h_actor = self.hidden_actor.reshape(-1, *self.hidden_actor_shape)[
                valid
            ]
            h_critic = self.hidden_critic.reshape(
                -1, *self.hidden_critic_shape
            )[valid]
        else:
            h_actor = None
            h_critic = None

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
            ],
        )(
            self.observations.reshape(-1, *self.obs_shape)[valid],
            self.actions.reshape(-1, *self.action_shape)[valid],
            self.rewards.flatten()[valid],
            self.terminated.flatten()[valid],
            self.values[:, :-1].flatten()[valid],
            self.values[:, 1:].flatten()[valid],
            h_actor,
            h_critic,
        )


class OneHotVecObservationWrapper(gym.vector.VectorObservationWrapper):
    def __init__(self, env: gym.vector.VectorEnv):
        if not isinstance(env.single_observation_space, gym.spaces.Discrete):
            raise TypeError(
                f"Expected Discrete observation space, got {type(env.single_observation_space)}"
            )
        super().__init__(env)

        self.obs_states = env.single_observation_space.n

        # New observation space becomes a one-hot vector
        self.single_observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.obs_states,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.num_envs, self.obs_states),
            dtype=np.float32,
        )

    def _one_hot(self, obs):
        one_hot = np.zeros((obs.shape[0], self.obs_states), dtype=np.float32)
        one_hot[np.arange(obs.shape[0]), obs] = 1.0
        return one_hot

    def observations(self, observations):
        return self._one_hot(observations)


class AppendHistoryVecEnvWrapper(gym.vector.VectorWrapper):
    """
    A wrapper that augments each observation with the last action, reward, and done flag.
    """

    def __init__(self, env: gym.vector.VectorEnv, one_hot_action=True):
        super().__init__(env)

        self.one_hot_action = one_hot_action
        if self.one_hot_action:
            if not isinstance(env.single_action_space, gym.spaces.Discrete):
                raise TypeError(
                    "One-hot action mode requires Discrete action space."
                )
            self.num_actions = env.single_action_space.n
        else:
            self.num_actions = np.prod(env.single_action_space.shape)

        base_obs_space = env.single_observation_space
        if not isinstance(base_obs_space, gym.spaces.Box):
            raise TypeError(
                "Expected Box observation space. If space is discrete, use AppendHistoryVecEnvWrapper first."
            )

        obs_len = base_obs_space.shape[0] + self.num_actions + 2
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_len,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(env.num_envs, obs_len),
            dtype=np.float32,
        )

        self.last_action = np.zeros(
            (env.num_envs, self.num_actions), dtype=np.float32
        )
        self.last_reward = np.zeros((env.num_envs, 1), dtype=np.float32)
        self.last_done = np.zeros((env.num_envs, 1), dtype=np.float32)

    def _augment_obs(self, obs):
        return np.concatenate(
            [obs, self.last_action, self.last_reward, self.last_done], axis=-1
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_action.fill(0)
        self.last_reward.fill(0)
        self.last_done.fill(0)
        return self._augment_obs(obs), info

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self.env.step(actions)

        # Update last action
        if self.one_hot_action:
            acts = np.zeros_like(self.last_action)
            acts[np.arange(self.env.num_envs), actions] = 1.0
            self.last_action = acts
        else:
            self.last_action = np.expand_dims(actions.astype(np.float32), -1)

        # Update reward and done flags
        self.last_reward = np.expand_dims(rewards.astype(np.float32), -1)
        dones = np.logical_or(terminated, truncated).astype(np.float32)
        self.last_done = np.expand_dims(dones, -1)

        # Return augmented observation
        return self._augment_obs(obs), rewards, terminated, truncated, infos


class NegativePerStepVecWrapper(gym.vector.VectorRewardWrapper):
    def __init__(self, env: gym.vector.VectorEnv):
        super().__init__(env)
        self.step_punish = -0.025

    def rewards(self, rewards):
        return np.where(
            rewards == 0, np.ones(rewards.shape) * self.step_punish, rewards
        )
