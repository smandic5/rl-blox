from collections import namedtuple
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


class TrajectoryCollector:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self._episode_start = jnp.zeros(num_envs, dtype=bool)

        self.trajectories: list[list[tuple]] = []
        for _ in range(self.num_envs):
            self.trajectories.append([])

        self.observations: jnp.ndarray | None = None
        self.actions: jnp.ndarray | None = None
        self.values: jnp.ndarray | None = None
        self.next_values: jnp.ndarray | None = None
        self.rewards: jnp.ndarray | None = None
        self.terminated: jnp.ndarray | None = None
        self.hidden_states_actor: jnp.ndarray | None = None
        self.hidden_states_critic: jnp.ndarray | None = None

    def _add_to_batch(
        self, batch: jnp.ndarray | None, trajectory: jnp.ndarray
    ) -> jnp.ndarray:
        return (
            trajectory
            if batch == None
            else jnp.concat([batch, trajectory], axis=0)
        )

    def update(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        value: jnp.ndarray,
        reward,
        terminated: jnp.ndarray,
        truncated: jnp.ndarray,
        hidden_state_actor: jnp.ndarray,
        hidden_state_critic: jnp.ndarray,
    ):
        for i in range(self.num_envs):
            if not self._episode_start[i]:
                step = (
                    obs[i],
                    action[i],
                    value[i],
                    reward[i],
                    terminated[i],
                    hidden_state_actor[i],
                    hidden_state_critic[i],
                )
                self.trajectories[i].append(step)
            else:
                self.finish_trajectory(i, value[i])
        self._episode_start = jnp.logical_or(terminated, truncated)

    def _extract_batch_stat(
        self, batch_index: int, stat_index: int
    ) -> jnp.ndarray:
        return jnp.concatenate(
            [
                jnp.array(step[stat_index][None, ...])
                for step in self.trajectories[batch_index]
            ]
        )

    def finish_trajectory(self, i: int, value: jnp.ndarray):
        observations = self._extract_batch_stat(i, 0)
        actions = self._extract_batch_stat(i, 1)
        values = self._extract_batch_stat(i, 2)
        next_values = jnp.append(values[1:], value)
        rewards = self._extract_batch_stat(i, 3)
        terminated_arr = self._extract_batch_stat(i, 4)
        hidden_states_actor = self._extract_batch_stat(i, 5)
        hidden_states_critic = self._extract_batch_stat(i, 6)
        self.trajectories[i].clear()

        self.observations = self._add_to_batch(self.observations, observations)
        self.actions = self._add_to_batch(self.actions, actions)
        self.values = self._add_to_batch(self.values, values)
        self.next_values = self._add_to_batch(self.next_values, next_values)
        self.rewards = self._add_to_batch(self.rewards, rewards)
        self.terminated = self._add_to_batch(self.terminated, terminated_arr)
        self.hidden_states_actor = self._add_to_batch(
            self.hidden_states_actor, hidden_states_actor
        )
        self.hidden_states_critic = self._add_to_batch(
            self.hidden_states_critic, hidden_states_critic
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
        for i in range(self.num_envs):
            if len(self.trajectories[i]) == 0:
                continue
            self.finish_trajectory(i, next_values[i])

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
            self.observations,
            self.actions.flatten(),
            self.rewards.flatten(),
            self.terminated.flatten(),
            self.values.flatten(),
            self.next_values.flatten(),
            self.hidden_states_actor,
            self.hidden_states_critic,
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
