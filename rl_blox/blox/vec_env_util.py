from collections import namedtuple
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
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
