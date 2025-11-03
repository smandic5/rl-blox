import gymnasium as gym
import numpy as np


class OneHotObservationWrapper(gym.ObservationWrapper):
    """
    Wraps a discrete observation space to return one-hot encoded observations.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        if not isinstance(env.observation_space, gym.spaces.Discrete):
            raise TypeError(
                f"Expected Discrete observation space, got {type(env.observation_space)}"
            )

        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(env.observation_space.n,), dtype=np.float32
        )

    def observation(self, obs):
        """Convert scalar discrete observation to one-hot vector."""
        one_hot = np.zeros(self.observation_space.shape, dtype=np.float32)
        one_hot[obs] = 1.0
        return one_hot


class AppendHistoryWrapper(gym.Wrapper):
    """
    Augments observations with the previous action, reward, and done flag.
    """

    def __init__(self, env: gym.Env, one_hot_action=True):
        super().__init__(env)

        self.one_hot_action = one_hot_action

        if self.one_hot_action:
            if not isinstance(env.action_space, gym.spaces.Discrete):
                raise TypeError(
                    "One-hot action mode requires Discrete action space."
                )
            self.num_actions = env.action_space.n
        else:
            self.num_actions = np.prod(env.action_space.shape)

        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError(
                "AppendHistoryWrapper expects a Box observation space. If space is discrete, use OneHotObservationWrapper first."
            )

        base_obs_space = env.observation_space

        # New observation space
        low = np.concatenate(
            [
                base_obs_space.low,
                np.full((self.num_actions,), -np.inf, dtype=np.float32),
                np.array([-np.inf, 0.0], dtype=np.float32),
            ]
        )
        high = np.concatenate(
            [
                base_obs_space.high,
                np.full((self.num_actions,), np.inf, dtype=np.float32),
                np.array([np.inf, 1.0], dtype=np.float32),
            ]
        )

        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32
        )

        # Initialize last values
        self.last_action = np.zeros((self.num_actions,), dtype=np.float32)
        self.last_reward = np.zeros((1,), dtype=np.float32)
        self.last_done = np.zeros((1,), dtype=np.float32)

    def _augment_obs(self, obs):
        return np.concatenate(
            [obs, self.last_action, self.last_reward, self.last_done]
        ).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_action.fill(0)
        self.last_reward.fill(0)
        self.last_done.fill(0)
        return self._augment_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Update last action
        if self.one_hot_action:
            act_vec = np.zeros_like(self.last_action)
            act_vec[action] = 1.0
            self.last_action = act_vec
        else:
            self.last_action = np.array([action], dtype=np.float32)

        # Update last reward and done
        self.last_reward = np.array([reward], dtype=np.float32)
        done = float(terminated or truncated)
        self.last_done = np.array([done], dtype=np.float32)

        return self._augment_obs(obs), reward, terminated, truncated, info
