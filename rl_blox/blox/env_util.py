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
