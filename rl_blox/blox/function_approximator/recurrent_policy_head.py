import gymnasium as gym
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as dist
from flax import nnx

from .rnn import RNN


class StochasticRecurrentPolicyBase(nnx.Module):
    """Base class for probabilistic recurrent policies.

    A subclass must define the functions

    * :func:`~StochasticRecurrentPolicyBase.__call__`
    * :func:`~StochasticRecurrentPolicyBase.sample`
    * :func:`~StochasticRecurrentPolicyBase.log_probability`
    * :func:`~StochasticRecurrentPolicyBase.init_hidden`
    """

    def __call__(self, observation: jnp.ndarray) -> jnp.ndarray:
        """Compute action probabilities for given observation."""
        raise NotImplementedError("Subclasses must implement __call__ method.")

    def sample(self, observation: jnp.ndarray, key: jnp.ndarray) -> jnp.ndarray:
        """Sample action from policy given observation.

        Parameters
        ----------
        observation : array
            Observation.

        key : array
            Pseudo random number generator key for sampling.

        Returns
        -------
        action : array
            Sampled action.
        """
        raise NotImplementedError("Subclasses must implement sample method.")

    def log_probability(
        self,
        observation: jnp.ndarray,
        action: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute log probability of action given observation.

        Parameters
        ----------
        observation : array
            Observation.

        action : array
            Action.

        Returns
        -------
        log_prob : array
            Log probability of action given observation.
        """
        raise NotImplementedError(
            "Subclasses must implement log_probability method."
        )

    def init_hidden_state(self, batch_size: int) -> jnp.ndarray:
        """Initialize and return a hidden state.

        Parameters
        ----------
        batch_size : int
            Batch size for the hidden state

        Returns
        -------
        hidden_state : array
            Hidden state.
        """
        raise NotImplementedError(
            "Subclasses must implement init_hidden method."
        )


class RecurrentSoftmaxPolicy(StochasticRecurrentPolicyBase):
    r"""Recurrent Softmax policy for discrete action spaces.

    Wraps a softmax neural network that maps observations to the logits of each
    action.

    Parameters
    ----------
    net : RNN
        Recurrent Neural network that maps observations and a hidden state to logits a new hidden state.
    """

    net: RNN

    def __init__(self, net: nnx.Module):
        self.net = net

    def __call__(
        self, observation: jnp.ndarray, hidden_state: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        logits, new_hidden = self.logits(observation, hidden_state)
        return nnx.softmax(logits), new_hidden

    def logits(
        self, observation: jnp.ndarray, hidden_state: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.net(observation, hidden_state)

    def sample(
        self,
        observation: jnp.ndarray,
        hidden_state: jnp.ndarray,
        key: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        logits, new_hidden = self.logits(observation, hidden_state)
        return (
            dist.Categorical(logits=logits).sample(
                seed=key,
                sample_shape=(),
            ),
            new_hidden,
        )

    def log_probability(
        self,
        observation: jnp.ndarray,
        hidden_state: jnp.ndarray,
        action: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        logits, new_hidden = self.logits(observation, hidden_state)
        return dist.Categorical(logits=logits).log_prob(action), new_hidden

    def init_hidden_state(self, batch_size: int):
        return self.net.init_hidden_state(batch_size)
