from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
from flax import nnx


class RNN(nnx.Module):
    """Model of stacked Gated Recurrent Units.

    Parameters
    ----------
    n_features : int
        Number of features.

    n_outputs : int
        Number of output components.

    hidden_nodes : list
        Numbers of hidden recurrent nodes of the MLP.

    activation : str
        Activation function. Has to be the name of a function defined in the
        flax.nnx module.

    rngs : nnx.Rngs
        Random number generator.
    """

    n_outputs: int
    """Number of output components."""

    activation: Callable[[jnp.ndarray], jnp.ndarray]
    """Activation function."""

    gru_layers: list[nnx.GRUCell]
    """GRU layers."""

    layer_sizes: list[int]
    """Size of GRU layers."""

    output_layer: nnx.Linear
    """Output layer."""

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        hidden_nodes: list[int],
        activation: str,
        rngs: nnx.Rngs,
    ):
        chex.assert_scalar_positive(n_features)
        chex.assert_scalar_positive(n_outputs)

        self.n_outputs = n_outputs
        self.activation = getattr(nnx, activation)

        self.gru_layers = []
        n_in = n_features
        for size in hidden_nodes:
            self.gru_layers.append(nnx.GRUCell(n_in, size, rngs=rngs))
            n_in = size

        self.layer_sizes = hidden_nodes
        self.output_layer = nnx.Linear(n_in, n_outputs, rngs=rngs)

    def __call__(
        self, x: jax.Array, h: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        new_hidden = h.copy()
        for i, gru_layer in enumerate(self.gru_layers):
            hi, x = gru_layer(h[..., i, : self.layer_sizes[i]], x)
            x = self.activation(x)
            new_hidden = new_hidden.at[..., i, : self.layer_sizes[i]].set(hi)
        return self.output_layer(x), new_hidden

    def init_hidden(self, batch_size: int) -> jax.Array:
        return jnp.zeros(
            (batch_size, len(self.layer_sizes), max(self.layer_sizes))
        )
