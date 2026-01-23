import jax
import jax.numpy as jnp
import optax
from flax import nnx

from rl_blox.blox.function_approximator.gaussian_mlp import GaussianMLP
from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.function_approximator.policy_head import (
    GaussianTanhPolicy,
    SoftmaxPolicy,
)
from rl_blox.blox.function_approximator.recurrent_policy_head import (
    RecurrentSoftmaxPolicy,
)
from rl_blox.blox.function_approximator.rnn import StackedGRU


def create_policies(
    is_recurrent: bool,
    features: int,
    actions: int,
    hparams_model: dict,
    key: jnp.ndarray,
    action_space=None,
) -> tuple[
    RecurrentSoftmaxPolicy | SoftmaxPolicy,
    StackedGRU | MLP,
    nnx.Optimizer,
    nnx.Optimizer,
]:
    key_actor, key_critic = jax.random.split(key)
    actor = (StackedGRU if is_recurrent else MLP)(
        features,
        # 2,
        actions,
        hparams_model["actor_hidden_layers"],
        hparams_model["actor_activation"],
        nnx.Rngs(key_actor),
    )
    actor = (RecurrentSoftmaxPolicy if is_recurrent else SoftmaxPolicy)(actor)
    # actor = (RecurrentSoftmaxPolicy if is_recurrent else GaussianTanhPolicy)(actor, action_space=action_space)
    policy_net = GaussianMLP(
        shared_head=True,
        n_features=features,
        n_outputs=action_space.shape[0],
        hidden_nodes=list(hparams_model["actor_hidden_layers"]),
        activation="swish",
        rngs=nnx.Rngs(key_actor),
    )
    actor = GaussianTanhPolicy(policy_net, action_space=action_space)

    critic = (StackedGRU if is_recurrent else MLP)(
        features,
        1,
        hparams_model["critic_hidden_layers"],
        hparams_model["critic_activation"],
        nnx.Rngs(key_critic),
    )

    optimizer_actor = nnx.Optimizer(
        actor,
        optax.adam(
            optax.linear_schedule(
                hparams_model["actor_learning_rate"],
                hparams_model["actor_learning_rate"] / 1000,
                1000,
            )
        ),
        wrt=nnx.Param,
    )
    optimizer_critic = nnx.Optimizer(
        critic,
        optax.adam(
            optax.linear_schedule(
                hparams_model["critic_learning_rate"],
                hparams_model["critic_learning_rate"] / 1000,
                1000,
            )
        ),
        wrt=nnx.Param,
    )

    return actor, critic, optimizer_actor, optimizer_critic
