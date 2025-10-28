import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from rl_blox.algorithm.meta.rl2_ppo import train_rl2_ppo
from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.function_approximator.policy_head import SoftmaxPolicy
from rl_blox.blox.function_approximator.recurrent_policy_head import (
    RecurrentSoftmaxPolicy,
)
from rl_blox.blox.function_approximator.rnn import StackedGRU
from rl_blox.blox.multitask import UniformTaskSelector
from rl_blox.logging.logger import AIMLogger, LoggerList, StandardLogger

env_name = "CartPole-v1"
seed = 1
test_episodes = 10

hparams_model = {
    "actor_hidden_layers": [64, 64],
    "actor_activation": "relu",
    "actor_learning_rate": 3e-4,
    "critic_hidden_layers": [64, 64],
    "critic_activation": "relu",
    "critic_learning_rate": 1e-3,
}
hparams_algorithm = dict(
    num_envs=5,
    batch_size=64,
    iterations=1000,
    epochs=2,
    seed=seed,
)

envs = gym.make_vec(
    env_name,
    num_envs=hparams_algorithm["num_envs"],
    vectorization_mode="sync",
    vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP},
)

features = envs.observation_space.shape[1]
actions = int(envs.single_action_space.n)
features = envs.observation_space.shape[1] + actions + 2

actor = StackedGRU(
    features,
    actions,
    hparams_model["actor_hidden_layers"],
    hparams_model["actor_activation"],
    nnx.Rngs(seed),
)
actor = RecurrentSoftmaxPolicy(actor)

critic = StackedGRU(
    features,
    1,
    hparams_model["critic_hidden_layers"],
    hparams_model["critic_activation"],
    nnx.Rngs(seed),
)

optimizer_actor = nnx.Optimizer(
    actor, optax.adam(hparams_model["actor_learning_rate"]), wrt=nnx.Param
)
optimizer_critic = nnx.Optimizer(
    critic, optax.adam(hparams_model["critic_learning_rate"]), wrt=nnx.Param
)

logger = AIMLogger()
logger.define_experiment(
    env_name=env_name,
    algorithm_name="PPO",
    hparams=hparams_model | hparams_algorithm,
)

actor, critic, optimizer_actor, optimizer_critic = train_rl2_ppo(
    envs,
    actor,
    critic,
    optimizer_actor,
    optimizer_critic,
    iterations=hparams_algorithm["iterations"],
    epochs=hparams_algorithm["epochs"],
    logger=None,
    batch_size=hparams_algorithm["batch_size"],
)

# Evaluation

env = gym.make(env_name, render_mode="human")

obs = jnp.concatenate(
    [env.reset(seed=seed)[0], jnp.zeros(env.action_space.n + 2)]
)
hidden_state = actor.init_hidden_state(1)[0]

while True:
    probs, hidden_state = actor(obs, hidden_state)
    action = int(jnp.argmax(probs))
    obs, reward, terminated, truncated, _ = env.step(int(action))
    if terminated or truncated:
        obs, _ = env.reset()
    obs = jnp.concatenate([obs, jnp.zeros(env.action_space.n + 2)])
