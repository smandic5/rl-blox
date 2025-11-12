import operator
from functools import reduce

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from rl_blox.algorithm.meta.rl2_ppo import train_rl2_ppo
from rl_blox.blox.env_util import AppendHistoryWrapper, OneHotObservationWrapper
from rl_blox.blox.function_approximator.recurrent_policy_head import (
    RecurrentSoftmaxPolicy,
)
from rl_blox.blox.function_approximator.rnn import StackedGRU
from rl_blox.blox.multitask import UniformTaskSelector
from rl_blox.blox.vec_env_util import (
    AppendHistoryVecEnvWrapper,
    OneHotVecObservationWrapper,
)
from rl_blox.logging.logger import AIMLogger, LoggerList, StandardLogger

env_name = "FrozenLake-v1"
lake_size = 3

hparams_model = {
    "actor_hidden_layers": [64, 64],
    "actor_activation": "relu",
    "actor_learning_rate": 3e-4,
    "critic_hidden_layers": [64, 64],
    "critic_activation": "relu",
    "critic_learning_rate": 1e-3,
}
hparams_algorithm = dict(
    num_envs=64,
    batch_size=128,
    iterations=50,
    epochs=1,
    train_set_size=2,
    test_set_size=1,
    seed=1,
)

key = jax.random.key(hparams_algorithm["seed"])
key, subkey = jax.random.split(key)
env_set_size = (
    hparams_algorithm["train_set_size"] + hparams_algorithm["test_set_size"]
)
env_seeds = jax.random.randint(subkey, (env_set_size,), 1, 100)
env_set = [
    gym.make_vec(
        env_name,
        desc=generate_random_map(size=lake_size, seed=env_seeds[i].item()),
        num_envs=hparams_algorithm["num_envs"],
        vectorization_mode="sync",
    )
    for i in range(env_set_size)
]
env_set = [OneHotVecObservationWrapper(envs) for envs in env_set]
env_set = [AppendHistoryVecEnvWrapper(envs) for envs in env_set]

features = env_set[0].observation_space.shape[1]
actions = int(env_set[0].single_action_space.n)

prep_key = jax.random.key(hparams_algorithm["seed"] + 1)
key, prep_key = jax.random.split(prep_key)
task_selector = UniformTaskSelector(env_set, key=prep_key)

actor = StackedGRU(
    features,
    actions,
    hparams_model["actor_hidden_layers"],
    hparams_model["actor_activation"],
    nnx.Rngs(hparams_algorithm["seed"]),
)
actor = RecurrentSoftmaxPolicy(actor)

critic = StackedGRU(
    features,
    1,
    hparams_model["critic_hidden_layers"],
    hparams_model["critic_activation"],
    nnx.Rngs(hparams_algorithm["seed"]),
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
    algorithm_name="RL2_PPO",
    hparams=hparams_model | hparams_algorithm,
)

actor, critic, optimizer_actor, optimizer_critic = train_rl2_ppo(
    env_set,
    task_selector,
    actor,
    critic,
    optimizer_actor,
    optimizer_critic,
    iterations=hparams_algorithm["iterations"],
    epochs=hparams_algorithm["epochs"],
    logger=logger,
    batch_size=hparams_algorithm["batch_size"],
    seed=hparams_algorithm["seed"],
)

for envs in env_set:
    envs.close()

# Evaluation

env = gym.make(
    env_name,
    desc=generate_random_map(
        size=lake_size,
        seed=env_seeds[hparams_algorithm["train_set_size"]].item(),
    ),
    render_mode="human",
)
env = OneHotObservationWrapper(env)
env = AppendHistoryWrapper(env)
obs, _ = env.reset(seed=hparams_algorithm["seed"])

hidden_state = actor.init_hidden_state(1)[0]
while True:
    key, subkey = jax.random.split(prep_key)
    action, hidden_state = actor.sample(obs, hidden_state, subkey)
    obs, reward, terminated, truncated, _ = env.step(int(action))
    if terminated or truncated:
        obs, _ = env.reset()
