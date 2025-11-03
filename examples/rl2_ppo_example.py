from functools import partial

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

params_frozen_lake = dict(
    env_name="FrozenLake-v1",
    lake_size=4,
    set_size_train=5,
    set_size_test=2,
    test_steps_per_env=1000,
)
hparams_model = {
    "actor_hidden_layers": [64, 64],
    "actor_activation": "relu",
    "actor_learning_rate": 3e-4,
    "critic_hidden_layers": [64, 64],
    "critic_activation": "relu",
    "critic_learning_rate": 1e-3,
}
hparams_algorithm = dict(
    num_envs=32,
    batch_size=64,
    iterations=1,
    epochs=2,
    seed=1,
)

prep_key = jax.random.key(hparams_algorithm["seed"] + 1)
prep_key, subkey = jax.random.split(prep_key)
env_seeds = jax.random.randint(
    subkey,
    (
        params_frozen_lake["set_size_train"]
        + params_frozen_lake["set_size_test"],
    ),
    minval=1,
    maxval=1000,
)
env_set: list[gym.vector.VectorEnv] = []
for envi in range(hparams_algorithm["num_envs"]):
    envs = gym.make_vec(
        params_frozen_lake["env_name"],
        desc=generate_random_map(
            size=params_frozen_lake["lake_size"], seed=env_seeds[envi].item()
        ),
        num_envs=hparams_algorithm["num_envs"],
        vectorization_mode="sync",
    )
    envs = OneHotVecObservationWrapper(envs)
    envs = AppendHistoryVecEnvWrapper(envs)

    env_set.append(envs)

actions = int(envs.single_action_space.n)
features = int(envs.single_observation_space.shape[0])

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
# logger = StandardLogger(verbose=1)
logger.define_experiment(
    env_name=params_frozen_lake["env_name"],
    algorithm_name="RL2_PPO",
    hparams=hparams_model | hparams_algorithm | params_frozen_lake,
)

actor, critic, optimizer_actor, optimizer_critic = train_rl2_ppo(
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

i = 0
while True:
    env_i = i % params_frozen_lake["set_size_test"]

    env = gym.make(
        params_frozen_lake["env_name"],
        desc=generate_random_map(
            size=params_frozen_lake["lake_size"],
            seed=env_seeds[hparams_algorithm["num_envs"] + env_i].item(),
        ),
        render_mode="human",
    )
    env = OneHotObservationWrapper(env)
    env = AppendHistoryWrapper(env)

    obs = env.reset(seed=hparams_algorithm["seed"])[0]
    hidden_state = actor.init_hidden_state(1)[0]

    env_reward = 0
    for _ in range(params_frozen_lake["test_steps_per_env"]):
        probs, hidden_state = actor(obs, hidden_state)
        action = int(jnp.argmax(probs))
        obs, reward, terminated, truncated, _ = env.step(int(action))
        env_reward += reward
        if terminated or truncated:
            obs, _ = env.reset()
    logger.record_stat("test_env_reward", env_reward, step=i)

    i += 1
    env.close()
