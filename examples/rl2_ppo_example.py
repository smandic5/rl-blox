import operator
from functools import reduce

import gymnasium as gym
import jax
import jax.numpy as jnp
import minigrid
import numpy as np
import optax
from flax import nnx

from rl_blox.algorithm.meta.rl2_ppo import train_rl2_ppo
from rl_blox.blox.env_util import AppendHistoryWrapper
from rl_blox.blox.function_approximator.recurrent_policy_head import (
    RecurrentSoftmaxPolicy,
)
from rl_blox.blox.function_approximator.rnn import StackedGRU
from rl_blox.blox.multitask import UniformTaskSelector
from rl_blox.blox.vec_env_util import AppendHistoryVecEnvWrapper
from rl_blox.logging.logger import AIMLogger, LoggerList, StandardLogger

params_minigrid = dict(
    env_name="MiniGrid-DistShift1-v0",
    train_names=[
        "MiniGrid-LavaCrossingS9N1-v0",
        "MiniGrid-LavaCrossingS9N2-v0",
    ],
    test_names=[
        "MiniGrid-LavaCrossingS9N1-v0",
        "MiniGrid-LavaCrossingS9N2-v0",
        "MiniGrid-LavaCrossingS9N3-v0",
    ],
    test_steps_per_env=1000,
)
hparams_model = {
    "actor_hidden_layers": [256, 256],
    "actor_activation": "relu",
    "actor_learning_rate": 3e-4,
    "critic_hidden_layers": [256, 256],
    "critic_activation": "relu",
    "critic_learning_rate": 1e-3,
}
hparams_algorithm = dict(
    num_envs=4,
    batch_size=256,
    iterations=1000,
    epochs=1,
    seed=1,
)


def make_minigrid_env(env_id=params_minigrid["env_name"], render_mode=None):
    def _thunk():
        env = gym.make(env_id, render_mode=render_mode)

        def obs_trans(obs):
            arr = np.zeros(4, dtype=np.uint8)
            arr[obs["direction"]] = 1
            return np.concatenate((arr, obs["image"].flatten()))

        box = gym.spaces.Box(
            low=0,
            high=255,
            shape=(
                4
                + reduce(
                    operator.mul, env.observation_space.spaces["image"].shape, 1
                ),
            ),
            dtype="uint8",
        )
        env = gym.wrappers.TransformObservation(env, obs_trans, box)
        return env

    return _thunk


env_set: list[gym.vector.VectorEnv] = []
for env_name in params_minigrid["train_names"]:
    envs = gym.vector.SyncVectorEnv(
        [
            make_minigrid_env(env_name)
            for _ in range(hparams_algorithm["num_envs"])
        ]
    )
    envs = AppendHistoryVecEnvWrapper(envs)
    env_set.append(envs)

actions = int(envs.single_action_space.n)
features = int(envs.single_observation_space.shape[0])

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

# logger = AIMLogger()
logger = StandardLogger(verbose=1)
logger.define_experiment(
    env_name=params_minigrid["env_name"],
    algorithm_name="RL2_PPO",
    hparams=hparams_model | hparams_algorithm | params_minigrid,
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
    env_i = i % len(params_minigrid["test_names"])
    env_name = params_minigrid["test_names"][env_i]

    env = make_minigrid_env(env_name, render_mode="human")()
    env = AppendHistoryWrapper(env)

    obs = env.reset(seed=hparams_algorithm["seed"])[0]
    hidden_state = actor.init_hidden_state(1)[0]

    env_reward = 0
    for _ in range(params_minigrid["test_steps_per_env"]):
        key, subkey = jax.random.split(key)
        action, hidden_state = actor.sample(obs, hidden_state, subkey)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        env_reward += reward
        if terminated or truncated:
            obs, _ = env.reset()
    logger.record_stat("test_env_reward", env_reward, step=i)

    i += 1
    env.close()
