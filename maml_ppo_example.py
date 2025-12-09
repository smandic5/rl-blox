import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from rl_blox.algorithm.meta.maml_ppo import train_maml_ppo
from rl_blox.algorithm.ppo import train_ppo
from rl_blox.blox.adaptation_metrics import (
    asymptotic_performance,
    jumpstart,
    total_reward,
)
from rl_blox.blox.env_util import OneHotObservationWrapper
from rl_blox.blox.function_approximator.mlp import MLP
from rl_blox.blox.function_approximator.policy_head import SoftmaxPolicy
from rl_blox.blox.multitask import UniformTaskSelector
from rl_blox.blox.vec_env_util import OneHotVecObservationWrapper
from rl_blox.logging.logger import AIMLogger, MemoryLogger

jax.config.update("jax_platforms", "cpu")


env_name = "FrozenLake-v1"
lake_size = 4
reward_model = (1.0, 0.0, -1.0)

hparams_model = {
    "actor_hidden_layers": [64, 64],
    "actor_activation": "relu",
    "actor_learning_rate": 3e-4,
    "critic_hidden_layers": [64, 64],
    "critic_activation": "relu",
    "critic_learning_rate": 1e-3,
}
hparams_algorithm = dict(
    num_envs=16,
    batch_size=128,
    iterations=500,
    epochs=20,
    train_set_size=1,
    test_set_size=1,
    seed=1,
)

key = jax.random.key(hparams_algorithm["seed"])
key, subkey = jax.random.split(key)
env_set_size = (
    hparams_algorithm["train_set_size"] + hparams_algorithm["test_set_size"]
)
env_seeds = jax.random.randint(subkey, (env_set_size,), 1, 100)
desc = generate_random_map(size=lake_size, seed=env_seeds[0].item())
env_set = [
    gym.make_vec(
        env_name,
        desc=desc,
        num_envs=hparams_algorithm["num_envs"],
        vectorization_mode="sync",
        is_slippery=False,
        success_rate=1.0,
        reward_schedule=reward_model,
    )
    for i in range(env_set_size)
]
env_set = [OneHotVecObservationWrapper(envs) for envs in env_set]

features = env_set[0].observation_space.shape[1]
actions = int(env_set[0].single_action_space.n)

actor = MLP(
    features,
    actions,
    hparams_model["actor_hidden_layers"],
    hparams_model["actor_activation"],
    nnx.Rngs(hparams_algorithm["seed"]),
)
actor = SoftmaxPolicy(actor)

critic = MLP(
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

selector = UniformTaskSelector(
    jnp.arange(hparams_algorithm["train_set_size"]), key=key
)

logger = AIMLogger()
logger.define_experiment(
    env_name=env_name,
    algorithm_name="MAML_PPO",
    hparams=hparams_model | hparams_algorithm,
)

actor, critic, optimizer_actor, optimizer_critic = train_maml_ppo(
    env_set[: hparams_algorithm["train_set_size"]],
    selector,
    actor,
    critic,
    optimizer_actor,
    optimizer_critic,
    iterations=hparams_algorithm["iterations"],
    epochs=hparams_algorithm["epochs"],
    logger=logger,
    batch_size=hparams_algorithm["batch_size"],
)

# Adaptation

envs = env_set[0]

memory_logger = MemoryLogger()
memory_logger.define_experiment(
    env_name=env_name,
    algorithm_name="MAML_PPO",
    hparams=hparams_model | hparams_algorithm,
)

actor, critic, optimizer_actor, optimizer_critic = train_ppo(
    envs,
    actor,
    critic,
    optimizer_actor,
    optimizer_critic,
    iterations=100,
    epochs=5,
    logger=memory_logger,
    batch_size=hparams_algorithm["batch_size"],
)

x, y = memory_logger.get_stat("return")
js = jumpstart(y, hparams_algorithm["batch_size"]).item()
ap = asymptotic_performance(y, hparams_algorithm["batch_size"]).item()
tr = total_reward(y).item()

print(f"Jumpstart: {js}")
print(f"Asymptotic Performance: {ap}")
print(f"Total Reward: {tr}")

x, y = memory_logger.get_stat("success")
js = jumpstart(y, hparams_algorithm["batch_size"]).item()
ap = asymptotic_performance(y, hparams_algorithm["batch_size"]).item()
tr = total_reward(y).item()

print(f"Jumpstart: {js}")
print(f"Asymptotic Performance: {ap}")
print(f"Total Successes: {tr}")

# Evaluation

env = gym.make(
    env_name,
    desc=desc,
    is_slippery=False,
    success_rate=1.0,
    reward_schedule=reward_model,
    render_mode="human",
)
env = OneHotObservationWrapper(env)
obs, _ = env.reset(seed=hparams_algorithm["seed"])

while True:
    action = int(jnp.argmax(actor(obs)))
    obs, reward, terminated, truncated, _ = env.step(int(action))
    if terminated or truncated:
        obs, _ = env.reset()
