MAML_NAME = "MAML_PPO"
RL2_NAME = "RL2_PPO"
algorithm_to_use = MAML_NAME
save_frequency = 250
checkpoint_path = "./experiments/checkpoints/"

params_frozen_lake = dict(
    env_name="FrozenLake-v1",
    lake_size=4,
)
params_env = params_frozen_lake

hparams_model = dict(
    actor_hidden_layers=[64, 64],
    actor_activation="relu",
    actor_learning_rate=3e-4,
    critic_hidden_layers=[64, 64],
    critic_activation="relu",
    critic_learning_rate=1e-3,
)

hparams_algorithm = dict(
    num_envs=32,
    batch_size=128,
    iterations=5002,
    set_size_train=5,
    epochs=10,
)

hparams_eval = dict(
    iterations=5000,
    set_size_test=1,
    epochs=1,
)
