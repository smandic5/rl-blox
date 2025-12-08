MAML_NAME = "MAML_PPO"
RL2_NAME = "RL2_PPO"
algorithm_to_use = MAML_NAME
save_frequency = 500
checkpoint_path = "./experiments/checkpoints/"

params_frozen_lake = dict(
    env_name="FrozenLake-v1",
    lake_size=4,
    step_success_rate=1.0,
    reward_goal=0,
    reward_frozen=-0.01,
    reward_hole=-1,
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
    iterations=10002,
    set_size_train=5,
    epochs=10,
)

hparams_eval = dict(
    seed_change_by=10,
    iterations=300,
    set_size_test=3,
    epochs=3,
)
