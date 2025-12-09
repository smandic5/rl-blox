MAML_NAME = "MAML_PPO"
RL2_NAME = "RL2_PPO"
algorithm_to_use = MAML_NAME
save_frequency = 500
checkpoint_path = "./experiments/checkpoints/"

params_frozen_lake = dict(
    env_name="FrozenLake-v1",
    lake_size=5,
    step_success_rate=1.0,
    reward_goal=1,
    reward_frozen=-0.01,
    reward_hole=-1,
)
params_env = params_frozen_lake

hparams_model = dict(
    actor_hidden_layers=[64, 64],
    actor_activation="relu",
    actor_learning_rate=0.0001,
    critic_hidden_layers=[64, 64],
    critic_activation="relu",
    critic_learning_rate=0.0005,
)

hparams_algorithm = dict(
    num_envs=32,
    batch_size=128,
    iterations=1002,
    set_size_train=20,
    epochs=20,
)

hparams_eval = dict(
    seed_change_by=10,
    iterations=50,
    set_size_test=3,
    epochs=1,
)
