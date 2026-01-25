MAML_NAME = "MAML_PPO"
RL2_NAME = "RL2_PPO"
algorithm_to_use = MAML_NAME
save_frequency = 500
checkpoint_path = "./experiments/checkpoints/"

params_frozen_lake = dict(
    env_name="FrozenLake-v1",
    lake_size=6,
    step_success_rate=1.0,
    reward_goal=1,
    reward_frozen=-0.01,
    reward_hole=-1,
)
params_mountain_car = dict(
    env_name="MountainCar-v0",
    goal_velocity=[9.8, 3.7, 8.8, 9.0, 10.5, 11.7, 24.7, 15.0, 6.0, 13.0],
)
params_half_cheetah = dict(
    env_name="HalfCheetah-v5",
    max_episode_steps=10,
    gravity=[9.8, 3.7, 8.8, 9.0, 10.5, 11.7, 6.7, 15.0, 6.0, 13.0],
)
params_cartpole = dict(
    env_name="CartPole-v1",
    gravity=[9.8, 3.7, 8.8, 9.0, 10.5, 11.7, 24.7, 15.0, 6.0, 13.0],
)
params_inverted_pendulum = dict(
    env_name="InvertedPendulum-v5",
    gravity=[9.8, 3.7, 8.8, 9.0, 10.5, 11.7, 24.7, 15.0, 6.0, 13.0],
)
params_pendulum = dict(
    env_name="Pendulum-v1",
    gravity=[9.8, 3.7, 8.8, 9.0, 10.5, 11.7, 24.7, 15.0, 6.0, 13.0],
)
params_env = params_half_cheetah

hparams_model = dict(
    actor_hidden_layers=[256, 256],
    actor_activation="relu",
    actor_learning_rate=2.0633e-05,
    critic_hidden_layers=[256, 256],
    critic_activation="relu",
    critic_learning_rate=2.0633e-05,
)

hparams_algorithm = dict(
    train_set_size=8,
    iterations=100000,
    set_size_train=10,
    meta_epochs=5,
)

hparams_backbone = dict(
    num_envs=1,
    rollout_length=512,
    batch_size=64,
    backbone_epochs=20,
)

hparams_eval = dict(
    seed_change_by=10,
    iterations=100000,
    set_size_test=2,
    epochs=1,
)
