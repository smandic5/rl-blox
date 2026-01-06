MAML_NAME = "MAML_PPO"
RL2_NAME = "RL2_PPO"
algorithm_to_use = MAML_NAME
save_frequency = 50
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
    goal_velocity=[0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
)
params_cartpole = dict(
    env_name="CartPole-v1",
    gravity=[9.8, 3.7, 8.8, 9.0, 10.5, 11.7, 24.7, 15.0, 6.0, 13.0],
)
params_env = params_cartpole

hparams_model = dict(
    actor_hidden_layers=[1024, 1024],
    actor_activation="relu",
    actor_learning_rate=3e-4,
    critic_hidden_layers=[1024, 1024],
    critic_activation="relu",
    critic_learning_rate=1e-3,
)

hparams_algorithm = dict(
    num_envs=32,
    iterations=10002,
    batch_size=512,
    set_size_train=10,
    epochs=20,
)

hparams_eval = dict(
    seed_change_by=10,
    iterations=50,
    set_size_test=2,
    epochs=3,
)
