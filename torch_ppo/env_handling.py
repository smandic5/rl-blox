import gymnasium as gym
import numpy as np
from args import Args
from cheetah_meta_wrapper import HalfCheetahMetaWrapper


def make_env(
    env_id, idx, capture_video, run_name, gamma, target_velocity: float = None
):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(
            env
        )  # deal with dm_control's Dict observation space
        if target_velocity is not None:
            env = HalfCheetahMetaWrapper(env, target_velocity)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env, lambda obs: np.clip(obs, -10, 10), env.observation_space
        )
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(
            env, lambda reward: np.clip(reward, -10, 10)
        )
        return env

    return thunk


def init_envs(
    args: Args, run_name: str, target_velocity: float = None
) -> gym.vector.SyncVectorEnv:
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                i,
                args.capture_video,
                run_name,
                args.gamma,
                target_velocity,
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    return envs


def init_envs_set(args: Args, run_name: str) -> list[gym.vector.SyncVectorEnv]:
    return [
        init_envs(args, run_name, target_velocity)
        for target_velocity in np.random.uniform(0, 2, args.train_set_size)
    ]
