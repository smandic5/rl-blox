import gymnasium as gym
from args import Args

from .single_env import make_env


def init_vec_envs(
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
