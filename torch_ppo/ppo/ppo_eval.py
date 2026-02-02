from typing import Callable

import gymnasium as gym
import torch


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = False,
    gamma: float = 0.99,
):
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, 0, capture_video, run_name, gamma)]
    )
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    ep_rew = 0
    ep_i = 0
    while len(episodic_returns) < eval_episodes:
        actions, _, _, _ = agent.get_action_and_value(
            torch.Tensor(obs).to(device)
        )
        next_obs, reward, terminations, truncations, infos = envs.step(
            actions.cpu().numpy()
        )
        dones = terminations or truncations
        ep_rew += reward[0]
        if dones[0] != 0:
            print(f"episode={ep_i}, episodic_return={ep_rew}")
            ep_rew = 0
            ep_i += 1
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(
                    f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}"
                )
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns
