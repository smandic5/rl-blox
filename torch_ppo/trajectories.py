import gymnasium as gym
import numpy as np
import torch
from agent import Agent
from args import Args

from rl_blox.logging.logger import LoggerBase


def collect_trajectories(
    envs: gym.vector.SyncVectorEnv,
    agent: Agent,
    obs: torch.Tensor,
    actions: torch.Tensor,
    logprobs: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_obs: torch.Tensor,
    next_done: torch.Tensor,
    args: Args,
    device: torch.device,
    global_step: int,
    logger: LoggerBase = None,
) -> tuple[int, np.ndarray, np.ndarray]:
    for step in range(0, args.num_steps):
        global_step += args.num_envs
        obs[step] = next_obs
        dones[step] = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, terminations, truncations, infos = envs.step(
            action.cpu().numpy()
        )
        next_done = np.logical_or(terminations, truncations)
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
            next_done
        ).to(device)

        if infos and "episode" in infos:
            print(
                f"global_step={global_step}, episodic_return={infos['episode']['r']}"
            )
            logger.record_stat(
                "episodic_return",
                infos["episode"]["r"],
                step=global_step,
            )

    return global_step, next_obs, next_done
