import gymnasium as gym
import numpy as np
import torch
from agent import Agent
from args import Args

from rl_blox.logging.logger import LoggerBase

from .storage import DataHolder, RunData


def collect_trajectories(
    envs: gym.vector.SyncVectorEnv,
    agent: Agent,
    data_holder: DataHolder,
    run_data: RunData,
    logger: LoggerBase = None,
) -> tuple[DataHolder, RunData, list]:
    device = data_holder.device
    next_obs, next_done = run_data.next_obs, run_data.next_done
    global_step = run_data.global_step
    rewards = []
    for step in range(0, data_holder.args.num_steps):
        global_step += data_holder.args.num_envs
        data_holder.obs[step] = next_obs
        data_holder.dones[step] = next_done

        # action logic
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            data_holder.values[step] = value.flatten()
        data_holder.actions[step] = action
        data_holder.logprobs[step] = logprob

        # execute the game and log data.
        next_obs, reward, terminations, truncations, infos = envs.step(
            action.cpu().numpy()
        )
        data_holder.rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_done = np.logical_or(terminations, truncations)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
            next_done
        ).to(device)

        if infos and "episode" in infos:
            print_reward = infos["episode"]["r"][0]
            print(f"global_step={global_step}, episodic_return={print_reward}")
            if logger is not None:
                logger.record_stat(
                    "episodic_return",
                    print_reward,
                    step=global_step,
                )
            rewards.append(print_reward)

    run_data.update(global_step, next_obs, next_done)

    return data_holder, run_data, rewards
