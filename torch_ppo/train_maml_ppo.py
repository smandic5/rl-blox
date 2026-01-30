import gymnasium as gym
import higher
import numpy as np
import torch
import torch.optim as optim
from agent import Agent
from args import Args
from loss import Loss
from ppo_update import update_agent
from storage import DataHolder, RunData
from task_selector import TaskSelector
from train_ppo import train_ppo
from trajectories import collect_trajectories

from rl_blox.logging.logger import LoggerBase


def lr_annealing(args: Args, optimizer: optim.Optimizer, iteration: int):
    frac = 1.0 - (iteration - 1.0) / args.num_iterations
    lrnow = frac * args.learning_rate
    optimizer.param_groups[0]["lr"] = lrnow


def train_maml_ppo(
    agent: Agent,
    selector: TaskSelector,
    optimizer: optim.Optimizer,
    data_holder: DataHolder,
    logger: LoggerBase = None,
):
    args = data_holder.args
    inner_optimizer = torch.optim.SGD(
        agent.parameters(), lr=args.inner_learning_rate
    )
    for iteration in range(args.total_meta_iterations):
        envs = selector.sample()
        optimizer.zero_grad()
        with higher.innerloop_ctx(
            agent, inner_optimizer, copy_initial_weights=False
        ) as (fast_agent, diff_opt):
            inner_loss, rewards = train_ppo(
                fast_agent,
                envs,
                diff_opt,
                data_holder,
                num_iteration=args.num_adaptation_steps + 1,
                is_meta_backbone=True,
            )
            if logger is not None:
                inner_loss.print(logger, iteration)
                adapting_reward = np.mean([np.mean(r) for r in rewards[:-1]])
                adapted_reward = np.mean(rewards[-1])
                logger.record_stat(
                    "Adapting_Reward", adapting_reward, step=iteration
                )
                logger.record_stat(
                    "Adapted_Reward", adapted_reward, step=iteration
                )
            inner_loss.loss.backward()
        selector.feedback(
            to_log=dict(
                adapting_reward=adapting_reward, adapted_reward=adapted_reward
            )
        )
        optimizer.step()
