import gymnasium as gym
import numpy as np
import torch.optim as optim
from agent import Agent
from args import Args
from logger_base import LoggerBase

from .lr_handling import lr_annealing
from .ppo_update import update_agent
from .storage import DataHolder, RunData
from .trajectories import collect_trajectories
from .update.loss import Loss


def train_ppo(
    agent: Agent,
    envs: gym.vector.SyncVectorEnv,
    optimizer: optim.Optimizer,
    data_holder: DataHolder,
    run_data: RunData = None,
    logger: LoggerBase = None,
    num_iteration: int = None,
    is_meta_backbone: bool = False,
    uses_inner_lr: bool = False,
) -> tuple[Loss, list]:
    args = data_holder.args
    rewards = []
    if run_data is None:
        run_data = RunData(envs, args, data_holder.device)
    if num_iteration is None:
        num_iteration = args.num_iterations
    for iteration in range(1, num_iteration + 1):
        if (args.anneal_ppo_lr and not uses_inner_lr) or (
            args.anneal_inner_lr and uses_inner_lr
        ):
            lr_annealing(
                args, optimizer, iteration, num_iteration, uses_inner_lr
            )

        _, _, iter_rewards = collect_trajectories(
            envs,
            agent,
            data_holder,
            run_data,
            logger,
        )
        rewards.append(iter_rewards)

        latest_loss = update_agent(
            agent,
            optimizer,
            data_holder,
            logger,
            run_data,
            is_inner_optimizer=is_meta_backbone,
            return_first_loss=is_meta_backbone and iteration == num_iteration,
        )
    return latest_loss, rewards
