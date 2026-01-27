import gymnasium as gym
import torch.optim as optim
from agent import Agent
from args import Args
from loss import Loss
from ppo_update import update_agent
from storage import DataHolder, RunData
from trajectories import collect_trajectories

from rl_blox.logging.logger import LoggerBase


def lr_annealing(args: Args, optimizer: optim.Optimizer, iteration: int):
    frac = 1.0 - (iteration - 1.0) / args.num_iterations
    lrnow = frac * args.learning_rate
    optimizer.param_groups[0]["lr"] = lrnow


def train_ppo(
    agent: Agent,
    envs: gym.vector.SyncVectorEnv,
    optimizer: optim.Optimizer,
    data_holder: DataHolder,
    run_data: RunData = None,
    logger: LoggerBase = None,
) -> Loss:
    args = data_holder.args
    if run_data is None:
        run_data = RunData(envs, args, data_holder.device)
    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            lr_annealing(args, optimizer, iteration)

        collect_trajectories(
            envs,
            agent,
            data_holder,
            run_data,
            logger,
        )

        latest_loss = update_agent(
            agent,
            optimizer,
            data_holder,
            logger,
            run_data,
        )
    return latest_loss
