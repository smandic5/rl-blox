import gymnasium as gym
import torch.optim as optim
from agent import Agent
from args import Args
from loss import Loss
from ppo_loss_calculator import calculate_loss
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
    num_iteration: int = None,
    is_meta_backbone: bool = False,
) -> tuple[Loss, list]:
    args = data_holder.args
    rewards = []
    if run_data is None:
        run_data = RunData(envs, args, data_holder.device)
    if num_iteration is None:
        num_iteration = args.num_iterations
    for iteration in range(1, num_iteration + 1):
        if args.anneal_lr:
            lr_annealing(args, optimizer, iteration)

        _, _, iter_rewards = collect_trajectories(
            envs,
            agent,
            data_holder,
            run_data,
            logger,
        )
        rewards.append(iter_rewards)

        if is_meta_backbone and iteration == num_iteration:
            (
                b_obs,
                b_actions,
                b_logprobs,
                b_values,
                b_advantages,
                b_returns,
            ) = data_holder.get_batch(agent, run_data)
            latest_loss, _ = calculate_loss(
                agent,
                b_obs,
                b_logprobs,
                b_actions,
                b_advantages,
                b_returns,
                b_values,
                args,
                [],
            )
        else:
            latest_loss = update_agent(
                agent,
                optimizer,
                data_holder,
                logger,
                run_data,
                is_meta_backbone,
            )
    return latest_loss, rewards
