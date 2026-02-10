import gymnasium as gym
import higher
import numpy as np
import torch
import torch.optim as optim
from agent import Agent
from args import Args
from checkpoint import save_model
from ppo.lr_handling import lr_annealing
from ppo.storage import DataHolder, RunData
from ppo.train_ppo import train_ppo
from task_selectors.task_selector import TaskSelector

from rl_blox.logging.logger import LoggerBase


def train_maml_ppo(
    agent: Agent,
    selector: TaskSelector,
    optimizer: optim.Optimizer,
    data_holder: DataHolder,
    logger: LoggerBase = None,
    run_name: str = None,
):
    args = data_holder.args
    inner_optimizer = torch.optim.SGD(
        agent.parameters(), lr=args.inner_learning_rate
    )
    for iteration in range(args.total_meta_iterations):
        if args.anneal_meta_lr:
            lr_annealing(
                args,
                optimizer,
                iteration,
                args.total_meta_iterations,
                False,
                start_lr=args.meta_learning_rate,
            )

        envs = selector.sample()
        optimizer.zero_grad()

        if iteration % args.eval_freq == 0:
            checkpoint(agent, args, iteration, run_name=run_name)

        print(f"Meta iteration: {iteration}")
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
                uses_inner_lr=True,
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
                    adapting_reward=adapting_reward,
                    adapted_reward=adapted_reward,
                ),
                used_model=fast_agent,
                reward=adapted_reward,
            )
            optimizer.step()
            logger.record_stat(
                "Learning_Rate", optimizer.param_groups[0]["lr"], step=iteration
            )


def checkpoint(
    agent: Agent,
    args: Args,
    iteration: int,
    run_name: str = None,
):
    if run_name is not None and args.save_checkpoints:
        save_model(args, run_name, agent, iteration=iteration)
