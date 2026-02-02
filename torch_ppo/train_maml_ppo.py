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

from rl_blox.logging.logger import AIMLogger, LoggerBase


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
    test_set: list[gym.vector.SyncVectorEnv] = None,
):
    args = data_holder.args
    inner_optimizer = torch.optim.SGD(
        agent.parameters(), lr=args.inner_learning_rate
    )
    for iteration in range(args.total_meta_iterations):
        envs = selector.sample()
        optimizer.zero_grad()

        if iteration % args.eval_freq == 0 and test_set is not None:
            envs = run_eval(agent, data_holder, test_set, args, iteration)

        print(f"Metal iteration: {iteration}")
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
                adapting_reward=adapting_reward, adapted_reward=adapted_reward
            )
        )
        optimizer.step()


def run_eval(
    agent: Agent,
    data_holder: DataHolder,
    test_set: list[gym.vector.SyncVectorEnv],
    args: Args,
    iteration: int,
    logger: LoggerBase = None,
):

    print("Evaluation started")
    for test_env_i, envs in enumerate(test_set):
        eval_logger = AIMLogger()
        eval_logger.define_experiment(
            env_name="Cheetah",
            algorithm_name=f"TorchMamlPPO_Eval_Env{test_env_i}",
            hparams=vars(args) | {iteration: iteration},
        )
        eval_logger.start_new_episode()
        agent_clone = Agent(envs)
        agent_clone.load_state_dict(agent.state_dict())
        optimizer_clone = torch.optim.SGD(
            agent.parameters(), lr=args.inner_learning_rate
        )
        inner_loss, rewards = train_ppo(
            agent_clone,
            envs,
            optimizer_clone,
            data_holder,
            num_iteration=args.eval_len,
            uses_inner_lr=True,
            logger=eval_logger,
        )
        if logger is not None:
            inner_loss.print(logger, iteration)
            adapting_reward = np.mean([np.mean(r) for r in rewards[:-1]])
            adapted_reward = np.mean(rewards[-1])
            logger.record_stat(
                f"Adapting_Reward_T{test_env_i}",
                adapting_reward,
                step=iteration,
            )
            logger.record_stat(
                f"Adapted_Reward_T{test_env_i}", adapted_reward, step=iteration
            )
        eval_logger.run.close()
    print("Evaluation ended.")
    return envs
