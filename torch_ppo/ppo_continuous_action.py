# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import tyro
from agent import Agent
from args import Args
from env_handling import init_envs, make_env
from gae import calc_gae
from loss import Loss
from ppo_eval import evaluate
from ppo_update import update_agent
from storage import DataHolder, RunData
from trajectories import collect_trajectories

from rl_blox.logging.logger import (
    AIMLogger,
    LoggerBase,
    LoggerList,
    StandardLogger,
)


def init_seeds(args: Args):
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic


def save_model(args: Args, run_name: str, agent: Agent):
    model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(agent.state_dict(), model_path)
    print(f"model saved to {model_path}")
    return model_path


def evaluate_model(
    args: Args,
    run_name: str,
    logger: LoggerBase,
    device: torch.device,
    model_path: str,
):
    episodic_returns = evaluate(
        model_path,
        make_env,
        args.env_id,
        eval_episodes=10,
        run_name=f"{run_name}-eval",
        Model=Agent,
        device=device,
        gamma=args.gamma,
    )
    for idx, episodic_return in enumerate(episodic_returns):
        logger.record_stat(
            "eval/episodic_return", episodic_return, episode=idx, step=idx
        )


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


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = (
        f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    )
    logger = AIMLogger()
    logger.define_experiment(
        env_name="TorchCheetah",
        algorithm_name="TorchPPO",
        hparams=vars(args) | {run_name: run_name},
    )
    logger.start_new_episode()

    init_seeds(args)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    )
    envs = init_envs(args, run_name)
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    data_holder = DataHolder(envs, args, device)

    latest_loss = train_ppo(agent, envs, optimizer, data_holder, logger=logger)

    if args.save_model:
        model_path = save_model(args, run_name, agent)
        evaluate_model(args, run_name, logger, device, model_path)

    envs.close()
