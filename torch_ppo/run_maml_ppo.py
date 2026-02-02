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
from envs.env_sets import init_env_sets
from ppo_eval import evaluate
from storage import DataHolder, RunData
from task_selector import UniformSelector
from train_maml_ppo import train_maml_ppo
from train_ppo import train_ppo

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
        env_name="MamlTorchCheetah",
        algorithm_name="TorchMamlPPO",
        hparams=vars(args) | {run_name: run_name},
    )
    logger.start_new_episode()

    init_seeds(args)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    )
    envs_train_set, envs_test_set = init_env_sets(args, run_name)
    selector = UniformSelector(envs_train_set, logger=logger)
    agent = Agent(envs_train_set[0]).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    data_holder = DataHolder(envs_train_set[0], args, device)

    latest_loss = train_maml_ppo(
        agent,
        selector,
        optimizer,
        data_holder,
        logger=logger,
        test_set=envs_test_set,
    )

    if args.save_model:
        model_path = save_model(args, run_name, agent)
        evaluate_model(args, run_name, logger, device, model_path)

    for e in envs_train_set:
        e.close()
