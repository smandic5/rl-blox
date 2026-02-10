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
from args import Args, init_args
from envs.env_sets import init_env_sets
from logger import init_logger
from ppo.ppo_eval import evaluate
from ppo.storage import DataHolder, RunData
from ppo.train_ppo import train_ppo
from seeds import init_seeds
from task_selectors.task_selector import (
    HardTaskSelector,
    InsSelector,
    UniformSelector,
)
from train_maml_ppo import train_maml_ppo

from rl_blox.logging.logger import (
    AIMLogger,
    LoggerBase,
    LoggerList,
    StandardLogger,
)

if __name__ == "__main__":
    args, run_name, device = init_args()
    logger = init_logger(args, run_name)
    seed = args.seed
    init_seeds(seed, args.torch_deterministic)

    envs_train_set, envs_test_set = init_env_sets(args, run_name)
    agent = Agent(envs_train_set[0]).to(device)
    """selector = InsSelector(
        envs_train_set,
        from_last=True,
        agents=[agent for _ in range(len(envs_train_set))],
        disimilarity=True,
        logger=logger,
    )"""
    selector = HardTaskSelector(
        envs_train_set,
        logger=logger,
    )
    optimizer = optim.Adam(
        agent.parameters(), lr=args.meta_learning_rate, eps=1e-5
    )
    data_holder = DataHolder(envs_train_set[0], args, device)

    latest_loss = train_maml_ppo(
        agent,
        selector,
        optimizer,
        data_holder,
        logger=logger,
        test_set=envs_test_set,
        run_name=run_name,
    )

    for e in envs_train_set:
        e.close()
