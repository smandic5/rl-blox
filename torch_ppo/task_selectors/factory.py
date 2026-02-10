import gymnasium as gym
import higher
import numpy as np
import scipy.special
import torch
import torch.optim as optim
from agent import Agent
from args import Args

from rl_blox.logging.logger import LoggerBase

from .task_selector import (
    HardTaskSelector,
    InsSelector,
    TaskSelector,
    UniformSelector,
)


def init_selector(
    index: int,
    envs: list[gym.vector.SyncVectorEnv],
    logger: LoggerBase,
    args: Args,
    agents: list[Agent] = None,
) -> TaskSelector:
    disimilarity = None
    from_last = None
    if index == 0:
        cls = UniformSelector
    elif index == 1:
        cls = HardTaskSelector
    elif index == 2:
        cls = InsSelector
        disimilarity = True
        from_last = True
    elif index == 3:
        cls = InsSelector
        disimilarity = True
        from_last = False
    elif index == 4:
        cls = InsSelector
        disimilarity = False
        from_last = True
    elif index == 5:
        cls = InsSelector
        disimilarity = False
        from_last = False
    else:
        raise Exception(f"Unrecognized selector index: {index}")
    return cls(
        envs,
        logger=logger,
        disimilarity=disimilarity,
        from_last=from_last,
    )
