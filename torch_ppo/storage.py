import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from agent import Agent
from args import Args
from env_handling import init_envs, make_env
from gae import calc_gae
from ppo_update import update_agent
from trajectories import collect_trajectories

from rl_blox.logging.logger import (
    AIMLogger,
    LoggerBase,
    LoggerList,
    StandardLogger,
)


def init_storage(
    envs: gym.vector.SyncVectorEnv,
    args: Args,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    return obs, actions, logprobs, rewards, dones, values
