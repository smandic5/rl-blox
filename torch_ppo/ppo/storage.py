import gymnasium as gym
import torch
from agent import Agent
from args import Args

from .gae import calc_gae


class RunData:
    def __init__(
        self,
        envs: gym.vector.SyncVectorEnv,
        args: Args,
        device: torch.device,
    ):
        # run data
        self.global_step = 0
        next_obs, _ = envs.reset(seed=args.seed)
        self.next_obs = torch.Tensor(next_obs).to(device)
        self.next_done = torch.zeros(args.num_envs).to(device)

    def update(
        self, global_step: int, next_obs: torch.Tensor, next_done: torch.Tensor
    ):
        self.global_step = global_step
        self.next_obs = next_obs
        self.next_done = next_done


class DataHolder:
    def __init__(
        self,
        envs: gym.vector.SyncVectorEnv,
        args: Args,
        device: torch.device,
    ):
        # core data
        self.args = args
        self.device = device
        self.obs_space = envs.single_observation_space
        self.act_space = envs.single_action_space

        # containers
        container_shape = (args.num_steps, args.num_envs)
        self.obs = torch.zeros(container_shape + self.obs_space.shape).to(
            device
        )
        self.actions = torch.zeros(container_shape + self.act_space.shape).to(
            device
        )
        self.logprobs = torch.zeros(container_shape).to(device)
        self.rewards = torch.zeros(container_shape).to(device)
        self.dones = torch.zeros(container_shape).to(device)
        self.values = torch.zeros(container_shape).to(device)

    def get_batch(self, agent: Agent, run_data: RunData) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        advantages, returns = calc_gae(
            agent,
            self.rewards,
            self.dones,
            self.values,
            run_data.next_obs,
            run_data.next_done,
            self.args,
            self.device,
        )

        b_obs = self.obs.reshape((-1,) + self.obs_space.shape)
        b_actions = self.actions.reshape((-1,) + self.act_space.shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_values = self.values.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        return b_obs, b_actions, b_logprobs, b_values, b_advantages, b_returns
