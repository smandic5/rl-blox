import os

import gymnasium as gym
import torch
from agent import Agent
from args import Args


def save_model(args: Args, run_name: str, agent: Agent, iteration: int = None):
    print("saving")
    model_path = f"torch_ppo/runs/{run_name}/{args.exp_name}-{iteration}.agent"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(agent.state_dict(), model_path)
    print(f"model saved to {model_path}")
    return model_path


def load_model(
    run_name: str,
    args: Args,
    envs: gym.vector.SyncVectorEnv,
    device: torch.device,
):
    model_path = f"torch_ppo/runs/{run_name}/{args.exp_name}.agent"
    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    return agent
