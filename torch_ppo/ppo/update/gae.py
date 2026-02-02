import torch
from agent import Agent
from args import Args


def calc_gae(
    agent: Agent,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_obs: torch.Tensor,
    next_done: torch.Tensor,
    args: Args,
    device: torch.device,
):
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = (
                rewards[t]
                + args.gamma * nextvalues * nextnonterminal
                - values[t]
            )
            advantages[t] = lastgaelam = (
                delta
                + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            )
        returns = advantages + values
    return advantages, returns
