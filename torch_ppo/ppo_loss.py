import torch
from agent import Agent
from args import Args
from loss import Loss


def calculate_loss(
    agent: Agent,
    b_obs: torch.Tensor,
    b_logprobs: torch.Tensor,
    b_actions: torch.Tensor,
    b_advantages: torch.Tensor,
    b_returns: torch.Tensor,
    b_values: torch.Tensor,
    args: Args,
    clipfracs: list,
) -> tuple[Loss, list]:
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
        b_obs, b_actions
    )
    logratio = newlogprob - b_logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfracs += [
            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
        ]

    mb_advantages = b_advantages
    if args.norm_adv:
        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
            mb_advantages.std() + 1e-8
        )

    # Policy loss
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(
        ratio, 1 - args.clip_coef, 1 + args.clip_coef
    )
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - b_returns) ** 2
        v_clipped = b_values + torch.clamp(
            newvalue - b_values,
            -args.clip_coef,
            args.clip_coef,
        )
        v_loss_clipped = (v_clipped - b_returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        v_loss = 0.5 * ((newvalue - b_returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    loss_container = Loss(
        loss, pg_loss, v_loss, entropy_loss, old_approx_kl, approx_kl
    )
    return loss_container, clipfracs
