# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agent import Agent
from args import Args

from rl_blox.logging.logger import LoggerBase


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
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
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

    return (
        loss,
        pg_loss,
        v_loss,
        entropy_loss,
        clipfracs,
        old_approx_kl,
        approx_kl,
    )


def update_agent(
    agent: Agent,
    optimizer: optim.Optimizer,
    b_obs: torch.Tensor,
    b_logprobs: torch.Tensor,
    b_actions: torch.Tensor,
    b_advantages: torch.Tensor,
    b_returns: torch.Tensor,
    b_values: torch.Tensor,
    args: Args,
    logger: LoggerBase,
    global_step: int,
):
    b_inds = np.arange(args.batch_size)
    clipfracs = []
    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            (
                loss,
                pg_loss,
                v_loss,
                entropy_loss,
                clipfracs,
                old_approx_kl,
                approx_kl,
            ) = calculate_loss(
                agent,
                b_obs[mb_inds],
                b_logprobs[mb_inds],
                b_actions[mb_inds],
                b_advantages[mb_inds],
                b_returns[mb_inds],
                b_values[mb_inds],
                args,
                clipfracs,
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

        if args.target_kl is not None and approx_kl > args.target_kl:
            break

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = (
        np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    )

    logger.record_stat(
        "learning_rate",
        optimizer.param_groups[0]["lr"],
        step=global_step,
    )
    logger.record_stat("value_loss", v_loss.item(), step=global_step)
    logger.record_stat("policy_loss", pg_loss.item(), step=global_step)
    logger.record_stat(
        "entropy",
        entropy_loss.item(),
        step=global_step,
    )
    logger.record_stat(
        "old_approx_kl",
        old_approx_kl.item(),
        step=global_step,
    )
    logger.record_stat("approx_kl", approx_kl.item(), step=global_step)
    logger.record_stat(
        "clipfrac",
        np.mean(clipfracs),
        step=global_step,
    )
    logger.record_stat("explained_variance", explained_var, step=global_step)

    return clipfracs, old_approx_kl, approx_kl, pg_loss, v_loss, entropy_loss
