import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agent import Agent
from differentiable_sgd import DifferentiableSGD
from ppo_loss import calculate_loss
from storage import DataHolder, RunData

from rl_blox.logging.logger import LoggerBase


def update_agent(
    agent: Agent,
    optimizer: DifferentiableSGD,
    data_holder: DataHolder,
    logger: LoggerBase,
    run_data: RunData,
):
    args = data_holder.args
    clipfracs = []
    b_inds = np.arange(args.batch_size)
    (
        b_obs,
        b_actions,
        b_logprobs,
        b_values,
        b_advantages,
        b_returns,
    ) = data_holder.get_batch(agent, run_data)

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

    log_loss(
        logger,
        run_data.global_step,
        optimizer.lr,
        clipfracs,
        b_values,
        b_returns,
        pg_loss,
        v_loss,
        entropy_loss,
        old_approx_kl,
        approx_kl,
    )

    return clipfracs, old_approx_kl, approx_kl, pg_loss, v_loss, entropy_loss


def log_loss(
    logger: LoggerBase,
    global_step: int,
    lr: float,
    clipfracs: float,
    b_values: torch.Tensor,
    b_returns: torch.Tensor,
    pg_loss: float,
    v_loss: float,
    entropy_loss: float,
    old_approx_kl: float,
    approx_kl: float,
):
    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = (
        np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    )

    logger.record_stat(
        "learning_rate",
        lr,
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
