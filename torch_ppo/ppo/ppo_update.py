import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agent import Agent

from rl_blox.logging.logger import LoggerBase

from .storage import DataHolder, RunData
from .update.loss import Loss
from .update.ppo_loss_calculator import calculate_loss


def log_progress(
    logger: LoggerBase,
    global_step: int,
    lr: float,
    loss: Loss,
    b_values: torch.Tensor,
    b_returns: torch.Tensor,
    clipfracs: list,
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
    logger.record_stat("explained_variance", explained_var, step=global_step)
    logger.record_stat(
        "clipfrac",
        np.mean(clipfracs),
        step=global_step,
    )
    loss.print(logger, global_step)


def update_agent(
    agent: Agent,
    optimizer: optim.Optimizer,
    data_holder: DataHolder,
    logger: LoggerBase,
    run_data: RunData,
    is_inner_optimizer: bool,
    return_first_loss: bool = None,
) -> Loss:
    args = data_holder.args
    if return_first_loss is None:
        return_first_loss = is_inner_optimizer
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

    if return_first_loss:
        loss, _ = calculate_loss(
            agent,
            b_obs,
            b_logprobs,
            b_actions,
            b_advantages,
            b_returns,
            b_values,
            args,
            clipfracs,
        )
        return loss

    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            loss, clipfracs = calculate_loss(
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

            if is_inner_optimizer:
                optimizer.step(loss.loss)
            else:
                optimizer.zero_grad()
                loss.loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        if args.target_kl is not None:
            if loss.approx_kl > args.target_kl:
                break

    if logger is not None:
        log_progress(
            logger,
            run_data.global_step,
            optimizer.param_groups[0]["lr"],
            loss,
            b_values,
            b_returns,
            clipfracs,
        )

    return loss
