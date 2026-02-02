import numpy as np
import torch

from rl_blox.logging.logger import LoggerBase


class Loss:
    def __init__(
        self,
        loss: torch.Tensor,
        pg_loss: torch.Tensor,
        v_loss: torch.Tensor,
        entropy_loss: torch.Tensor,
        old_approx_kl: torch.Tensor,
        approx_kl: torch.Tensor,
    ):
        self.loss = loss
        self.pg_loss = pg_loss
        self.v_loss = v_loss
        self.entropy_loss = entropy_loss
        self.old_approx_kl = old_approx_kl
        self.approx_kl = approx_kl

    def print(self, logger: LoggerBase, global_step: int):
        logger.record_stat("value_loss", self.v_loss.item(), step=global_step)
        logger.record_stat("policy_loss", self.pg_loss.item(), step=global_step)
        logger.record_stat(
            "entropy",
            self.entropy_loss.item(),
            step=global_step,
        )
        logger.record_stat(
            "old_approx_kl",
            self.old_approx_kl.item(),
            step=global_step,
        )
        logger.record_stat("approx_kl", self.approx_kl.item(), step=global_step)
