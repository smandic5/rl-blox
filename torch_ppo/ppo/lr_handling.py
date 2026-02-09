import numpy as np
import torch.optim as optim
from args import Args


def fix_anneal(args: Args, iteration: int) -> float:
    progress = np.clip(
        [(iteration - 1) / args.inner_learning_rate_anneal_steps], 0, 1
    )[0]
    start = args.inner_learning_rate * (1 - progress)
    end = args.inner_learning_rate_goal * progress
    return start + end


def constant_anneal(
    args: Args, iteration: int, max_iterations: int, base_lr: float
) -> float:
    frac = 1.0 - (iteration - 1.0) / (args.eval_len * 2)  # max_iterations
    return frac * base_lr


def lr_annealing(
    args: Args,
    optimizer: optim.Optimizer,
    iteration: int,
    max_iterations: int,
    uses_inner_lr: bool = False,
):
    if uses_inner_lr:
        # lrnow = fix_anneal(args, iteration)
        lrnow = constant_anneal(
            args, iteration, max_iterations, args.inner_learning_rate
        )
    else:
        lrnow = constant_anneal(
            args, iteration, max_iterations, args.learning_rate
        )
    optimizer.param_groups[0]["lr"] = lrnow
