import copy
import random

import higher
import numpy as np
import scipy.optimize
import torch

# import tyro
from agent import Agent
from args import Args
from envs.env_sets import init_env_sets
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from .higher_to_torch import copy_from_fast


def swap_rows(weights: torch.Tensor, m: int, n: int):
    weights[[m, n]] = weights[[n, m]]


def swap_columns(weights: torch.Tensor, m: int, n: int):
    weights[:, [m, n]] = weights[:, [n, m]]


def swap_perceptron(model: list[torch.nn.Linear], layer_i: int, m: int, n: int):
    swap_rows(model[layer_i].weight, m, n)
    swap_columns(model[layer_i + 1].weight, m, n)


def compare_weights(
    w1: torch.Tensor, w2: torch.Tensor, method: str = "cos"
) -> tuple[torch.Tensor, np.ndarray]:

    if method == "cos":
        result = cosine_similarity(w1, w2)
        negated = result < 0
        result = np.abs(result)
    elif method == "euclid":
        result_normal = euclidean_distances(w1, w2)
        result_negated = euclidean_distances(w1, -w2)
        negated = result_negated < result_normal
        result = torch.where(negated, result_negated, result_normal)
    else:
        raise Exception(f"Wrong matrix sim method selected: {method}")
    return result, negated


def compare_sequentials(
    target: torch.nn.Sequential,
    to_align: torch.nn.Sequential,
) -> float:
    target_layers = [
        layer for layer in target if type(layer) == torch.nn.Linear
    ]
    to_align_layers = [
        layer for layer in to_align if type(layer) == torch.nn.Linear
    ]
    num_layers = len(target_layers)

    difference = 0

    for layer_i in range(num_layers - 1):
        with torch.no_grad():
            to_align_layer = to_align_layers[layer_i]
            w_target = target_layers[layer_i].weight
            w_to_align = to_align_layer.weight
            w_next = to_align_layers[layer_i + 1].weight

            distances, negations = compare_weights(w_target, w_to_align)
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(distances)
            to_negate = (
                torch.Tensor(
                    [negations[i, j] for i, j in zip(row_ind, col_ind)]
                )
                .bool()
                .reshape((-1, 1))
            )

            # swapping perceptons
            w_to_align = w_to_align[col_ind]
            w_next = w_next[:, col_ind]

            # negating perceptons
            w_to_align = torch.where(to_negate, -w_to_align, w_to_align)
            # negating unit output is not necessary as we are only measuring distancee
            # between models and are not otherwise using the cannonical forms

            to_align_layer.weight = torch.nn.Parameter(
                w_to_align, requires_grad=True
            )
            to_align_layers[layer_i + 1].weight = torch.nn.Parameter(
                w_next, requires_grad=True
            )

            difference += torch.mean((w_target - w_to_align) ** 2).item()

    return difference


def compare_agents(
    target=Agent,
    to_align=Agent,
) -> float:
    aligned_copy = copy.deepcopy(to_align)
    return compare_sequentials(
        target.critic, aligned_copy.critic
    ) + compare_sequentials(target.actor_mean, aligned_copy.actor_mean)


def init_seeds(args: Args):
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic


if __name__ == "__main__":
    args = Args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = "test"
    init_seeds(args)
    envs_train_set, envs_test_set = init_env_sets(args, run_name)
    agent1 = Agent(envs_train_set[0])
    opt = torch.optim.SGD(agent1.parameters(), 1)

    with higher.innerloop_ctx(agent1, opt) as (fmodel, fopt):
        inp = torch.ones((2, 17))
        o = fmodel.get_value(inp)
        loss = torch.sum(1 - o)
        fopt.step(loss)
        agent_new = copy_from_fast(agent1, fmodel)

    agent2 = Agent(envs_train_set[0])

    print(agent_new.get_value(inp))
    print(agent2.get_value(inp))

    print(compare_agents(agent_new, agent2))

    print(agent_new.get_value(inp))
    print(agent2.get_value(inp))
