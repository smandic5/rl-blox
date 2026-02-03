import copy
import random

import higher
import numpy as np
import torch
import tyro
from agent import Agent
from args import Args
from envs.env_sets import init_env_sets
from task_selectors.ins.higher_to_torch import copy_from_fast


def swap_rows(weights: torch.Tensor, m: int, n: int):
    weights[[m, n]] = weights[[n, m]]


def compare_module(
    target: torch.nn.Sequential,
    to_align: torch.nn.Sequential,
):
    num_layers = len(target)
    for layer_i in range(num_layers):
        if type(to_align[layer_i]) != torch.nn.Linear:
            continue

        with torch.no_grad():
            to_align_layer: torch.nn.Linear = to_align[layer_i]
            target_layer: torch.nn.Linear = to_align[layer_i]

            w_target = target_layer.weight
            w_to_align = to_align_layer.weight

            print("--------------------")
            print(to_align_layer(torch.ones(w_target.shape[1])))
            print("----")
            print(w_target.shape)
            print(w_to_align.shape)
            print("----")

            print(w_target)
            swap_rows(w_target, 0, 1)
            print(w_target)

            print("----")

            print(to_align_layer(torch.ones(w_target.shape[1])))
            print(target_layer(torch.ones(w_to_align.shape[1])))

    pass


def compare_agents(
    target=Agent,
    to_align=Agent,
) -> Agent:
    return compare_module(target.critic, to_align.critic)


def init_seeds(args: Args):
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic


if __name__ == "__main__":
    args = tyro.cli(Args)
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
    print(compare_agents(agent_new, agent2))
