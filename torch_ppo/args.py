import os
import time
from dataclasses import dataclass

import numpy as np
import torch

# import tyro


# @dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    exp_name: str = "PpoStandalone"

    # Meta specific arguments
    total_meta_iterations: int = 10000
    meta_learning_rate: float = 3e-4
    inner_learning_rate: float = 3e-2
    anneal_meta_lr: bool = True
    anneal_inner_lr: bool = True
    inner_learning_rate_goal: float = 3e-6
    inner_learning_rate_anneal_steps: float = 50
    num_adaptation_steps: int = 1
    train_set_size: int = 15
    test_set_size: int = 3
    eval_freq: int = 100
    eval_len: int = 50
    save_checkpoints = True

    # Selectors
    uniform_start_duration: int = 100
    hard_task_scale = 3.0
    ins_scale = 3.0

    # Cheetah specific arguments
    target_velocity_min: float = 0.0
    target_velocity_max: float = 2.0

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v5"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_ppo_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.5
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    velocities: list[float] = []


def init_args(seed: int) -> tuple[Args, str, torch.device]:
    # args = tyro.cli(Args)
    args = Args()
    args.seed = seed
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = (
        f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    )
    args.velocities = np.random.uniform(
        args.target_velocity_min, args.target_velocity_max, args.train_set_size
    ).tolist()
    return args, run_name, device
