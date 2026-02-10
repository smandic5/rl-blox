from typing import Callable

import gymnasium as gym
import torch


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = False,
    gamma: float = 0.99,
):
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, 0, capture_video, run_name, gamma)]
    )
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    ep_rew = 0
    ep_i = 0
    while len(episodic_returns) < eval_episodes:
        actions, _, _, _ = agent.get_action_and_value(
            torch.Tensor(obs).to(device)
        )
        next_obs, reward, terminations, truncations, infos = envs.step(
            actions.cpu().numpy()
        )
        dones = terminations or truncations
        ep_rew += reward[0]
        if dones[0] != 0:
            print(f"episode={ep_i}, episodic_return={ep_rew}")
            ep_rew = 0
            ep_i += 1
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(
                    f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}"
                )
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns


def checkpoint(
    agent: Agent,
    data_holder: DataHolder,
    test_set: list[gym.vector.SyncVectorEnv],
    args: Args,
    iteration: int,
    logger: LoggerBase = None,
    run_name: str = None,
):
    # if run_name is not None and False:
    #    save_model(args, run_name, agent)
    #    from checkpoint import load_model#

    #    a = load_model(run_name, args, test_set[0], data_holder.device)

    print("Evaluation started")
    for test_env_i, envs in enumerate(test_set):
        eval_logger = AIMLogger()
        eval_logger.define_experiment(
            env_name="Cheetah",
            algorithm_name=f"TorchMamlPPO_Eval_Env{test_env_i}",
            hparams=vars(args) | {iteration: iteration},
        )
        eval_logger.start_new_episode()

        agent_clone = Agent(envs)
        agent_clone.load_state_dict(agent.state_dict())
        optimizer_clone = torch.optim.SGD(
            agent_clone.parameters(), lr=args.inner_learning_rate
        )
        inner_loss, rewards = train_ppo(
            agent_clone,
            envs,
            optimizer_clone,
            data_holder,
            num_iteration=args.eval_len,
            uses_inner_lr=True,
            logger=eval_logger,
        )
        if logger is not None:
            inner_loss.print(logger, iteration)
            adapting_reward = np.mean([np.mean(r) for r in rewards[:-1]])
            adapted_reward = np.mean(rewards[-1])
            logger.record_stat(
                f"Adapting_Reward_T{test_env_i}",
                adapting_reward,
                step=iteration,
            )
            logger.record_stat(
                f"Adapted_Reward_T{test_env_i}", adapted_reward, step=iteration
            )
        eval_logger.run.close()
    print("Evaluation ended.")
