from typing import Type

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np

from rl_blox.blox.multitask import TaskSelector


def create_task_selector(
    selector_class: Type[TaskSelector],
    tasks,
    key: jnp.ndarray,
    envs: list = [],
    policies: list = [],
    max_reward: float = 1.0,
    prefer_similar: bool = True,
    choose_from_last_pick: bool = False,
) -> TaskSelector:
    selector = selector_class(
        tasks,
        key=key,
        envs=envs,
        policies=policies,
        prefer_similar=prefer_similar,
        choose_from_last_pick=choose_from_last_pick,
        max_reward=max_reward,
    )
    return selector


def get_task_selector(
    hparams_task_selector: dict,
    hparams_algorithm: dict,
    key: jnp.ndarray,
    actor: nnx.Module,
    envs: list,
):
    task_selector = create_task_selector(
        hparams_task_selector["selector_class"],
        np.arange(hparams_algorithm["set_size_train"]),
        key,
        envs,
        [nnx.clone(actor) for _ in range(hparams_algorithm["set_size_train"])],
        prefer_similar=hparams_task_selector["prefer_similar"],
        choose_from_last_pick=hparams_task_selector["choose_from_last_pick"],
        max_reward=hparams_task_selector["max_reward"],
    )

    return task_selector
