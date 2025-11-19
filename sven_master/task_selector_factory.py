from typing import Type

import jax.numpy as jnp

from rl_blox.blox.multitask import TaskSelector


def create_task_selector(
    selector_class: Type[TaskSelector],
    tasks,
    key: jnp.ndarray,
    envs: list = [],
    policies: list = [],
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
    )
    return selector
