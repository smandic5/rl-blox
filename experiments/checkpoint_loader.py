import os
import re

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx

from .factory_getters import get_policies
from .hparams import checkpoint_path


def get_all_checkpoint_str():
    try:
        return os.listdir(checkpoint_path)
    except FileNotFoundError:
        print("The specified path does not exist.")
        return []
    except PermissionError:
        print("You do not have permission to access this path.")
        return []


def parse_checkpoint_str(checkpoint_str: str):
    parts = checkpoint_str.split("_")

    env = parts[0]

    algo = "_".join(parts[1:3])  # METAALG_BASEALG_TS_SEED
    ts = parts[3]
    seed = parts[4]

    is_actor = "ACTOR" in checkpoint_str

    step_match = re.search(r"step_(\d+)", checkpoint_str)
    step = int(step_match.group(1)) if step_match else None

    epoch_match = re.search(r"epoch_(\d+)", checkpoint_str)
    epoch = int(epoch_match.group(1)) if epoch_match else None

    name = checkpoint_str.split(".")[0]

    return env, algo, int(ts), int(seed), is_actor, int(step), int(epoch), name


def load_weights(model: nnx.Module, name: str):
    checkpointer = ocp.StandardCheckpointer()
    path = f"{os.path.abspath(checkpoint_path)}/{name}"
    state = checkpointer.restore(path, target=nnx.state(model))
    nnx.update(model, state)


def load_models(
    key: jnp.ndarray, features: int, actions: int, folder_name: str
):
    actor, critic, _, _ = get_policies(
        features=features, actions=actions, key=key
    )
    load_weights(actor, folder_name)
    load_weights(critic, folder_name.replace("ACTOR", "CRITIC"))

    return actor, critic
