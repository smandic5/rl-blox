import copy
from typing import TypeVar

import higher
import torch

T = TypeVar("Net", bound=torch.nn.Module)


def _state_dict_from_fast(
    module: T, fmodule: higher.patch._MonkeyPatchBase
) -> dict:
    state_dict = {}
    fast_params = list(fmodule.fast_params)
    idx = 0

    for name, param in module.named_parameters():
        state_dict[name] = fast_params[idx].detach().clone()
        idx += 1

    if idx != len(fast_params):
        raise RuntimeError("Mismatch between fast params and module params")

    return state_dict


def copy_from_fast(
    original_module: T, fast_model: higher.patch._MonkeyPatchBase
) -> T:
    new_module = copy.deepcopy(original_module)
    new_module.load_state_dict(
        _state_dict_from_fast(new_module, fast_model), strict=True
    )
    return new_module
