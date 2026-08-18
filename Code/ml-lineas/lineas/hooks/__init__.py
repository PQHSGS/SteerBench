# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

from .aura_hook import AURAHook
from .backprop_transport import BPLinearHook, BPNonLinearHook
from .gaussian_transport import GaussianAcTHook, OnlyMeanHook
from .identity import IdentityHook
from .intervention_hook import InterventionHook
from .parameterized_transport import LinearAcTHook
from .postprocess_and_save_hook import PostprocessAndSaveHook
from .responses_hook import ResponsesHook
from .return_outputs_hook import ReturnOutputsHook

HOOK_REGISTRY = {
    "postprocess_and_save": PostprocessAndSaveHook,
    "return_outputs": ReturnOutputsHook,
    "aura": AURAHook,
    "mean_act": OnlyMeanHook,
    "gaussian_act": GaussianAcTHook,
    "linear_act": LinearAcTHook,
    "lineas": BPLinearHook,
    "identity": IdentityHook,
    "none": IdentityHook,
}


def get_hook(name: str, *args, **kwargs) -> ResponsesHook | InterventionHook:
    hook_cls = HOOK_REGISTRY[name]
    hook = hook_cls(*args, **kwargs)
    return hook


def add_hook(name: str, hook_class: ResponsesHook | InterventionHook) -> None:
    if name in HOOK_REGISTRY:
        raise ValueError(f"Hook {name} already exists")

    HOOK_REGISTRY[name] = hook_class
