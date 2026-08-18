# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import typing as t

import torch
from lineas.optimal_transport.wasserstein import wasserstein_trim_sort


def get_loss(name: str) -> t.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    This method allows safely choosing implemented losses between predictions and targets.
    """
    loss_map = {
        "wasserstein": lambda x, y, p: wasserstein_trim_sort(x, y, p=p).mean(),
    }
    return loss_map[name]


def get_optimizer(name: str) -> torch.optim.Optimizer:
    return getattr(torch.optim, name)
