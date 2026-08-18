# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.


import torch


def solve_ot_1d(p: torch.Tensor, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    OT 1D for same number of points amounts to sorting.
    """
    assert len(p) == len(q), (
        "Very simple 1D OT matching for now. "
        "Please use the same number of samples for p, q."
    )
    p_sort, _ = torch.sort(p, 0)
    q_sort, _ = torch.sort(q, 0)
    return p_sort, q_sort
