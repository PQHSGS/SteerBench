# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import math
import typing as t

import torch
from torch import nn

__all__ = [
    "L1",
    "L2",
    "PostComposition",
    "PreComposition",
    "SparseGroupLasso",
    "apply_sparse_group_lasso",
    "apply_l1",
]

from lineas.hooks import InterventionHook

Arrays = torch.Tensor | list[torch.Tensor]


class L1(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(x, ord=1, dim=-1)

    def prox(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        return torch.sign(x) * torch.relu(torch.abs(x) - tau)


class L2(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(x, ord=2, dim=-1)

    def prox(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        norm_x = torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
        prox_x = torch.relu(1.0 - (tau / norm_x)) * x
        # handle `tau=0` and `norm_x=0`
        return torch.where(torch.isnan(prox_x), 0.0, prox_x)


class PostComposition(nn.Module):
    def __init__(self, f: t.Any, alpha: float = 1.0, b: float = 0.0):
        super().__init__()
        # https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf, section 2.2
        self.f = f
        self.alpha = alpha
        self.b = b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.f(x) + self.b

    def prox(self, v: torch.Tensor, tau: float = 1.0) -> Arrays:
        return self.f.prox(v, tau=tau * self.alpha)


class PreComposition(nn.Module):
    def __init__(self, f: t.Any, alpha: float = 1.0, b: float = 0.0):
        super().__init__()
        # https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf, section 2.2
        self.f = f
        self.alpha = alpha
        self.b = b

    def forward(self, x: Arrays) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return self.f(self.alpha * x + self.b)
        # for sparse group lasso, etc.
        tmp = [(self.alpha * arr + self.b) for arr in x]
        return self.f(tmp)

    def prox(self, v: Arrays, tau: float = 1.0) -> Arrays:
        if isinstance(v, torch.Tensor):
            prox_f = self.f.prox(self.alpha * v + self.b, tau=tau * (self.alpha**2))
            return (1.0 / self.alpha) * (prox_f - self.b)

        tmp = [(self.alpha * arr + self.b) for arr in v]
        prox_f = self.f.prox(tmp, tau=tau * (self.alpha**2))
        return [(1.0 / self.alpha) * (arr - self.b) for arr in prox_f]


class SparseGroupLasso(nn.Module):
    def __init__(self, *, lam_l1: float = 1.0, lam_l2: float = 1.0):
        super().__init__()
        self.prox_l1 = PostComposition(L1(), alpha=lam_l1)
        self.prox_l2 = PostComposition(L2(), alpha=lam_l2)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        l1_norms, l2_norms = [], []
        for group in x:  # assumes non-overlapping groups
            dim = group.shape[-1]
            prox_l2 = PostComposition(self.prox_l2, alpha=math.sqrt(dim))
            l1_norms.append(self.prox_l1(group))
            l2_norms.append(prox_l2(group))
        reg = sum(l1_norms) + sum(l2_norms)
        return reg

    def prox(self, x: list[torch.Tensor], tau: float = 1.0) -> list[torch.Tensor]:
        res = []
        for group in x:
            dim = group.shape[-1]
            prox_l2 = PostComposition(self.prox_l2, alpha=math.sqrt(dim))
            prox = self.prox_l1.prox(group, tau=tau)
            prox = prox_l2.prox(prox, tau=tau)
            res.append(prox)
        return res


@torch.no_grad()
def apply_sparse_group_lasso(
    hooks: list[InterventionHook],
    tau: float,
    *,
    lam_l1_weight: float,
    lam_l2_weight: float,
    lam_l1_bias: float | None = None,
    lam_l2_bias: float | None = None,
) -> float:
    if lam_l1_bias is None:
        lam_l1_bias = lam_l1_weight
    if lam_l2_bias is None:
        lam_l2_bias = lam_l2_weight

    prox_weight = SparseGroupLasso(lam_l1=lam_l1_weight, lam_l2=lam_l2_weight)
    prox_weight = PreComposition(prox_weight, b=-1.0)
    prox_bias = SparseGroupLasso(lam_l1=lam_l1_bias, lam_l2=lam_l2_bias)

    ws, bs = [], []
    for hook in hooks:
        ws.append(hook.net.w1)
        bs.append(hook.net.b1)
    prox_ws = prox_weight.prox(ws, tau=tau)
    prox_bs = prox_bias.prox(bs, tau=tau)

    for hook, prox_w, prox_b in zip(hooks, prox_ws, prox_bs, strict=True):
        hook.net.w1.data = prox_w.data
        hook.net.b1.data = prox_b.data

    reg_ws = prox_weight(ws)
    reg_bs = prox_bias(bs)
    reg = reg_ws + reg_bs

    return reg.item()


@torch.no_grad()
def apply_l1(
    hooks: list[InterventionHook],
    tau: float,
    *,
    lam_l1_weight: float,
    lam_l1_bias: float | None = None,
) -> float:
    if lam_l1_bias is None:
        lam_l1_bias = lam_l1_weight
    l1_prox_weight = PostComposition(L1(), alpha=lam_l1_weight)
    l1_prox_weight = PreComposition(l1_prox_weight, b=-1.0)
    l1_prox_bias = PostComposition(L1(), alpha=lam_l1_bias)

    reg = 0.0
    for hook in hooks:
        hook.net.w1.data = l1_prox_weight.prox(hook.net.w1, tau=tau).data
        hook.net.b1.data = l1_prox_bias.prox(hook.net.b1, tau=tau).data
        # Get the prox grad values for logging purposes only
        reg += l1_prox_weight(hook.net.w1) + l1_prox_bias(hook.net.b1)

    return reg.item()
