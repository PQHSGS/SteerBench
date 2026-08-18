# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import functools
import logging
import time
import typing as t

import multiprocess
import numpy as np
import torch
from scipy.optimize import minimize
from torch import nn


def sigmoid_np(x):
    e = np.exp(x)
    return e / (1.0 + e)


class LinearProj(nn.Module):
    def __init__(
        self,
        dim: int,
        init_identity: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.w1 = nn.Parameter(torch.empty(size=(1, dim)))
        self.b1 = nn.Parameter(torch.empty(size=(1, dim)))
        self._init_params(init_identity)

    def _init_params(self, identity: bool):
        self.w1.data = torch.ones(1, self.dim) if identity else torch.randn(1, self.dim)
        self.b1.data = torch.zeros((1, self.dim))

    def use_grad(self, use: bool):
        self.w1.requires_grad_(use)
        self.b1.requires_grad_(use)

    def support(self, threshold) -> int:
        """Returns the number of dimensions in the map that have at leas w != 1 or b != 0."""
        num = (
            ((torch.abs(self.w1 - 1) > threshold) | (torch.abs(self.b1) > threshold))
            .sum()
            .item()
        )
        return num

    def forward(self, x: torch.Tensor, reverse: bool = False):
        assert x.shape[-1] == self.w1.shape[-1], f"{x.shape[-1]} != {self.w1.shape[-1]}"
        if not reverse:
            return x * self.w1 + self.b1
        else:
            return (x - self.b1) / (self.w1 + 1e-10)

    def register_mask_hooks(
        self, weights_mask: list[bool], biases_mask: list[bool]
    ) -> None:
        def mask_weight_grad(grad: torch.Tensor) -> torch.Tensor:
            grad = grad.clone()
            grad[:, weights_mask] = 0.0
            return grad

        def mask_bias_grad(grad: torch.Tensor) -> torch.Tensor:
            grad = grad.clone()
            grad[:, biases_mask] = 0.0
            return grad

        with torch.no_grad():
            self.w1.data[:, weights_mask] = 1.0
            self.b1.data[:, biases_mask] = 0.0

        self.w1.register_hook(mask_weight_grad)
        self.b1.register_hook(mask_bias_grad)

    def optimize(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict]:
        x, y = x.astype(np.float64), y.astype(np.float64)

        m_x = np.mean(x, axis=0, keepdims=True)
        m_y = np.mean(y, axis=0, keepdims=True)

        # Add small noise to prevent divisions by 0
        x += 1e-8 * np.random.randn(*x.shape)

        x_bar = x - m_x
        y_bar = y - m_y
        beta = np.sum((x_bar * y_bar), axis=0, keepdims=True) / np.sum(
            (x_bar**2), axis=0, keepdims=True
        )
        alpha = m_y - beta * m_x
        params = np.concatenate([beta, alpha], 0)
        beta = torch.tensor(beta, dtype=self.w1.dtype, device=self.w1.device)
        alpha = torch.tensor(alpha, dtype=self.w1.dtype, device=self.w1.device)
        self.load_state_dict(
            {
                "w1": beta,
                "b1": alpha,
            }
        )
        return params, {}


class MaskedLinearProj(nn.Module):
    def __init__(self, weights_mask: list[bool], biases_mask: list[bool], device: str):
        super().__init__()
        assert len(weights_mask) == len(biases_mask), (
            len(weights_mask),
            len(biases_mask),
        )
        weights = [
            nn.Parameter(torch.ones(1, 1), requires_grad=rg) for rg in weights_mask
        ]
        biases = [
            nn.Parameter(torch.zeros(1, 1), requires_grad=rg) for rg in biases_mask
        ]
        self.weights = nn.ParameterList(weights).to(device)
        self.biases = nn.ParameterList(biases).to(device)

    def forward(self, x: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        w1, b1 = self.w1, self.b1
        assert x.shape[-1] == w1.shape[-1], f"{x.shape[-1]} != {w1.shape[-1]}"
        if not reverse:
            return x * w1 + b1
        else:
            return (x - b1) / (w1 + 1e-10)

    def from_linear_proj(self, state_dict: t.Mapping[str, t.Any], **kwargs: t.Any):
        dim = len(self.weights)
        w1 = {f"weights.{i}": state_dict["w1"][:, i].unsqueeze(1) for i in range(dim)}
        b1 = {f"biases.{i}": state_dict["b1"][:, i].unsqueeze(1) for i in range(dim)}
        return self.load_state_dict({**w1, **b1}, **kwargs)

    @property
    def w1(self) -> torch.Tensor:
        return torch.hstack([w for w in self.weights])

    @property
    def b1(self) -> torch.Tensor:
        return torch.hstack([b for b in self.biases])


class NonLinearProjSigSum(nn.Module):
    def __init__(
        self,
        dim: int,
        init_identity: bool = False,
        **kwargs,
    ):
        super().__init__()
        if init_identity:
            self.w1 = nn.Parameter(torch.ones((1, dim)))
            self.w2 = nn.Parameter(torch.ones((1, dim)))
            self.b1 = nn.Parameter(torch.zeros((1, dim)))
            self.b2 = nn.Parameter(torch.zeros((1, dim)))
            self.w11 = nn.Parameter(torch.ones((1, dim)))
            self.w22 = nn.Parameter(torch.ones((1, dim)))
        else:
            self.w1 = nn.Parameter(torch.randn((1, dim)))
            self.w2 = nn.Parameter(torch.randn((1, dim)))
            self.b1 = nn.Parameter(torch.zeros((1, dim)))
            self.b2 = nn.Parameter(torch.zeros((1, dim)))
            self.w11 = nn.Parameter(torch.randn((1, dim)))
            self.w22 = nn.Parameter(torch.randn((1, dim)))

    def __str__(self):
        txt = f"{self.__class__.__name__}:\n"
        for name, param in self.named_parameters():
            txt += f" {name} {param.shape}\n"
        return txt

    def forward(self, x: torch.Tensor, reverse: bool = False):
        assert x.shape[-1] == self.w1.shape[-1]
        if not reverse:
            z1 = self.w11 * torch.sigmoid(x * self.w1 + self.b1)
            z2 = self.w22 * torch.sigmoid(x * self.w2 + self.b2)
            return z1 + z2
        else:
            raise NotImplementedError("Inverse not analytical.")

    def optimize(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x0: np.ndarray | None = None,
        opt: str = "L-BFGS-B",
        tol: float = 1e-9,
        repeats: int = 1,
        use_bounds: bool = False,
        num_workers: int = 16,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        tic = time.time()
        # Make sure we optimize with float64
        x, y = x.astype(np.float64), y.astype(np.float64)
        if x0 is not None:
            x0 = x0.astype(np.float64)

        history = []

        def mse(a, b):
            return np.mean((a - b) ** 2)

        def mae(a, b):
            return np.mean(np.abs(a - b))

        def fn_mae(w, i):
            z1 = w[4] * sigmoid_np(w[0] * x[:, i] + w[1])
            z2 = w[5] * sigmoid_np(w[2] * x[:, i] + w[3])
            x_proj = z1 + z2
            # diff = x_proj - y[:, i]  # dividing to avoid overflow
            return mae(x_proj, y[:, i])

        def fn_mse(w, i):
            z1 = w[4] * sigmoid_np(w[0] * x[:, i] + w[1])
            z2 = w[5] * sigmoid_np(w[2] * x[:, i] + w[3])
            x_proj = z1 + z2
            # diff = x_proj - y[:, i]  # dividing to avoid overflow
            return mse(x_proj, y[:, i])

        def jac_mae(w, i):
            p1 = w[0] * x[:, i] + w[1]
            p2 = w[2] * x[:, i] + w[3]
            z1 = w[4] * sigmoid_np(p1)
            z2 = w[5] * sigmoid_np(p2)
            x_proj = z1 + z2
            diff = x_proj - y[:, i]  # dividing to avoid overflow
            sign = np.sign(diff)
            d_w0 = np.mean(
                sign * w[4] * sigmoid_np(p1) * (1 - sigmoid_np(p1)) * x[:, i]
            )
            d_w1 = np.mean(sign * w[4] * sigmoid_np(p1) * (1 - sigmoid_np(p1)))
            d_w2 = np.mean(
                sign * w[5] * sigmoid_np(p2) * (1 - sigmoid_np(p2)) * x[:, i]
            )
            d_w3 = np.mean(sign * w[5] * sigmoid_np(p2) * (1 - sigmoid_np(p2)))
            d_w4 = np.mean(sign * sigmoid_np(p1))
            d_w5 = np.mean(sign * sigmoid_np(p2))
            return [d_w0, d_w1, d_w2, d_w3, d_w4, d_w5]

        def jac_mse(w, i):
            p1 = w[0] * x[:, i] + w[1]
            p2 = w[2] * x[:, i] + w[3]
            z1 = w[4] * sigmoid_np(p1)
            z2 = w[5] * sigmoid_np(p2)
            x_proj = z1 + z2
            diff = x_proj - y[:, i]  # dividing to avoid overflow
            d_w0 = np.mean(
                2 * diff * w[4] * sigmoid_np(p1) * (1 - sigmoid_np(p1)) * x[:, i]
            )
            d_w1 = np.mean(2 * diff * w[4] * sigmoid_np(p1) * (1 - sigmoid_np(p1)))
            d_w2 = np.mean(
                2 * diff * w[5] * sigmoid_np(p2) * (1 - sigmoid_np(p2)) * x[:, i]
            )
            d_w3 = np.mean(2 * diff * w[5] * sigmoid_np(p2) * (1 - sigmoid_np(p2)))
            d_w4 = np.mean(2 * diff * sigmoid_np(p1))
            d_w5 = np.mean(2 * diff * sigmoid_np(p2))
            return [d_w0, d_w1, d_w2, d_w3, d_w4, d_w5]

        def constraint1(w):
            return w[4]

        def constraint2(w):
            return w[5]

        def repeated_minimization(i, iters=repeats):
            params = []
            vals = []
            r = 10
            bounds = [
                [-r, r],
                [-r, r],
                [-r, r],
                [-r, r],
                [None, None],
                [None, None],
            ]
            for _ in range(iters):
                x_init = x0[i] if x0 is not None else np.random.randn(6)
                res = minimize(
                    fun=functools.partial(fn_mse, i=i),
                    jac=functools.partial(jac_mse, i=i),
                    # fun=functools.partial(fn_mae, i=i),
                    # jac=functools.partial(jac_mae, i=i),
                    # x0=x0[i] if x0 is not None else np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0]), #np.random.randn(6),
                    x0=x_init,
                    bounds=bounds if use_bounds else None,
                    # constraints=(
                    #     [
                    #         {"type": "ineq", "fun": constraint1},
                    #         {"type": "ineq", "fun": constraint2},
                    #     ]
                    #     if use_bounds
                    #     else ()
                    # ),
                    method=opt,
                    tol=tol,
                    options={"disp": False, "maxiter": 15000},
                )
                params.append(res.x)
                vals.append(res.fun)
            min_id = np.array(vals).argmin()
            # print(params[min_id])
            return params[min_id]

        with multiprocess.Pool(num_workers) as p:
            params = p.map(repeated_minimization, range(x.shape[-1]))
        # FIXME: Non multiprocess version for debugging
        # params = [
        #     repeated_minimization(i, iters=repeats)
        #     for i in range(x.shape[-1])
        # ]
        params = torch.tensor(
            np.array(params), dtype=self.w1.dtype, device=self.w1.device
        )
        self.load_state_dict(
            {
                "w1": params[:, 0].unsqueeze(0),
                "b1": params[:, 1].unsqueeze(0),
                "w2": params[:, 2].unsqueeze(0),
                "b2": params[:, 3].unsqueeze(0),
                "w11": params[:, 4].unsqueeze(0),
                "w22": params[:, 5].unsqueeze(0),
            }
        )
        logging.info(f"Optimization took: {time.time() - tic:0.2f}s")
        return params.cpu().numpy(), {"history": np.array(history)}
