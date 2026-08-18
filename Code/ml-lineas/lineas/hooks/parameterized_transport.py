# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import abc
import copy
import logging
import typing as t

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from lineas.hooks.intervention_hook import InterventionHook
from lineas.optimal_transport.archs import LinearProj
from lineas.optimal_transport.ot_maps import solve_ot_1d
from lineas.utils.quantiles import compute_quantiles

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def mse_loss(x: torch.Tensor, y: torch.Tensor) -> list[torch.Tensor]:
    return [
        torch.mean(torch.pow((x - y), 2)),
    ]


class XYDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)


def optimize_loop(
    x: torch.Tensor,
    y: torch.Tensor,
    net: nn.Module,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int | None = None,
    loss_fn: t.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = mse_loss,
):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    dataset = XYDataset(x, y)
    if batch_size is not None and batch_size < len(x):
        loader = DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=0, shuffle=True
        )
    else:
        loader = DataLoader(
            dataset=dataset, batch_size=len(x), num_workers=0, shuffle=False
        )
    net.train()
    for epoch in range(epochs):
        # print(epoch)
        for _i, batch in enumerate(loader):
            optimizer.zero_grad()
            x_batch, y_batch = batch
            x_proj = net(x_batch)
            loss_terms = loss_fn(x_proj, y_batch)
            loss = sum(loss_terms)
            # Backprop
            loss.backward()
            optimizer.step()
        if epoch == 0 or (epoch % 50) == 0 or epoch == (epochs - 1):
            loss_items_float = [f"{loss_term.item():0.5f}" for loss_term in loss_terms]
            print(
                f"epoch {epoch}, loss {loss.item():0.5f} [{', '.join(loss_items_float)}]"
            )
    return net


class ParameterizedOTHook(abc.ABC, InterventionHook):
    def __init__(
        self,
        module_name: str,
        device: str = None,
        intervention_position: str = "all",
        dtype: torch.dtype = torch.float32,
        strength: float = 1.0,
        std_eps: float = 1e-4,
        quantiles_src: str = "q_all",
        **kwargs,
    ):
        """
        Gaussian Optimal Transport hook. Assumes the output of each neuron is independent and Gaussianly distributed.
        Transports from N(mu1, std1) to N(mu2, std2) if hook_forward is True, and N(mu2, std2) to N(mu1, std1) otherwise.
        Args:
            module_name (str): Module name on which the hook is applied.
            device (torch.device): Torch device.
            **kwargs: Any extra arguments, for compatibility with other hooks.
        """
        super().__init__(
            module_name=module_name,
            device=device,
            intervention_position=intervention_position,
            dtype=dtype,
            **kwargs,
        )

        self.strength = float(strength)
        self.std_eps = float(std_eps)
        self.quantiles_src = str(quantiles_src)

        # Register non-buffer placeholders
        self.quantiles_dict_src = None
        self.net = None

    def __str__(self):
        txt = (
            f"LearnableOT("
            f"module_name={self.module_name}, "
            f"quantiles_src={self.quantiles_src}, "
            f"strength={self.strength:0.2f}, "
            f")"
        )
        return txt

    @abc.abstractmethod
    def get_net(self) -> type[torch.nn.Module]:
        pass

    def state_dict(self, *args, **kwargs) -> dict:
        d = super().state_dict()
        # Merge non-buffers into dict
        d.update(
            {
                "quantiles_dict_src": self.quantiles_dict_src,
            }
        )
        # Merge hook_args required for incremental learning
        d.update(
            {
                "quantiles_src": self.quantiles_src,
            }
        )
        return d

    def load_state_dict(
        self,
        state_dict: t.Mapping[str, t.Any],
        strict: bool = True,
        assign: bool = False,
    ):
        # Fill in non buffers
        self.quantiles_dict_src = {
            k: [v[0].to(self.device), v[1].to(self.device)]
            for k, v in state_dict["quantiles_dict_src"].items()
        }

        # Fill in the net
        net_type = self.get_net()

        dim = state_dict["net.w1"].shape[-1]
        self.net = net_type(dim=dim)

        # This loads the net statedict.
        net_dict = {k: v for k, v in state_dict.items() if k.startswith("net.")}
        super().load_state_dict(net_dict)
        self.net = self.net.to(self.device)

        if not torch.allclose(self.net.w1, state_dict["net.w1"]) or (
            "net.A" in state_dict
            and not torch.allclose(self.net.A, state_dict["net.A"])
        ):
            msg = "Net load_state_dict() failed.\n"
            msg += "self.net:\n"
            for k, v in self.net.named_parameters():
                msg += f"{k}  {v.shape}\n"
            raise RuntimeError(msg)

        # Non tensor parameters. If these exist in state_dict, they will override the ones passed through constructor.
        # This is useful for hooks learnt for some specific args (eg. incremental hooks).
        for hook_arg in [attr for attr in dir(self) if attr.startswith("hook_")]:
            if hook_arg in state_dict:
                # We know the type, __init__() cas been called.
                vartype = type(getattr(self, hook_arg))
                varval = vartype(state_dict[hook_arg])
                logging.warning(
                    f"Overriding {hook_arg} to {varval} in {self.__class__.__name__}."
                )
                setattr(self, hook_arg, varval)
        self._post_load()

    @abc.abstractmethod
    def _fit_net(
        self,
        z_src_ot: torch.Tensor,
        z_dst_ot: torch.Tensor,
        use_gd: bool = False,
        **kwargs,
    ):
        pass

    def fit(
        self, responses: torch.Tensor, labels=torch.Tensor, use_gd=False, **kwargs
    ) -> None:
        z_f64 = responses.to(torch.float32)
        labels = labels.to(torch.bool)
        z_src = z_f64[labels]  # Bs, D
        z_dst = z_f64[~labels]  # Bd, D
        dim = z_src.shape[-1]

        # Find OT pairs "PER NEURON" (not jointly!).
        z_src_ot, z_dst_ot = torch.empty_like(z_src), torch.empty_like(z_dst)  # B, S
        for idx in range(dim):
            z_src_ot[:, idx], z_dst_ot[:, idx] = solve_ot_1d(
                z_src[:, idx], z_dst[:, idx]
            )  # D

        logger.info(
            f"Transport data after filtering {z_src_ot.shape} --> {z_dst_ot.shape}"
        )

        self.quantiles_dict_src = compute_quantiles(z_f64[labels])
        self._fit_net(
            z_src_ot=z_src_ot,
            z_dst_ot=z_dst_ot,
            use_gd=kwargs.get("use_gd", False),
            **kwargs,
        )
        self.net = self.net.to(self.dtype).to(self.device)
        self._post_load()

    def _post_load(self) -> None:
        """
        This method should be called after loading the states of the hook.
        So calls must be placed at the end of .fit() and at the end of .load_state_dict().
        """
        super()._post_load()

        # TODO: Move this to some post_load method to avoid doing it every time.
        if self.quantiles_src == "q_all":
            self.quantiles_src = [-1e6, 1e6]
        else:
            self.quantiles_src = copy.deepcopy(
                self.quantiles_dict_src[self.quantiles_src]
            )
            self.quantiles_src[0] = self.quantiles_src[0].view(1, -1)
            self.quantiles_src[1] = self.quantiles_src[1].view(1, -1)

    def forward(self, module, input, output) -> t.Any:
        output_shape = output.shape
        output_orig = output
        z_unit = output

        # Quantile filtering
        ot_mask = (
            (self.quantiles_src[0] < z_unit) & (z_unit < self.quantiles_src[1])
        ).to(z_unit.dtype)

        # Transport the outputs
        z_ot = self.net(z_unit, reverse=False).to(z_unit.dtype)

        # Apply transport with specific strength
        z_ot = self.strength * z_ot + (1 - self.strength) * z_unit

        output = ot_mask * z_ot + (1 - ot_mask) * output_orig
        return output


class LinearAcTHook(ParameterizedOTHook):
    def get_net(self) -> type[LinearProj]:
        return LinearProj

    def _fit_net(
        self, z_src_ot: torch.Tensor, z_dst_ot: torch.Tensor, use_gd=False, **kwargs
    ):
        # The linear case
        dim = z_src_ot.shape[-1]
        self.net = LinearProj(dim=dim).to(self.device).to(torch.float32)
        if use_gd:
            self.net = optimize_loop(
                z_src_ot,
                z_dst_ot,
                self.net,
                epochs=kwargs.get("epochs"),
                lr=kwargs.get("learning_rate"),
            )
        else:
            opt_params, opt_extra = self.net.optimize(
                z_src_ot.cpu().numpy(),
                z_dst_ot.cpu().numpy(),
            )
