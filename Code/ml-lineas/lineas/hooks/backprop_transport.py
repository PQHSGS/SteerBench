# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import abc
import logging
import typing as t

import torch

from lineas.hooks.intervention_hook import InterventionHook
from lineas.optimal_transport.archs import LinearProj, NonLinearProjSigSum

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BPHook(abc.ABC, InterventionHook):
    def __init__(
        self,
        module_name: str,
        dim: int = None,
        device: str = None,
        intervention_position: str = "all",
        dtype: torch.dtype = torch.float32,
        strength: torch.Tensor | float = 1.0,
        **kwargs,
    ):
        """
        A back-propagable (BP) hook (e.g., LinEAS)

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

        self.strength = torch.tensor(strength, dtype=self.dtype)

        if dim is None:
            # Register buffer placeholders, must be filled in when loading state_dict
            self.register_buffer(
                "dim", torch.tensor(1, dtype=torch.int64, device=device)
            )
            self.net = None
        else:
            # Instantiate net directly
            self.register_buffer(
                "dim", torch.tensor(dim, dtype=torch.int64, device=device)
            )
            self.net = self.get_net()(self.dim, **kwargs).to(device).to(dtype)

    def __str__(self):
        txt = (
            f"BackpropableHook("
            f"module_name={self.module_name}, "
            f"strength={self.strength:0.2f}, "
            f")"
        )
        return txt

    @abc.abstractmethod
    def get_net(self) -> type[torch.nn.Module]:
        pass

    def load_state_dict(
        self,
        state_dict: t.Mapping[str, t.Any],
        strict: bool = True,
        assign: bool = False,
    ):
        dim = state_dict["dim"]
        self.net = self.get_net()(dim=dim)
        super().load_state_dict(state_dict, strict, assign)
        self.net = self.net.to(self.device)
        super()._post_load()

    def forward(self, module, input, output) -> t.Any:
        z_unit = output

        # Transport the outputs
        z_ot = self.net(z_unit).to(z_unit.dtype)

        # Apply transport with specific strength
        z_ot = self.strength * z_ot + (1 - self.strength) * z_unit

        return z_ot


class BPLinearHook(BPHook):
    def get_net(self) -> torch.nn.Module:
        return LinearProj


class BPNonLinearHook(BPHook):
    def get_net(self) -> torch.nn.Module:
        return NonLinearProjSigSum
