# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import copy
import logging
import typing as t

import torch

from lineas.hooks.intervention_hook import InterventionHook
from lineas.utils.quantiles import compute_quantiles


class GaussianAcTHook(InterventionHook):
    def __init__(
        self,
        module_name: str,
        device: str = None,
        intervention_position: str = "all",
        dtype: torch.dtype = torch.float32,
        strength: float = 1.0,
        std_eps: float = 1e-4,
        quantiles_src: str = "q_all",
        hook_onlymean: bool = False,
        **kwargs,
    ):
        """
        Gaussian Optimal Transport hook. Assumes the output of each neuron is independent and Gaussianly distributed.
        Transports from N(mu1, std1) to N(mu2, std2) if hook_forward is True, and N(mu2, std2) to N(mu1, std1) otherwise.

        Args:
            module_name (str): Module name on which the hook is applied.
            device (torch.device): Torch device.
            dtype (str): The torch dtype for this hook.
            intervention_position (str): The position of the token to intervene upon (all, last)
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
        self.onlymean = bool(int(hook_onlymean))
        self.std_eps = float(std_eps)
        self.quantiles_src = str(quantiles_src)

        # Register buffer placeholders
        for buffer_name in ["mu1", "mu2", "std1", "std2"]:
            self.register_buffer(buffer_name, torch.empty(0))

        # Register non-buffer placeholders
        self.quantiles_dict_src = None

    def __str__(self):
        txt = (
            f"GaussianOT("
            f"module_name={self.module_name}, "
            f"quantiles_src={self.quantiles_src}, "
            f"strength={self.strength:0.2f}, "
            f"onlymean={self.hook_onlymean}"
            f")"
        )
        return txt

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
    ) -> None:
        # Fill in registered buffers
        for buffer_name, _ in self.named_buffers():
            setattr(
                self,
                buffer_name,
                state_dict[buffer_name].to(self.device).to(self.dtype),
            )

        # Fill in non buffers
        self.quantiles_dict_src = {
            k: [v[0].to(self.device), v[1].to(self.device)]
            for k, v in state_dict["quantiles_dict_src"].items()
        }

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

    def fit(self, responses: torch.Tensor, labels=torch.Tensor, **kwargs) -> None:
        # Typecasting to float64 to avoid overflow in the computation of the mean/std.
        z_f64 = responses.to(torch.float64)
        labels = labels.to(torch.bool)
        self.mu1 = torch.mean(z_f64[labels], dim=0).to(self.dtype)
        self.std1 = torch.std(z_f64[labels], dim=0).to(self.dtype)
        self.mu2 = torch.mean(z_f64[~labels], dim=0).to(self.dtype)
        self.std2 = torch.std(z_f64[~labels], dim=0).to(self.dtype)
        self.quantiles_dict_src = compute_quantiles(z_f64[labels])
        self._post_load()

    def _post_load(self) -> None:
        """
        This method should be called after loading the states of the hook.
        So calls must be placed at the end of .fit() and at the end of .load_state_dict().
        """
        super()._post_load()

        self.mask = (
            torch.ones_like(
                self.mu1, dtype=torch.bool
            )  # just here to be able to comment/uncomment masks below easily]
            # TODO: We've had this all the time, remove?
            & (self.std1 > self.std_eps)
            & (self.std2 > self.std_eps)
        )

        # Pre-computing things beforehand
        self.mu1_m = self.mu1[self.mask]
        self.mu2_m = self.mu2[self.mask]
        self.std1_m = self.std1[self.mask]
        self.std2_m = self.std2[self.mask]
        self.std1_2_m = self.std2_m / self.std1_m

        # TODO: Move this to some post_load method to avoid doing it every time.
        if self.quantiles_src == "q_all":
            self.quantiles_src = [-1e6, 1e6]
        else:
            self.quantiles_src = copy.deepcopy(
                self.quantiles_dict_src[self.quantiles_src]
            )
            self.quantiles_src[0] = self.quantiles_src[0][self.mask].view(1, -1)
            self.quantiles_src[1] = self.quantiles_src[1][self.mask].view(1, -1)

    def forward(self, module, input, output) -> t.Any:
        output_shape = output.shape
        if len(output_shape) == 3:
            output = output.view(-1, output_shape[2])
            self.mu1_m = self.mu1_m.view(1, -1)
            self.mu2_m = self.mu2_m.view(1, -1)
            self.std1_m = self.std1_m.view(1, -1)
            self.std2_m = self.std2_m.view(1, -1)
            self.std1_2_m = self.std1_2_m.view(1, -1)
            z_unit = output[:, self.mask]  # B,  U
            z_mask = z_unit
        elif len(output_shape) == 4:
            raise DeprecationWarning(
                "InterventionHook already takes care of fomatting the input to dims"
            )
            # Handles image model case
            self.mu1_m = self.mu1_m.view(1, -1, 1, 1)
            self.mu2_m = self.mu2_m.view(1, -1, 1, 1)
            self.std1_m = self.std1_m.view(1, -1, 1, 1)
            self.std2_m = self.std2_m.view(1, -1, 1, 1)
            self.std1_2_m = self.std1_2_m.view(1, -1, 1, 1)
            z_unit = output[:, self.mask]  # B,  U
            z_mask = z_unit.mean(dim=(2, 3), keepdim=False)  # B, C
        else:
            raise NotImplementedError("Can't handle tensors with dim=2.")

        # Select outputs that fall in statistics of pooled
        pool_mask = (
            (self.quantiles_src[0] < z_mask) & (z_mask < self.quantiles_src[1])
        ).to(z_unit.dtype)

        # Transport the outputs
        if self.onlymean:
            z_ot = (z_unit - self.mu1_m) + self.mu2_m
        else:
            z_ot = self.std1_2_m * (z_unit - self.mu1_m) + self.mu2_m

        # Apply transport with specific strength
        z_ot = self.strength * z_ot + (1 - self.strength) * z_unit

        # Only apply masked outputs
        output[:, self.mask] = pool_mask * z_ot + (1 - pool_mask) * z_unit
        output = output.view(*output_shape)
        return output


class OnlyMeanHook(GaussianAcTHook):
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
        # Hardcoded to only mean.
        kwargs.update({"hook_onlymean": True})

        super().__init__(
            module_name=module_name,
            device=device,
            intervention_position=intervention_position,
            dtype=dtype,
            strength=strength,
            std_eps=std_eps,
            quantiles_src=quantiles_src,
            **kwargs,
        )

    def __str__(self):
        txt = (
            f"OnlyMeanOT("
            f"module_name={self.module_name}, "
            f"quantiles_src={self.quantiles_src}, "
            f"strength={self.strength:0.2f}, "
            f")"
        )
        return txt
