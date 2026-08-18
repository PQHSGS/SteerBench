# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import abc
import typing as t
from pathlib import Path

import torch


class InterventionHook(torch.nn.Module):
    """
    Abstract base class for a hook that intervenes during the forward pass of a PyTorch module.

    This class allows you to specify which part of the output tensor (or all outputs) to modify and at what point in the computation this modification should occur.

    Args:
        module_name (str): The name of the module or layer where the intervention is needed. If a specific output index is required, it can be specified after a colon (e.g., "module_name:output_index").
        intervention_position (str): Specifies when to intervene in the forward pass. 'all' means at every step, 'last' means only on the last element of the output tensor sequence.
        dtype (torch.dtype): The desired data type for the intervention. Default is torch.float32.
    """

    def __init__(
        self,
        module_name: str,
        device: str,
        intervention_position: t.Literal["all", "last"],
        dtype: torch.dtype = torch.float32,
        use_inputs: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.module_name = module_name
        self.device = device
        if ":" in module_name:
            self.select_tensor = int(module_name.split(":")[1])
        else:
            self.select_tensor = None
        self.intervention_position = intervention_position
        self.dtype = dtype
        self.use_inputs = use_inputs

    def register_named_buffers(self, **kwargs) -> None:
        for k, v in kwargs.items():
            self.register_buffer(k, v.to(self.dtype))

    def save_state_dict(self, state_path: Path) -> None:
        torch.save(self.state_dict(), state_path)

    def from_state_path(self, state_path: Path) -> None:
        """
        Loads intervention state from a state path pointing to a torch-saved state_dict.

        :param state_path: The state path to load.
        """
        self.load_state_dict(torch.load(state_path, weights_only=False))

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError("Method fit() must be implemented.")

    def _post_load(self) -> None:
        """
        This method should be called after loading the states of the hook.
        So calls must be placed at the end of .fit() and at the end of .load_state_dict().

        Re-implement as needed, but do not forget to call super()._post_load() in the implementation.
        """
        # Check all buffers are duly initialized.
        for buffer_name, buffer in self.named_buffers():
            assert buffer.numel() > 0, f"Buffer {buffer_name} has not been initialized."

    def update(self, *args, **kwargs):
        """
        Updates the state or arguments of this hook with new input data at runtime.

        This method can be overridden by subclasses to provide custom updating logic. By default, it does nothing and returns None.

        Parameters:
            *args : variable-length argument list
                Variable length argument list that will be used as is for the update operation.

            **kwargs : keyworded arguments
                Keyworded arguments that can also be used to update state or arguments of this hook.

        Returns:
            None
        """
        return None

    def __call__(self, module, input_, output):
        """
        PyTorch call method overridden to implement the intervention logic.

        Args:
            module (torch.nn.Module): The module for which the forward pass is being evaluated.
            input_ (tuple): Input tensors to the module.
            output (tuple or torch.Tensor): Output of the module's forward function. If `select_output` is specified, it will be a tuple containing this single element; otherwise, it's expected to be a tuple of outputs.

        Returns:
            The modified output after intervention. If `select_output` is specified, returns a modified version of the corresponding output in the tuple. Otherwise, returns the entire sequence of modified outputs.
        """
        x = input_ if self.use_inputs else output

        # Select the appropriate part of the output if select_output is specified
        if isinstance(x, tuple) and self.select_tensor is not None:
            selected_x = x[self.select_tensor]
        elif isinstance(x, tuple):
            selected_x = x[0]
        else:
            selected_x = x
        original_ndim = selected_x.ndim

        # Handle 2D xs by adding an extra dimension
        if original_ndim == 2:
            selected_x = selected_x[:, None, :]

        # Determine the part of the x where intervention will be applied
        if self.intervention_position == "last":
            assert original_ndim < 4, "Last is incompatible with 2D tensors"
            intervention_target = selected_x[:, -1, None, ...]
        elif self.intervention_position == "all":
            if original_ndim <= 3:
                intervention_target = selected_x
            elif original_ndim == 4:
                # The intervention will be applied to each pixel independently
                # (batch_size, num_features, h, w) -> (batch_size, h*w, num_features)
                intervention_target = (
                    selected_x.permute(0, 2, 3, 1)
                    .contiguous()
                    .view(selected_x.shape[0], -1, selected_x.shape[1])
                )
            else:
                raise ValueError(
                    "Unsupported number of dimensions for 'all' intervention"
                )
        elif self.intervention_position == "avg":
            if original_ndim <= 3:
                intervention_target = selected_x.mean(1, keepdims=True)
            elif original_ndim == 4:
                intervention_target = selected_x.mean((2, 3))[:, None, :]
            else:
                raise ValueError(
                    "Unsupported number of dimensions for 'avg' intervention"
                )
        else:
            intervention_target = selected_x

        # Ensure the intervention target is on the correct device and dtype
        dtype = intervention_target.dtype
        device = intervention_target.device
        intervention_target = intervention_target.to(
            dtype=self.dtype, device=self.device
        )

        # Apply the intervention using the forward method
        modified_intervention_target = self.forward(module, input_, intervention_target)

        # Convert back to the original dtype and device
        modified_intervention_target = modified_intervention_target.to(
            dtype=dtype, device=device
        )

        # Update the selected_x based on the intervention position
        if self.intervention_position == "last":
            if original_ndim == 3:
                selected_x[:, -1, ...] = modified_intervention_target[:, 0, ...]
        elif self.intervention_position == "avg":
            if original_ndim <= 3:
                selected_x = (
                    selected_x
                    + modified_intervention_target
                    - intervention_target.to(device=device, dtype=dtype)
                )
            elif original_ndim == 4:
                selected_x = (
                    selected_x
                    + modified_intervention_target.view(
                        selected_x.shape[0], selected_x.shape[1], 1, 1
                    )
                    - intervention_target.to(device=device, dtype=dtype).view(
                        selected_x.shape[0], selected_x.shape[1], 1, 1
                    )
                )
        elif self.intervention_position == "all":
            if original_ndim <= 3:
                selected_x = modified_intervention_target
            elif original_ndim == 4:
                # The intervention will be applied to each pixel independently
                # (batch_size, num_features, h, w) -> (batch_size*h*w, num_features)
                selected_x = (
                    modified_intervention_target.view(
                        x.shape[0],
                        x.shape[2],
                        x.shape[3],
                        x.shape[1],
                    )
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
        else:
            selected_x = modified_intervention_target

        # Remove the extra dimension if the original x was 2D
        if original_ndim == 2:
            selected_x = selected_x[:, 0, :]

        # Reconstruct the x tuple if necessary
        if isinstance(x, tuple) and self.select_tensor is not None:
            x = list(x)
            x[self.select_tensor] = selected_x
            x = tuple(x)
        else:
            x = selected_x

        # We re-apply the module to the modified inputs if inputs are used!
        # NOTE: This will add compute cost.
        if self.use_inputs:
            if isinstance(x, tuple):
                x = x[0]
            x = module.forward(x)
        return x

    @abc.abstractmethod
    def forward(self, module, input_, output):
        """
        Abstract method to be implemented by subclasses. This method defines the logic for how the intervention should modify the output.

        Args:
            module (torch.nn.Module): The module for which the forward pass is being evaluated.
            input_ (tuple): Input tensors to the module.
            output (torch.Tensor): Output tensor of the module's forward function, modified according to the intervention logic.

        Returns:
            A modified version of the output tensor after applying the intervention logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")
