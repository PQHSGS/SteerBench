#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#


import contextlib
import logging
import os
import re
from pathlib import Path
from threading import Thread

import diffusers
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.hooks import RemovableHandle
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

from lineas.hooks import InterventionHook, get_hook
from lineas.hooks.responses_hook import ResponsesHook
from lineas.models import get_model
from lineas.utils.utils import is_module_name_in_regex

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TASK_MAPPING = {
    "text_generation": AutoModelForCausalLM,
    "sequence-classification": AutoModelForSequenceClassification,
}


def load_hooked_model(
    model_path: str,
    intervention_name: str,
    intervention_state_path: Path,
    cache_dir: str = None,
    module_names: list[str] = [".*"],
    dtype=None,
    device=None,
    seq_len: int | None = None,
    rand_weights: bool = False,
    model_task: str | None = "text_generation",
) -> dict:
    """
    Loads a model with hooks attached to it according to specified intervention name

    Args:
        model_path (str): The path to the directory containing the model files. This is required.
        intervention_name (str, optional): The name of the intervention. Defaults to "aura".
        cache_dir (str, optional): Path for storing/loading cached tensors. If None, no caching will be used. Defaults to None.
        module_names (list of str, optional): List of module names to load. Use ["*"] to load all modules. Defaults to ["*"].
        dtype (str, optional): The data type for the weights. If None, will be inferred from the model if possible. Defaults to None.
        device (str, optional): The device on which the tensors will be loaded/stored. Defaults to None.
        seq_len (int, optional): The sequence length for text-based models. If None, no effect. Defaults to None.
        rand_weights (bool, optional): Whether or not to initialize weights with random values. Defaults to False.
        model_task (str, optional): The task the model is designed for. Can be "text_generation", etc. Defaults to "text_generation".


    Returns:
        dict (model, tokenizer): a dictionary with keys model and tokenizer containing model with hooks attached and tokenizer
    """
    model, tokenizer = get_model(
        model_path=model_path,
        cache_dir=cache_dir,
        dtype=dtype,
        device=device,
        model_task=model_task,
        seq_len=seq_len,
        rand_weights=rand_weights,
    )
    logger.info(f"Loaded model {model_path} from {cache_dir}. Got the following:")
    logger.info(f"{model}")
    # Create hooks
    module_names_hooks = ModelWithHooks.find_module_names(model, module_names)
    logger.info(f"MODULE_NAMES: {module_names}")
    hooks = []
    logger.info("Loading hooks")
    for module_name in module_names_hooks:
        state_path = (
            Path(cache_dir) / intervention_state_path / f"{module_name}.statedict"
        )
        logger.info(f"{state_path}")
        assert os.path.exists(state_path), logger.error(
            f"Error: {state_path} does not exist, but is needed to proceed."
        )

        hook = get_hook(
            intervention_name,
            module_name=module_name,
            device=device,
        )
        hook.from_state_path(state_path)
        hooks.append(hook)

    # Create hooked model
    logger.info(f"{hooks}")
    model_with_hooks = ModelWithHooks(
        module=model,
        hooks=hooks,
    )
    logger.info("Attaching hooks")
    model_with_hooks.register_hooks()

    return {"model": model_with_hooks, "tokenizer": tokenizer}


HookType = InterventionHook | ResponsesHook
HookList = list[HookType] | None


class ModelWithHooks:
    """
    Class wrapping a Pytorch model so that we can apply forward hooks on its responses.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        hooks: HookList = None,
        device: str = None,
        **kwargs,
    ) -> None:
        """
        Initializes the ModelWithHooks instance with a Pytorch module and optional hooks.

        Args:
            module (nn.Module): The PyTorch module to be wrapped.
            hooks (Optional[List[ResponsesHook or InterventionHook]]): List of hooks to apply on the responses. Default is None.
            device (str): The device where the model should run (e.g., "cpu", "cuda:0"). Default is None.

        Returns:
            None
        """
        self.hooks: HookList = hooks if hooks is not None else []

        self.module = module
        self._forward_hook_handles: list[RemovableHandle] = []
        self._postprocessing_threads: list[Thread] = []
        self.device = device

    @staticmethod
    def find_module_names(
        model: torch.nn.Module,
        regex_list: list[str],
        skip_first: int = 0,  # first layer tends to be the model root
    ) -> list[str]:
        """
        Finds module names of a PyTorch model that match any of the given regular expressions.

        Args:
            model (nn.Module): The PyTorch model.
            regex_list (List[str]): List of regular expressions to match module names against.
            skip_first (int): Number of modules to skip at the beginning. Default is 0, meaning no modules are skipped.

        Returns:
            List[str]: Sorted list of matching module names.
        """
        module_names = [name[0] for name in model.named_modules()][skip_first:]
        _module_names = []
        for module_name in module_names:
            if module_name == "":
                continue
            matches = is_module_name_in_regex(module_name, regex_list)
            for match in matches:
                if match not in _module_names:
                    _module_names.append(match)
        return _module_names

    @classmethod
    def find_module_names_with_shapes(
        cls,
        model: torch.nn.Module,
        regex_list: list[str],
        skip_first: int = 0,  # first layer tends to be the model root
        device: str = "cuda",
        use_inputs: int = False,
    ) -> dict[str, torch.Size]:
        """
        Finds module names of a PyTorch model that match any of the given regular expressions.

        Args:
            model (nn.Module): The PyTorch model.
            regex_list (List[str]): List of regular expressions to match module names against.
            skip_first (int): Number of modules to skip at the beginning. Default is 0, meaning no modules are skipped.

        Returns:
            List[str]: Sorted list of matching module names.
        """

        class ShapeHook:
            def __init__(self, module_name, use_inputs=False):
                self.shape = None
                self.use_inputs = use_inputs
                self.module_name = module_name
                if ":" in module_name:
                    self.select_input = int(module_name.split(":")[1])
                else:
                    self.select_input = 0

            def get_shape(self, data):
                if isinstance(data, tuple):
                    data = data[self.select_input]
                return data.shape

            def __call__(self, module, input, output):
                if self.use_inputs:
                    self.shape = self.get_shape(input)
                else:
                    self.shape = self.get_shape(output)

        # Grab sorted modules
        _module_names = cls.find_module_names(model, regex_list, skip_first)

        hook_handles = []
        shape_hooks = {}
        names_modules_map = {}
        for module_name, module in model.named_modules():
            names_modules_map[module_name] = module
        for _module_name in _module_names:
            module_name = _module_name.split(":")[0]
            if module_name in names_modules_map:
                module = names_modules_map[module_name]
                # Attach a hook to capture the output shape
                shape_hooks[_module_name] = ShapeHook(
                    module_name=_module_name, use_inputs=use_inputs
                )
                hook = module.register_forward_hook(shape_hooks[_module_name])
                hook_handles.append(hook)

        # Forward pass to trigger hooks
        fake_inputs = {
            "input_ids": torch.randint(
                10, 100, size=(1, 3), requires_grad=False, device=device
            ),
            # "attention_mask": torch.ones((1, 3), requires_grad=False),
        }
        model.eval()
        with torch.no_grad():
            model(**fake_inputs)

        # Finally remove all hooks
        for handle in hook_handles:
            handle.remove()

        return {m: shape_hooks[m].shape for m in _module_names}

    def find_module_state_paths(
        self, path: Path, regex_list: list[str], strict: bool = False
    ) -> list[Path]:
        """
        Find matching state dict files in a directory based on given regular expressions and verify consistency with model module names.

        Args:
        - path (Path): The directory path to search for state dict files.
        - regex_list (List[str]): A list of regular expressions used to match state dict file names.
        - strict (bool, optional): If True, raise an error if any state dict does not match a module name. Default is False.

        Returns:
        - List[Path]: A list of matching state dict file paths.

        Raises:
        - RuntimeError: If strict is True and any state dict does not match a module name.
        """
        all_state_paths = list(Path(path).glob("*.statedict"))
        _regex_list = [f"{pattern}(:[0-9]+)?" for pattern in regex_list]
        ret = []
        for state_path in all_state_paths:
            for pattern in _regex_list:
                if re.fullmatch(pattern, state_path.stem):
                    ret.append(Path(state_path))
                    break
        if strict:
            module_names = self.find_module_names(self.module, regex_list)
            for state_path in ret:
                found = False
                for module_name in module_names:
                    # Strict does not work when :1 statedicts exist, so we are removing it for the moment
                    if str(module_name).split(":")[0] in str(state_path):
                        found = True
                        break
                if not found:
                    raise RuntimeError("State dicts do not match all module names")
        logger.info(f"Found {len(ret)} state dicts in {path}")
        return ret

    @staticmethod
    def sort_module_names(
        model: torch.nn.Module,
        module_names: list[str],
    ) -> list[str]:
        """
        Sorts a list of module names based on their order in the PyTorch model graph.

        Args:
            model (nn.Module): The PyTorch model.
            module_names (List[str]): List of module names to sort.

        Returns:
            List[str]: Sorted list of module names.
        """
        sorted_module_names = [name[0] for name in model.named_modules()]
        indices = [sorted_module_names.index(v) for v in module_names]
        sorted_indices = np.sort(indices)
        output = [sorted_module_names[i] for i in sorted_indices]
        return output

    @staticmethod
    def find_largest_modulelist(
        model: torch.nn.Module,
    ) -> tuple[torch.nn.ModuleList | None, int]:
        """
        Recursively search through a PyTorch model to find the largest nn.ModuleList.

        Args:
            model: The PyTorch model to search through

        Returns:
            Tuple containing:
            - The largest ModuleList found (or None if none exists)
            - The size (number of modules) in the largest ModuleList
        """
        largest_modulelist = None
        largest_size = 0

        # Check direct children first
        for _name, module in model.named_children():
            # If we found a ModuleList, check its size
            if isinstance(module, torch.nn.ModuleList):
                current_size = len(module)
                if current_size > largest_size:
                    largest_size = current_size
                    largest_modulelist = module

            # Recursively check any submodules that might contain ModuleLists
            if len(list(module.children())) > 0:
                sub_modulelist, sub_size = ModelWithHooks.find_largest_modulelist(
                    module
                )
                if sub_size > largest_size:
                    largest_size = sub_size
                    largest_modulelist = sub_modulelist

        return largest_modulelist, largest_size

    def guess_num_blocks(self) -> int | None:
        # Find the largest layer container dynamically
        _, max_size = self.find_largest_modulelist(self.module)
        return max_size

    def get_module(self) -> torch.nn.Module:
        """
        Gets the wrapped PyTorch module.

        Returns:
            nn.Module: The wrapped PyTorch module.
        """
        return self.module

    def register_hooks(
        self,
        hooks=None,
    ):
        """
        Registers forward hooks on the modules of the network with the given names.

        Args:
            hooks (Optional[List[ResponsesHook]]): List of hooks to register. Default is None.

        Returns:
            None
        """
        # register forward hook for all modules in the network with the exception of the root
        # module and container modules.
        if hooks is not None:
            assert len(self.hooks) == 0, "Hooks already registered"
            self.hooks: HookList = hooks if hooks is not None else []

        logger.info(f"Registering {len(self.hooks)}.")
        self._forward_hook_handles = []
        module_dict = {name: module for name, module in self.module.named_modules()}
        for hook in self.hooks:
            base_module_name = hook.module_name.split(":")[0]
            if base_module_name in module_dict:
                module = module_dict[base_module_name]
                self._forward_hook_handles.append(module.register_forward_hook(hook))

    def get_hooks(self) -> HookList:
        return self.hooks

    def get_hook_parameters(self) -> list[torch.Tensor]:
        params = []
        for hook in self.hooks:
            if hasattr(hook, "parameters"):
                params += list(hook.parameters())
        return params

    def set_hooks(self, hooks: HookList) -> None:
        self.remove_hooks()
        self.hooks = hooks

    def remove_hooks(self):
        """
        Removes all registered hooks and joins any remaining threads.

        Returns:
            None
        """
        logger.info(f"Removing {len(self.hooks)} hooks.")
        for h in self._forward_hook_handles:
            h.remove()
        for hook in self.hooks:
            with contextlib.suppress(Exception):
                hook.join()
        self._forward_hook_handles = []
        self.hooks = []

    def load_hooks_from_folder(
        self, folder: Path, module_names: list[str], hook_type: str, **hook_params
    ) -> HookList:
        assert len(self.hooks) == 0, "Trying to overwrite existing hooks!"
        if hook_type == "none":
            return []
        # Create hooks
        state_paths = self.find_module_state_paths(
            path=Path(folder),
            regex_list=module_names,
            strict=True,
        )
        self.hooks = []
        for state_path in state_paths:
            saved_module_name = str(Path(state_path).stem)
            hook = get_hook(
                hook_type,
                module_name=saved_module_name,
                **hook_params,
            )
            hook.from_state_path(state_path)
            self.hooks.append(hook)
        logger.info(
            f"Loaded {len(state_paths)}/{len(list(Path(folder).glob('*.statedict')))} hooks from {folder}"
        )
        assert len(self.hooks) > 0, f"No hooks loaded from {folder}!"

        return self.hooks

    def get_hook_outputs(self) -> dict[str, torch.Tensor]:
        """
        Gets the outputs of all registered hooks.

        Returns:
            Dict: Dictionary mapping module names to their corresponding outputs.
        """
        outputs = {}
        for hook in self.hooks:
            if hasattr(hook, "outputs"):
                assert all(
                    [k not in outputs for k in hook.outputs.keys()]
                ), "Trying to grab outputs from hook for an already existing module_name."
                outputs.update(hook.outputs)
        return outputs

    def update_hooks(self, *args, **kwargs):
        assert len(self.hooks) > 0
        for hook in self.hooks:
            with contextlib.suppress(Exception):
                hook.join()
            hook.update(*args, **kwargs)

    def get_tokenizer(self):
        """
        Gets the tokenizer module associated with this model.

        Returns:
            Optional[nn.Module]: The tokenizer module, or None if no tokenizer is available.
        """
        return None

    def get_target_module_names(self) -> list[str]:
        """
        Returns a list of names of the modules to which hooks are registered.

        Returns:
            List[str]: A list of module names that have associated hooks.
        """
        return [hook.module_name for hook in self.hooks]

    def print_module_names_responses_sizes(self) -> None:
        """
        Prints the names and sizes of the responses for each registered hook.

        This method is not yet implemented and will raise a NotImplementedError if called.

        Raises:
            NotImplementedError: Indicates that this method has not been implemented yet.
        """
        raise NotImplementedError

    def __call__(self, batch):
        """
        Forward pass for huggingface models. This method is called when an instance of the ModelWithHooks class is invoked as a function.

        Args:
            batch (Dict[str, torch.Tensor]): A dictionary containing input tensors for the model. It should include 'input_ids' and 'attention_mask' keys with corresponding tensor values.

        Returns:
            Dict: The output of the model after processing the inputs. This is typically a dictionary or tuple depending on the model architecture, but here it is assumed to return a dictionary containing the model outputs.
        """
        input_ids, attention_mask = (
            batch["input_ids"],
            batch["attention_mask"],
        )
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        return self.module(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def named_parameters(self):
        for hook in self.hooks:
            yield from hook.named_parameters()

    def parameters(self, requires_grad=True):
        p = []
        for hook in self.hooks:
            p += [
                p
                for p in hook.parameters()
                if (not requires_grad) or (p.requires_grad == True)
            ]
        yield from iter(p)

    def apply_(self, f) -> None:
        for hook in self.hooks:
            f(hook)

    def map(self, f) -> InterventionHook:
        ret = []
        for hook in self.hooks:
            ret.append(f(hook))
        return ret


class StableDiffusionWithHooks(ModelWithHooks):
    """
    A class that extends ModelWithHooks for stable diffusion models.

    This class provides functionality to register hooks on the VAE (Variational Autoencoder)
    and UNet modules of a stable diffusion model. It also overrides the `__call__` method to
    perform inference using the stable diffusion model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_level = "text_encoder"

    @staticmethod
    def find_module_names(
        model: torch.nn.Module,
        regex_list: list[str],
        skip_first: int = 0,  # first layer tends to be the model root
    ) -> list[str]:
        """
        Finds module names that match any of the given regular expressions in the
        VAE and UNet modules of a PyTorch module.

        Args:
            model (torch.nn.Module): The PyTorch module to search for module names.
            regex_list (List[str]): A list of regular expressions to match against the
                module names.
            skip_first (int, optional): Number of modules to skip from the start. Defaults to 0.

        Returns:
            List[str]: A list of module names that matched any of the given regular expressions.
        """

        def sort_unet(module_names):
            start = []
            down = []
            mid = []
            up = []
            end = []
            for name in module_names:
                if "down_block" in name:
                    down.append(name)
                elif "mid_block" in name:
                    mid.append(name)
                elif "up_block" in name:
                    up.append(name)
                elif len(down) == 0 and len(mid) == 0 and len(up) == 0:
                    start.append(name)
                else:
                    end.append(name)
            sorted_module_names = start + down + mid + up + end
            return sorted_module_names

        module_names = [
            f"text_encoder.{name[0]}" for name in model.text_encoder.named_modules()
        ][skip_first:]
        if hasattr(model, "text_encoder_2") and model.text_encoder_2 is not None:
            module_names += [
                f"text_encoder_2.{name[0]}"
                for name in model.text_encoder_2.named_modules()
            ][skip_first:]
        if hasattr(model, "unet"):
            module_names += sort_unet(
                [f"unet.{name[0]}" for name in model.unet.named_modules()][skip_first:]
            )
        elif hasattr(model, "transformer"):
            module_names += [
                f"transformer.{name[0]}" for name in model.transformer.named_modules()
            ][skip_first:]
        module_names += [f"vae.{name[0]}" for name in model.vae.named_modules()][
            skip_first:
        ]

        _module_names = []
        for module_name in module_names:
            match = is_module_name_in_regex(module_name, regex_list)
            if match is not None and match not in _module_names:
                _module_names.extend(match)
        return _module_names

    @classmethod
    def find_module_names_with_shapes(
        cls,
        model: torch.nn.Module,
        regex_list: list[str],
        skip_first: int = 0,  # first layer tends to be the model root
        device: torch.device = "cuda",
    ) -> dict[str, torch.Size]:
        """
        Finds module names of a PyTorch model that match any of the given regular expressions.

        Args:
            model (nn.Module): The PyTorch model.
            regex_list (List[str]): List of regular expressions to match module names against.
            skip_first (int): Number of modules to skip at the beginning. Default is 0, meaning no modules are skipped.

        Returns:
            List[str]: Sorted list of matching module names.
        """

        class ShapeHook:
            def __init__(self):
                self.shape = None

            def __call__(self, module, input, output):
                self.shape = output.shape

        # Grab sorted modules
        _module_names = cls.find_module_names(model, regex_list, skip_first)

        hook_handles = []
        shape_hooks = {}
        for name in _module_names:
            if ":" in name:
                name = name.split(":")[0]

            # Attach a hook to capture the output shape
            component, *submodule_name = name.split(".")
            submodule_name = ".".join(submodule_name)
            if component == "text_encoder_2":
                module = model.text_encoder_2.get_submodule(submodule_name)
            elif component == "unet":
                module = model.unet.get_submodule(submodule_name)
            elif component == "transformer":
                module = model.transformer.get_submodule(submodule_name)
            elif component == "vae":
                module = model.vae.get_submodule(submodule_name)

            shape_hooks[name] = ShapeHook()
            hook = module.register_forward_hook(shape_hooks[name])
            hook_handles.append(hook)

        # Forward pass to trigger hooks
        fake_inputs = {"prompt": " ", "num_inference_steps": 1}
        with torch.no_grad():
            model(**fake_inputs)

        # Finally remove all hooks
        for handle in hook_handles:
            handle.remove()

        return {m: shape_hooks[m].shape for m in _module_names}

    def register_hooks(
        self,
        hooks=None,
    ):
        """
        Registers forward hooks on all modules in the VAE and UNet networks of the model,
        except for the root module and container modules.

        Args:
            hooks (List[ResponsesHook], optional): A list of ResponsesHook objects to register as
                hooks on the model modules. Defaults to None.
        """
        # register forward hook for all modules in the network with the exception of the root
        # module and container modules.
        if hooks is not None:
            assert len(self.hooks) == 0, "Hooks already registered"
            self.hooks: HookList = hooks if hooks is not None else []

        logger.info(f"StableDiffusionWithHooks: Registering {len(self.hooks)} hooks.")
        self._forward_hook_handles = []

        def add_submodule_handles(submodule_name: str):
            submodule = getattr(self.module, submodule_name)
            module_dict = {
                f"{submodule_name}.{k}": v for k, v in submodule.named_modules()
            }
            for hook in self.hooks:
                base_module_name = hook.module_name.split(":")[0]
                if base_module_name in module_dict:
                    module = module_dict[base_module_name]
                    logger.info(f"REGISTERING {type(hook)} {base_module_name}")
                    self._forward_hook_handles.append(
                        module.register_forward_hook(hook)
                    )

        add_submodule_handles("text_encoder")
        if (
            hasattr(self.module, "text_encoder_2")
            and self.module.text_encoder_2 is not None
        ):
            add_submodule_handles("text_encoder_2")
        if hasattr(self.module, "unet"):
            add_submodule_handles("unet")
            self.forward_level = "unet"
        elif hasattr(self.module, "transformer"):
            add_submodule_handles("transformer")
            self.forward_level = "transformer"
        add_submodule_handles("vae")

    def __call__(self, batch, **generation_kwargs):
        """
        Performs inference using the stable diffusion model with a given batch of data.

        Args:
            batch (dict): A dictionary containing the input data for the model.
                It should have keys 'prompt' and other parameters as required by the stable
                diffusion model.
        """
        if self.forward_level == "text_encoder":
            self.module._interrupt = True
        diffusers.utils.logging.disable_progress_bar()
        self.module(
            batch["prompt"],
            output_type="latent",
            **generation_kwargs,
        )
        diffusers.utils.logging.enable_progress_bar()


TASK_REGISTRY = {
    "text_generation": ModelWithHooks,
    "text-classification": ModelWithHooks,
    "question-answering": ModelWithHooks,
    "text_to_image_generation": StableDiffusionWithHooks,
}


def get_model_with_hooks(
    module: torch.nn.Module,
    hooks: list[ResponsesHook] | None = None,
    model_task: str = None,
    device: str = None,
    **model_params: DictConfig,
) -> ModelWithHooks:
    """
    Returns a hookable model for the given task.

    This function takes in a PyTorch module and returns a model with the
    specified hooks for a specific task. If no task is provided, it defaults to None.

    Args:
        module (torch.nn.Module): The PyTorch module that needs to be processed.
        hooks (t.Optional[t.List[ResponsesHook]], optional): A list of hooks to apply on the model. Defaults to None.
        model_task (str, optional): The specific task for which the model is being created. Defaults to None.
        device (str, optional): The device where the computation will take place. Defaults to None.

    Returns:
        ModelWithHooks: A model with specified hooks for the given task.
    """
    return TASK_REGISTRY[model_task](
        module=module,
        hooks=hooks,
        device=device,
        **model_params,
    )
