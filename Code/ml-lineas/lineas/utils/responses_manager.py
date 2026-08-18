# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import contextlib
import logging
import typing as t
from pathlib import Path

import torch
from omegaconf import DictConfig
from tqdm import tqdm

import lineas.hooks as hooks
from lineas.datasets import get_dataloader, get_dataset
from lineas.models.get_model import get_model
from lineas.models.model_with_hooks import ModelWithHooks, get_model_with_hooks
from lineas.utils import gradient_mode

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ResponsesManager:
    """Computes and saves responses from specified modules within a pretrained language model.

    This class handles loading a pretrained model, setting up custom hooks to extract responses from desired modules,
    and iterating over a dataset to compute and save the extracted responses.

    Attributes:
        cfg (DictConfig): Configuration object containing model parameters, task settings, data paths, and intervention details.
        output_path (Path): Path where computed responses will be saved.
        model (nn.Module): Loaded pretrained language model with registered hooks for response extraction.
        module_names (List[str]): List of module names from which to extract responses.

    Args:
        cfg (DictConfig): Configuration object defining the model, task, and intervention parameters.

    Example usage:

    ```python
    # Assuming cfg is a loaded DictConfig object
    responses_manager = ResponsesManager(cfg)
    output_path = responses_manager.compute_responses()
    print(f"Responses saved to: {output_path}")
    ```

    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.output_path = self.get_output_path(cfg)

        assert cfg.device != "cuda" or torch.cuda.is_available()
        logger.info("Loading model to extract responses.")
        # Models and Tokenizers
        module, self.tokenizer = get_model(
            cache_dir=cfg.cache_dir,
            device=cfg.device,
            rand_weights=False,
            model_task=cfg.task_params.model_task,
            **cfg.model_params,
        )
        logger.info("Done loading model.")
        self.model = get_model_with_hooks(
            module=module,
            model_task=cfg.task_params.model_task,
            device=cfg.device,
            **cfg.model_params,
        )

        self.module_names = self.get_module_names(cfg.model_params.module_names)

        # Datasets
        self.set_dataloader()

    def get_model(self) -> torch.nn.Module:
        """
        Returns the pytorch model.
        """
        return self.model.model

    def get_model_with_hooks(self) -> ModelWithHooks:
        """
        Returns the ModelWithHooks object.
        """
        return self.model

    def set_dataloader(
        self,
        cfg: DictConfig | None = None,
        drop_last: bool = False,
        max_batches: int = None,
    ) -> None:
        """
        Sets the dataloader and also the config if passed.

        :param cfg: A config object
        """
        if cfg is not None:
            assert (
                cfg.model_params.model_path == self.cfg.model_params.model_path
            ), f"Found new model_path in config {cfg.model_params.model_path}, expected {self.cfg.model_params.model_path} already in the ResponseGenerator. Cannot set dataloader."
            self.cfg = cfg

        train_dataset, collate_fn = get_dataset(
            name=self.cfg.task_params.dataset,
            datasets_folder=Path(self.cfg.data_dir),
            split="train",
            subsets=list(
                self.cfg.task_params.src_subsets + self.cfg.task_params.dst_subsets
            ),
            tokenizer=self.tokenizer,
            **self.cfg.task_params.get("dataset_params", {}),
        )

        if max_batches is not None:
            train_dataset.data = train_dataset.data[: max_batches * self.cfg.batch_size]

        # Sampling and dataloader
        self.dataloader = get_dataloader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=collate_fn,
            drop_last=drop_last,
            shuffle=self.cfg.shuffle,
            balanced=self.cfg.balanced_data,
            seed=self.cfg.seed,
        )

    def get_module_names(self, module_names: list[str]) -> list[str]:
        """Get the names of modules in the model.

        Args:
            module_names (list of str): A regex list of module names.

        Returns:
            None
        """
        return self.model.find_module_names(self.model.module, module_names)

    def get_module_names_with_shapes(
        self, module_names: list[str]
    ) -> dict[str, torch.Size]:
        """Get the names of modules in the model with shapes associated.

        Args:
            module_names (list of str): A regex list of module names.

        Returns:
            None
        """
        return self.model.find_module_names_with_shapes(
            self.model.module, module_names, device=self.model.device
        )

    def _set_hooks(
        self,
        module_names: list[str],
        extra_pre_hooks: list = None,
        extra_post_hooks: list[t.Any] = None,
        keep_gradients: bool = False,
    ) -> None:
        """Registers hooks to extract responses from specified modules and save them.

        This method iterates through the list of desired module names (`module_names`) and registers a forward hook on each module.
        The hook function, defined using `hooks.get_hook`, captures the output of the module during the forward pass
        and saves it to disk along with other relevant metadata specified in `self.cfg.save_fields`.

        The `pooling_op` parameter determines how the outputs from different modules are aggregated.  It can be a single string
        representing a pooling operation (e.g., "mean", "max") or a list of strings for more complex aggregation schemes.

        Args:
            module_names (t.List[str]): A list of module names within the model from which to extract responses.
            extra_pre_hooks (list): Additional hooks to register during computation of responses, before responses are being stored. Defaults to an empty list.
            extra_post_hooks (list): Additional hooks to register during computation of responses, after responses are being stored. Defaults to an empty list.
            keep_gradients: (bool) Whether to keep gradients stored during forward pass. Defaults to False.

        Raises:
            KeyError: If a specified module name does not exist within the model.

        """
        if isinstance(self.cfg.intervention_params.pooling_op, str):
            pooling_op = [self.cfg.intervention_params.pooling_op]
        else:
            pooling_op = self.cfg.intervention_params.pooling_op
        hook_fns = [
            hooks.get_hook(
                "postprocess_and_save",
                module_name=module_name,
                pooling_op_names=pooling_op,
                output_path=self.output_path if not keep_gradients else None,
                save_fields=[
                    "id",
                ]
                + self.cfg.save_fields,
                threaded=False,
                return_outputs=keep_gradients,
                keep_gradients=keep_gradients,
                **self.cfg.intervention_params.hook_params,
            )
            for module_name in module_names
        ]
        if extra_pre_hooks is not None:
            hook_fns = extra_pre_hooks + hook_fns
        if extra_post_hooks is not None:
            hook_fns += extra_post_hooks
        self.model.remove_hooks()
        self.model.register_hooks(hook_fns)

    @staticmethod
    def get_output_path(cfg: DictConfig) -> Path:
        """Returns the output path for the computed responses."""
        # Sanity check, only allowing multiplicity for args.module_names and args.subset
        model_name = Path(cfg.model_params.model_path).name
        # Setting paths
        return Path(cfg.cache_dir) / cfg.tag / model_name / cfg.task_params.dataset

    def compute_responses(
        self,
        module_names: list[str] | None = None,
        extra_pre_hooks: list = None,
        extra_post_hooks: list[t.Any] = None,
        keep_gradients: bool = False,
    ) -> Path:
        """Computes and saves responses from specified modules in the model.

        This method orchestrates the process of extracting responses from designated modules within the model during inference. It handles hook registration, batch processing, and saving the extracted responses to disk.

        Args:
            module_names (List[str], optional): List of module names from which to extract responses. If None, defaults to the configured module names (`self.module_names`).
            extra_pre_hooks (list): Additional hooks to register during computation of responses, before responses are being stored. Defaults to an empty list.
            extra_post_hooks (list): Additional hooks to register during computation of responses, after responses are being stored. Defaults to an empty list.
            keep_gradients (bool, optional): Whether to keep gradients in the returned Tensors. Defaults to False.

        Returns:
            Path: The path where the computed responses are saved.

        """
        if not self.cfg.enabled:
            return self.output_path

        if module_names is None:
            module_names = self.module_names

        self._set_hooks(
            module_names,
            extra_pre_hooks=extra_pre_hooks,
            extra_post_hooks=extra_post_hooks,
            keep_gradients=keep_gradients,
        )
        logger.info(f"Computing responses for {len(module_names)} modules.")
        logger.debug("\n".join(module_names))

        current_batch = 0
        max_batches = (
            len(self.dataloader)
            if self.cfg.max_batches is None
            else min(len(self.dataloader), self.cfg.max_batches)
        )
        if current_batch == self.cfg.max_batches:
            logger.info("All batches found, nothing to compute")
        else:
            if current_batch > 0:
                logger.info(f"Resuming from batch {current_batch}")
            else:
                logger.info("Computing batch responses")

            iloader = iter(self.dataloader)
            for idx in tqdm(range(max_batches), desc="Computing responses"):
                batch = next(iloader)
                if idx >= current_batch:
                    with gradient_mode(keep_gradients):
                        self.model.update_hooks(batch_idx=idx, batch=batch)
                        with contextlib.suppress(
                            hooks.custom_exceptions.TargetModuleReached
                        ):
                            self.model(
                                batch,
                                **self.cfg.model_params.get("training_kwargs", {}),
                            )
                # checkpoint["current_batch"] = idx + 1
                # TODO Save hooks state if stateful
                # torch.save(checkpoint, checkpoint_path)
            logger.info("Done")
        return self.output_path
