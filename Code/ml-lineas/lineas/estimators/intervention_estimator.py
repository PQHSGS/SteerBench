# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import functools
import logging
import sys
import time
import typing as t
from collections import defaultdict
from pathlib import Path

import matplotlib
import pandas as pd
import torch
import wandb
from omegaconf import DictConfig
from torch.optim.lr_scheduler import CosineAnnealingLR

import lineas.hooks as hooks
from lineas.datasets.responses_io import ResponsesLoader
from lineas.estimators.base_estimator import BaseEstimator
from lineas.hooks.backprop_transport import BPHook
from lineas.utils import copy_hydra_config, gradient_mode, proximal
from lineas.utils.losses import get_loss, get_optimizer
from lineas.utils.responses_manager import ResponsesManager

matplotlib.use("agg")
import pylab

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LayerwiseEstimator(BaseEstimator):
    def _learn_intervention(self, module_name: str):
        """Learn an intervention strategy for a specific module.

        This method loads relevant model responses, applies the intervention to those responses,
        and learns the parameters of the intervention strategy based on the modified data.

        Args:
            module_name (str): The name of the module within the language model for which the intervention will be learned.

        Returns:
            None
        """
        # Load responses for a given module
        data_subset = self.responses_loader.load_data_subset(
            {
                "module_names": [module_name],
                "pooling_op": [self.interventions_cfg.intervention_params.pooling_op],
                "subset": list(
                    self.interventions_cfg.task_params.src_subsets
                    + self.interventions_cfg.task_params.dst_subsets
                ),
            },
            num_workers=0,
        )
        data_subset = ResponsesLoader.label_src_dst_subsets(
            data_subset,
            src_subset=self.interventions_cfg.task_params.src_subsets,
            dst_subset=self.interventions_cfg.task_params.dst_subsets,
            balanced=True,
            seed=self.interventions_cfg.seed,
        )
        z = torch.tensor(data_subset["responses"])
        y = torch.tensor(data_subset["label"]).to(torch.bool)

        # Create hook object with requested hook args
        hook = hooks.get_hook(
            self.interventions_cfg.intervention_params.name,
            module_name=module_name,
            **self.interventions_cfg.intervention_params.hook_params,
        )

        # Estimate hook parameters
        hook.fit(
            responses=z,
            labels=y,
            **self.interventions_cfg.intervention_params.hook_params,
        )

        # Save hook
        self.output_path.mkdir(exist_ok=True, parents=True)
        hook.save_state_dict(state_path=self.output_path / (module_name + ".statedict"))


class AtonceEstimator(LayerwiseEstimator):
    """An implementation of the Atonce intervention strategy."""

    def __init__(
        self,
        cfg: DictConfig,
        *args,
        **kwargs,
    ):
        """Initialize the InterventionsManager.

        Args:
            cfg (DictConfig): A configuration object defining intervention parameters and settings.

        Returns:
            None
        """
        super().__init__(cfg=cfg)

    def fit(self) -> None:
        """Learn interventions on all modules specified in the configuration.

        Args:
            None

        Returns:
            None
        """
        logger.info("Learning all interventions at once.")
        responses_path = ResponsesManager(self.responses_cfg).compute_responses()
        self.responses_loader = ResponsesLoader(
            root=responses_path,
            from_folders=[
                f"*/*/{self.interventions_cfg.intervention_params.pooling_op}",
            ],
            columns=["responses", "id", "subset"] + self.interventions_cfg.load_fields,
        )
        # Get only the module names requested via args
        module_names = self.responses_loader.get_attribute_values(
            "module_names",
            filter_patterns=self.interventions_cfg.model_params.module_names,
        )
        for module_name in module_names:
            self._learn_intervention(module_name=module_name)


class IncrementalEstimator(LayerwiseEstimator):
    """Incremental estimator class."""

    def __init__(
        self,
        cfg: DictConfig,
    ):
        """Initialize the InterventionsManager.

        Args:
            cfg (DictConfig): A configuration object defining intervention parameters and settings.

        Returns:
            None
        """
        super().__init__(cfg=cfg)

    def fit(self):
        """Learn interventions incrementally for all specified modules.

        This method allows for learning interventions on a module-by-module basis,
        leveraging previously learned interventions from previous modules when learning a new one.

        Returns:
            None
        """
        logger.info("Learning interventions incrementally")
        responses_manager = ResponsesManager(self.responses_cfg)
        # Get only the module names requested via args
        module_names = responses_manager.get_module_names(
            module_names=self.interventions_cfg.model_params.module_names
        )
        # Here we load intervention hooks for all those layers already computed (used_module_names).
        used_module_names = []
        for module_name in module_names:
            intervention_hooks = []
            for used in used_module_names:
                state_path = (
                    Path(self.interventions_cfg.cache_dir)
                    / self.output_path
                    / f"{used}.statedict"
                )
                # These hooks will intervene on the model, that's why we need hook_* args.
                hook = hooks.get_hook(
                    self.interventions_cfg.intervention_params.name,
                    module_name=used,
                    **self.interventions_cfg.intervention_params.hook_params,
                )
                hook.from_state_path(state_path)
                intervention_hooks.append(hook)
            logger.info(
                f"Loaded {len(intervention_hooks)} hooks: [{''.join(used_module_names[:1])}:{''.join(used_module_names[-1:])}]"
            )
            # Register all the hooks.
            logger.info(f"New inference on {module_name}.")

            # 1. Save responses for current module
            responses_path = responses_manager.compute_responses(
                [module_name], extra_pre_hooks=intervention_hooks
            )
            # 2. Learn intervention for current module
            # 2.1. Search directory tree for new responses
            self.responses_loader = ResponsesLoader(
                root=responses_path,
                from_folders=[
                    f"*/*/{self.interventions_cfg.intervention_params.pooling_op}",
                ],
                columns=["responses", "id", "subset"]
                + self.interventions_cfg.load_fields,
            )
            # 2.2. Learn intervention
            self._learn_intervention(module_name)
            # 3. Finally add the current module name in the used ones
            used_module_names.append(module_name)


class End2EndEstimator(BaseEstimator):
    """End2EndEstimator class.

    This class is used to learn interventional models using backprop.
    """

    def __init__(
        self,
        cfg: DictConfig,
    ):
        """Initialize the InterventionsManager.

        Args:
            cfg (DictConfig): A configuration object defining intervention parameters and settings.

        Returns:
            None
        """
        super().__init__(cfg=cfg)

    def fit(self):
        """Learn interventions globally for all specified modules.

        This method allows for learning interventions globaly end-to-end through backprop.

        Args:
            responses_manager (ResponsesManager): A manager to handle and load responses from files.

        Returns:
            None
        """
        logger.info("Learning interventions end-to-end")

        # Validate batch size
        if self.interventions_cfg.batch_size < 4:
            force_small_batch = self.interventions_cfg.get("force_small_batch", False)
            if not force_small_batch:
                raise ValueError(
                    f"Batch size {self.interventions_cfg.batch_size} is too small. "
                    f"The End2EndEstimator requires batch_size >= 4 for stable training. "
                    f"If you absolutely need to train with batch_size < 4, set "
                    f"'interventions.force_small_batch=true' to override this check."
                )
            logger.warning(
                f"Training with batch_size={self.interventions_cfg.batch_size} < 4. "
                f"This may lead to unstable training and poor results."
            )

        ######################################################################
        # DST RESPONSES
        # -------------
        # We first compute a set of responses for the dst (target) data mode.
        # These responses do not have gradients, and are only used as static
        # targets sampled from the dst distribution.
        ######################################################################
        responses_manager = ResponsesManager(self.responses_cfg)
        responses_cfg = copy_hydra_config(self.responses_cfg)
        dst_responses_cfg = copy_hydra_config(self.responses_cfg)
        dst_responses_cfg.balanced_data = False
        dst_responses_cfg.task_params.src_subsets = []  # d not compute dst responses anymore
        responses_manager.set_dataloader(dst_responses_cfg, drop_last=False)

        # Get only the module names requested via args
        dst_module_names = responses_manager.get_module_names(
            module_names=self.interventions_cfg.model_params.target_module_names
        )

        # Save target responses for all target modules to disk
        dst_responses_path = responses_manager.compute_responses(
            dst_module_names, keep_gradients=False
        )
        logger.info(f"Responses in {dst_responses_path}")

        # Load responses for all target modules
        # TODO: (Xavi) we're saving to disk and loading again.
        #       Not a big deal, but maybe we can optimize here.
        responses_loader = ResponsesLoader(
            root=dst_responses_path,
            from_folders=[
                f"*/*/{self.interventions_cfg.intervention_params.pooling_op}",
            ],
            columns=["responses", "id", "subset"] + self.interventions_cfg.load_fields,
        )

        dst_responses = {}
        dst_size = None
        for module_name in dst_module_names:
            dst_subset = responses_loader.load_data_subset(
                {
                    "module_names": [module_name],
                    "pooling_op": [
                        self.interventions_cfg.intervention_params.pooling_op
                    ],
                    "subset": list(self.interventions_cfg.task_params.dst_subsets),
                },
                num_workers=0,
            )
            dst_subset = ResponsesLoader.label_src_dst_subsets(
                dst_subset,
                src_subset=[],
                dst_subset=list(self.interventions_cfg.task_params.dst_subsets),
                balanced=False,
                seed=self.interventions_cfg.seed,
            )
            # Move responses out of GPU
            z = (
                torch.tensor(dst_subset["responses"])
                .detach()
                .to(self.responses_cfg.device)
            )
            # Sort target responses "per neuron"
            dst_size = z.shape[0]  # number of dst samples
            if self.interventions_cfg.batch_size > dst_size:
                raise ValueError(
                    f"{self.interventions_cfg.batch_size} must be equal or smaller than {dst_size}."
                )

            if ":" not in module_name:
                module_name = f"{module_name}:0"
            dst_responses[module_name] = (
                z.T
            )  # D, B (criterion expects batch to be last)
            logger.info(
                f"DST samples [{module_name}]: {dst_responses[module_name].shape}"
            )

        ######################################################################
        # SRC RESPONSES
        # -------------
        # Now we compute responses for the src (source) data mode.
        # These responses have gradients, and contribute to backpropagation
        # through BPHooks in the model.
        ######################################################################

        # Create response hooks for the target modules, so we can collect responses
        # to compute a loss.
        responses_manager.model.remove_hooks()
        src_module_names_with_shapes = responses_manager.get_module_names_with_shapes(
            module_names=self.interventions_cfg.model_params.module_names
        )

        # Here we create backpropagable intervention hooks for all those src layers requested.
        bp_hooks = []
        for module_name, module_shape in src_module_names_with_shapes.items():
            dim = module_shape[1] if len(module_shape) == 4 else module_shape[-1]
            hook = hooks.get_hook(
                self.interventions_cfg.intervention_params.name,
                module_name=module_name,
                dim=dim,
                **self.interventions_cfg.intervention_params.hook_params,
                **self.interventions_cfg.intervention_params.optimization_params,
            )
            bp_hooks.append(hook)
        logger.info(f"Created {len(bp_hooks)} backpropagable hooks.")
        for hook in bp_hooks:
            logger.debug(hook)

        # Now we want to keep gradients and we don't need dst responses
        # We update the responses_manager dataloader only, so we do not load the model twice.
        opt_params = self.interventions_cfg.intervention_params.optimization_params
        src_responses_cfg = copy_hydra_config(responses_cfg)
        # do not compute dst responses anymore
        src_responses_cfg.task_params.dst_subsets = []
        src_responses_cfg.max_batches = 1
        src_responses_cfg.batch_size = self.interventions_cfg.batch_size
        src_responses_cfg.balanced_data = False

        responses_manager.set_dataloader(
            src_responses_cfg, drop_last=True, max_batches=opt_params.max_batches
        )

        # This call actually sets response hooks for dst_module_names and pre_hooks for bp_hooks.
        responses_manager._set_hooks(
            module_names=dst_module_names,
            extra_pre_hooks=bp_hooks,
            keep_gradients=True,
        )

        # Infinitely cycle over the dataloader (an infinite iter())
        # and creates reshuffled cycles
        def reshuffled_cycle(dataloader):
            while True:
                yield from dataloader

        # Create the cycling iterator
        iloader = reshuffled_cycle(responses_manager.dataloader)

        # Optimization steps!
        parameters_to_optimize = []
        for hook in bp_hooks:
            parameters_to_optimize += list(hook.parameters())

        assert opt_params.optimizer in [
            "SGD",
            "Adam",
        ], f"Only SGD and Adam optimizers are supported."
        opt_kwargs = (
            opt_params.optimizer_kwargs if opt_params.optimizer == "Adam" else {}
        )
        optimizer = get_optimizer(opt_params.optimizer)(
            params=parameters_to_optimize, lr=opt_params.learning_rate, **opt_kwargs
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=opt_params.steps,
            eta_min=opt_params.learning_rate * opt_params.dampen_lr,
        )
        criterion = get_loss(opt_params.criterion)
        criterion_kwargs = opt_params.get("criterion_kwargs", {})

        # Choose proximal gradients strategy
        assert opt_params.proximal in ("l1", "sparse", None), opt_params.proximal
        if opt_params.proximal == "l1":
            apply_prox = functools.partial(
                proximal.apply_l1,
                lam_l1_weight=opt_params.regularization_l1_weight,
                lam_l1_bias=opt_params.regularization_l1_bias,
            )
        elif opt_params.proximal == "sparse":
            apply_prox = functools.partial(
                proximal.apply_sparse_group_lasso,
                lam_l1_weight=opt_params.regularization_l1_weight,
                lam_l2_weight=opt_params.regularization_l2_weight,
                lam_l1_bias=opt_params.regularization_l1_bias,
                lam_l2_bias=opt_params.regularization_l2_bias,
            )
        else:
            apply_prox = None

        history = defaultdict(list)
        tic = time.time()
        for step in range(opt_params.steps + opt_params.post_steps):
            # Post-training (re-fitting) after steps.
            # This is a WIP strategy to fine-tune those maps that have been
            # pre-selected by sparsity during training.
            if step == opt_params.steps:
                # Restart training only updating those parameters selected by sparsity!
                # Also restarts optimizer and scheduler.
                for module_idx, h in enumerate(bp_hooks):
                    # Find 'untouched' neurons
                    weights_mask = (
                        torch.abs(h.net.w1 - 1.0) <= opt_params.sparsity_threshold
                    )
                    biases_mask = torch.abs(h.net.b1) <= opt_params.sparsity_threshold
                    h.net.register_mask_hooks(
                        weights_mask=[m.item() for m in weights_mask.ravel()],
                        biases_mask=[m.item() for m in biases_mask.ravel()],
                    )

                for g in optimizer.param_groups:
                    g["lr"] = opt_params.learning_rate

                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=opt_params.post_steps,
                    eta_min=opt_params.learning_rate * opt_params.dampen_lr,
                )

            src_batch = next(iloader)
            with gradient_mode(True):
                # TODO: remove batch_idx
                responses_manager.model.update_hooks(batch_idx=0, batch=src_batch)
                outputs = None
                try:
                    if len(self.interventions_cfg.task_params.dst_subsets) == 1:
                        responses_manager.model(
                            src_batch,
                            **self.interventions_cfg.model_params.get(
                                "training_kwargs", {}
                            ),
                        )
                        outputs = responses_manager.model.get_hook_outputs()

                    elif len(self.interventions_cfg.task_params.dst_subsets) > 1:
                        raise NotImplementedError(
                            "For now, only 1 dst subset is considered. "
                            "The code allows using more since we store responses in a dictionary, but this capability "
                            "has not been incorporated."
                        )

                except hooks.custom_exceptions.TargetModuleReached:
                    pass

                # Get the responses, sort and adapt shape
                src_responses = {}
                src_size = None
                for module_name, output in outputs.items():
                    src_responses[module_name] = torch.stack(
                        [output[i]["responses"] for i in range(len(output))],
                        dim=0,
                    ).T  # D, B with gradients
                    src_size = src_responses[module_name].shape[1]

                cost_x_layer = []
                wandb_log = {}
                total_loss = 0
                for module_name in src_responses.keys():
                    # IMPORTANT: Our loss is matching sorted samples per neuron. We apply then MSE on transposed responses,
                    # so they are (D, B) where B is the number of samples. MSE will compute distance between samples
                    # and average across neurons (D).
                    module_loss = criterion(
                        src_responses[module_name],
                        dst_responses[module_name],
                        **criterion_kwargs,
                    )
                    cost_x_layer.append(module_loss)
                    wandb_log[f"losses/{module_name}"] = module_loss
                    total_loss += module_loss / len(src_responses)

                # Final loss averaged across layers
                history["loss"].append(total_loss.item())
                last_lr = scheduler.get_last_lr()[0]

                wandb_log["losses/cost"] = total_loss.item()
                wandb_log["learning_rate"] = last_lr
                wandb_log["num_trainable_parameters"] = len(parameters_to_optimize)

                # Backward pass and update
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Apply Proximal Gradient
                reg = 0.0 if apply_prox is None else apply_prox(bp_hooks, tau=last_lr)
                # Step after proximal, so LR is the same as in the optimizer.
                scheduler.step()

                # Log stuff
                logger.info(
                    f"Step {step}/{opt_params.steps + opt_params.post_steps}, cost: {total_loss:0.4f}, reg: {reg:0.4f}, loss: {total_loss:0.4f}"
                )
                if self.interventions_cfg.wandb.mode != "disabled":
                    stats = _compute_statistics(
                        bp_hooks,
                        sparsity_threshold=opt_params.sparsity_threshold,
                        # careful, can overwhelm W&B
                        add_histograms=False,
                    )
                    wandb_log.update(stats)
                    wandb_log["losses/regularization"] = reg
                    wandb.log(wandb_log, step=step)

        elapsed = time.time() - tic
        logger.info(
            f"Training took {elapsed:.2f} seconds ({elapsed / opt_params.steps:0.2f} s/step)"
        )
        # Hooks learnt, saving!
        self.output_path.mkdir(exist_ok=True, parents=True)
        for hook in bp_hooks:
            hook.save_state_dict(
                state_path=self.output_path / (hook.module_name + ".statedict")
            )
        #### The following block saves a plot with per_layer support ####
        layer_groups = defaultdict(list)
        for module_idx, h in enumerate(bp_hooks):
            total_support = float(h.net.support(opt_params.sparsity_threshold))
            total_activations = float(h.net.dim)
            module_group = ".".join(
                [name for name in h.module_name.split(".") if not name.isdigit()]
            )
            layer_groups[module_group].append(
                (module_idx, h.module_name, total_support / total_activations)
            )
        # Create a bar plot
        pylab.figure(figsize=(14, 6))
        df_data = []
        for module_group, module_stats in layer_groups.items():
            module_idxs, _, module_supports = zip(*module_stats, strict=True)
            df_data.extend(module_stats)
            pylab.plot(
                module_idxs,
                module_supports,
                label=module_group,
                marker="o",
            )  # Directly using labels
        pylab.ylabel("Support")
        pylab.legend()
        pylab.xticks(rotation=90, ha="right")  # Fully vertical labels, right-aligned
        pylab.tight_layout()
        pylab.ylim(0, 1.05)
        # Save and log plot
        results_dir = Path(self.interventions_cfg.results_dir)
        results_dir.mkdir(exist_ok=True, parents=True)
        plot_path = results_dir / "module_support.png"
        pylab.savefig(
            plot_path,
            bbox_inches="tight",
        )
        pylab.close()
        df = (
            pd.DataFrame(df_data, columns=["id", "module_name", "support"])
            .sort_values(["id"])
            .set_index("id")
        )
        df.to_csv(results_dir / "module_support.csv")
        if wandb.run is not None:
            wandb.log({"module_support": wandb.Image(str(plot_path))})
            wandb.log(
                {"support": wandb.Table(data=df, columns=["module_name", "support"])}
            )
        #### End Plot ####


def _compute_statistics(
    hooks: list[BPHook],
    sparsity_threshold: float,
    *,
    add_histograms: bool = False,
) -> dict[str, t.Any]:
    total_support = 0.0
    total_activations = 0.0
    stats = {}
    for h in hooks:
        total_support += float(h.net.support(sparsity_threshold))
        total_activations += float(h.net.dim)
        # careful, can overwhelm W&B
        if add_histograms:
            stats[f"weights/{h.module_name}"] = wandb.Histogram(
                h.net.w1.detach().cpu().squeeze(0).numpy()
            )
            stats[f"biases/{h.module_name}"] = wandb.Histogram(
                h.net.b1.detach().cpu().squeeze(0).numpy()
            )
    stats["total_support"] = total_support / total_activations
    return stats
