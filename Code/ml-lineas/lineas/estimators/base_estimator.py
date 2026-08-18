# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

from abc import ABC
from pathlib import Path

from omegaconf import DictConfig


class BaseEstimator(ABC):
    """Manages the learning and application of interventions on a generative model.

    This class handles loading model responses, configuring intervention parameters,
    and learning intervention strategies for specific modules within the model.

    Interventions are techniques used to modify the behavior of a model by injecting biases
    or constraints during inference. They can be used for tasks like domain adaptation,
    style transfer, or controlling the output of a model.

    Attributes:
        cfg (DictConfig): A configuration object containing parameters for the interventions manager.

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
        self.responses_cfg = cfg.responses
        self.interventions_cfg = cfg.interventions
        self.output_path = self.get_output_path(self.interventions_cfg)

    @staticmethod
    def get_output_path(cfg: DictConfig) -> Path:
        """Determine the output directory for storing intervention data.

        Args:
            cfg (DictConfig): The configuration object containing intervention parameters.

        Returns:
            Path: The path to the directory where intervention results will be saved.
        """
        model_name = Path(cfg.model_params.model_path).name
        sorted_keys = list(sorted(cfg.intervention_params.keys()))
        name = "-".join(
            [
                str(cfg.intervention_params[k])
                for k in sorted_keys
                if k
                not in ["hook_params", "optimization_params", "state_path", "layers"]
            ]
        )
        return Path(cfg.cache_dir) / cfg.tag / model_name / name

    @staticmethod
    def patch_output_path_(cfg: DictConfig) -> Path:
        """Patch the output path with the state path."""
        output_path = BaseEstimator.get_output_path(cfg)
        cfg.intervention_params.state_path = output_path

    def fit(self) -> None:
        """Learn an intervention strategy for a specific module."""
        raise NotImplementedError
