# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from lineas.estimators.base_estimator import (
    BaseEstimator,
)
from lineas.estimators.get_estimator import get_estimator
from lineas.utils import utils

os.environ["TOKENIZERS_PARALLELISM"] = "False"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Fix to cast dtypes in yaml config
OmegaConf.register_new_resolver(
    name="dtype", resolver=lambda dtype: hydra.utils.get_object(dtype)
)


@hydra.main(config_path="../configs", config_name="text_generation")
def main(cfg: DictConfig) -> BaseEstimator | None:
    return learn_intervention(cfg)


def learn_intervention(cfg: DictConfig) -> BaseEstimator | None:
    """Learn interventions on a model from a given configuration.

    Args:
        cfg (DictConfig): The configuration parameters for the intervention learning process.

    Returns:
        BaseEstimator: An instance of the BaseEstimator class if any interventions were learned, otherwise None.
    """
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg, resolve=False)}")

    # Set random seed
    utils.seed_all(cfg.seed)
    interventions_estimator = get_estimator(
        cfg,
    )
    interventions_estimator.fit()
    return interventions_estimator


if __name__ == "__main__":
    main()
