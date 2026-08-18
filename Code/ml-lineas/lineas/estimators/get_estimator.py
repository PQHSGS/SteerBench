# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import contextlib

from omegaconf import DictConfig

from lineas.estimators.intervention_estimator import (
    AtonceEstimator,
    End2EndEstimator,
    IncrementalEstimator,
)

ESTIMATORS_REGISTRY = {
    "atonce": AtonceEstimator,
    "incr": IncrementalEstimator,
    "end2end": End2EndEstimator,
}


def get_estimator(
    cfg: DictConfig,
    **kwargs,
):
    return ESTIMATORS_REGISTRY[cfg.intervention_params.estimation_strategy](
        cfg,
        **kwargs,
    )
