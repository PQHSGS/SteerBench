"""
Configuration Module.

Provides all configuration classes and registries for the Steering framework.
"""

# Model configuration
from .models import ModelConfig, MODEL_SAE_REGISTRY

# Method configuration  
from .methods import (
    ExtractorConfig,
    SteerConfig,
    SAE_METHODS,
)

# Post-process configuration
from .post_process import PostProcessConfig

# Dataset registries
from .datasets import (
    TRAIN_ROOT,
    TEST_ROOT,
    TrainDatasetConfig,
    CompositeDatasetConfig,
    TestDatasetConfig,
)

# Experiment configuration
from .pipeline import PipelineConfig

# Result structures
from .results import SampleResult, EvalResult, SteeringVector


__all__ = [
    # Model
    "ModelConfig",
    "MODEL_SAE_REGISTRY",
    # Methods
    "ExtractorConfig",
    "SteerConfig",
    "SAE_METHODS",
    "PostProcessConfig",
    # Datasets
    "TRAIN_ROOT",
    "TEST_ROOT",
    "TrainDatasetConfig",
    "CompositeDatasetConfig",
    "TestDatasetConfig",
    # Experiment
    "PipelineConfig",
    # Results
    "SampleResult",
    "EvalResult",
    "SteeringVector",
]
