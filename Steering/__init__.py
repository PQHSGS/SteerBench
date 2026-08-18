"""
Steering Vector Benchmark

A unified framework for steering vector extraction and application.

Quick Start:
    from Steering import SteeringPipeline, PipelineConfig
    
    config = PipelineConfig.load("configs/caa_example.json")
    pipeline = SteeringPipeline.from_config(config)
    results = pipeline.run()

Factory Usage:
    from Steering import create_extractor, create_steer_model
    
    extractor = create_extractor("CAA", model, layer=14)
    vector = extractor.extract(target_data, contrast_data)
    
    steer_model = create_steer_model("CAA", model, layer=14, vector=vector)
    response = steer_model.generate("Your prompt", coeff=2.0)
"""

# Base classes
from .base import BaseExtractor, BaseSteerModel

# Configuration (from config/)
from .config import (
    # Model
    ModelConfig,
    MODEL_SAE_REGISTRY,
    # Methods
    ExtractorConfig,
    SteerConfig,
    PostProcessConfig,
    SAE_METHODS,

    # Experiment
    PipelineConfig,
)

# Data loading (from data/)
from .data import (
    DataLoader, 
    EvalDataLoader,
    # Datasets
    TRAIN_DATASET_REGISTRY,
    TEST_DATASET_REGISTRY
)

# Pipeline
from .pipeline import SteeringPipeline, EvalResult, SampleResult
# Utilities
from .logger import setup_logger
from .exceptions import (
    SteeringError,
    ConfigError,
    DataLoadError,
    ModelError,
    ExtractionError,
    EvaluationError,
)


# =============================================================================
# Factory Functions
# =============================================================================


def list_methods():
    """List available steering methods."""
    return SteeringPipeline.list_methods()





__all__ = [
    # Main API
    "list_methods",
    "SteeringPipeline",
    "EvalResult",
    "SampleResult",
    # Base classes
    "BaseExtractor",
    "BaseSteerModel",
    # Config
    "ModelConfig",
    "ExtractorConfig",
    "SteerConfig",
    "PostProcessConfig",
    "PipelineConfig",
    "SAE_METHODS",
    "MODEL_SAE_REGISTRY",
    # Data
    "DataLoader",
    "EvalDataLoader",
    "TRAIN_DATASET_REGISTRY",
    "TEST_DATASET_REGISTRY",
    # Utilities
    "setup_logger",
    "SteeringError",
    "ConfigError",
    "DataLoadError",
    "ModelError",
    "ExtractionError",
    "EvaluationError",
]
