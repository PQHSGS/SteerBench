"""
Test configuration and fixtures for the Steering Vector Benchmark.
"""

import pytest
from pathlib import Path
import json
import tempfile


@pytest.fixture
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def sample_config_dict():
    """Create a sample config dictionary for testing."""
    return {
        "name": "test_experiment",
        "description": "Test experiment configuration",
        "model": {
            "name": "google/gemma-2-2b",
            "device": "cuda:0",
            "dtype": "bfloat16"
        },
        "extractor": {
            "method": "CAA",
            "layer": 14,
            "batch_size": 8
        },
        "steer": {
            "method": "CAA",
            "layer": 14,
            "coeff": 2.0
        },
        "train_dataset": "sycophancy",
        "test_dataset": "csqa",
        "n_train": 100,
        "n_test": 50
    }


@pytest.fixture
def sample_config_file(sample_config_dict, tmp_path):
    """Create a temporary config file for testing."""
    config_path = tmp_path / "test_config.json"
    with open(config_path, 'w') as f:
        json.dump(sample_config_dict, f, indent=2)
    return config_path


@pytest.fixture
def invalid_config_dict():
    """Create an invalid config dictionary for testing validation."""
    return {
        "name": "invalid_test",
        "extractor": {
            "method": "NONEXISTENT_METHOD",
            "layer": 14
        },
        "steer": {
            "method": "CAA",
            "layer": 14,
            "coeff": 2.0
        },
        "train_dataset": "sycophancy",
        "test_dataset": "csqa"
    }


@pytest.fixture
def invalid_config_file(invalid_config_dict, tmp_path):
    """Create a temporary invalid config file for testing."""
    config_path = tmp_path / "invalid_config.json"
    with open(config_path, 'w') as f:
        json.dump(invalid_config_dict, f, indent=2)
    return config_path
