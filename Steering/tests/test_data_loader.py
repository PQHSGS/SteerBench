"""
Unit tests for the data loading system.

Tests:
- DataLoader basic functionality (new unified API)
- EvalDataLoader basic functionality
- Error handling for invalid datasets
- Registry integrity tests
"""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from Steering.data import DataLoader, EvalDataLoader
from Steering.exceptions import DataLoadError
from Steering.config import TrainDatasetConfig, TestDatasetConfig
from Steering.data.data_registry import TRAIN_DATASET_REGISTRY, TEST_DATASET_REGISTRY


class TestDataLoader:
    """Tests for DataLoader class (new unified API)."""
    
    def test_initialization(self):
        """Test DataLoader initializes correctly."""
        loader = DataLoader()
        assert loader is not None
    
    def test_load_invalid_dataset(self):
        """Test DataLoader raises DataLoadError for invalid dataset."""
        loader = DataLoader()
        with pytest.raises(DataLoadError) as exc_info:
            loader.load("nonexistent_dataset")
        
        assert "Unknown dataset" in str(exc_info.value)
        assert "nonexistent_dataset" in str(exc_info.value)
    
    def test_load_sycophancy(self):
        """Test loading sycophancy dataset."""
        loader = DataLoader()
        data = loader.load("sycophancy", n_samples=5)
        
        assert isinstance(data, list)
        assert len(data) == 5
        
        # Check expected fields after formatting
        first_entry = data[0]
        assert "correct_prompt" in first_entry
        assert "false_prompt" in first_entry
    def test_load_composite_data(self):
        """Test loading composite dataset."""
        loader = DataLoader()
        data = loader.load("refusal_cast_responses", n_samples=5)
        
        assert isinstance(data, list)
        assert len(data) == 5
        
        # Check expected fields after formatting
        first_entry = data[0]
        assert "correct_prompt" in first_entry
        assert "false_prompt" in first_entry
    
    def test_load_without_formatting(self):
        """Test loading data without schema formatting."""
        loader = DataLoader()
        data = loader.load("sycophancy", n_samples=5, format=False)
        
        assert isinstance(data, list)
        assert len(data) == 5
    
    def test_get_config(self):
        """Test get_config returns correct config."""
        loader = DataLoader()
        cfg = loader.get_config("sycophancy")
        
        assert isinstance(cfg, TrainDatasetConfig)
        assert cfg.target_key == "correct_prompt"
        assert cfg.contrast_key == "false_prompt"
    
    def test_list_datasets(self):
        """Test list_datasets returns available datasets."""
        loader = DataLoader()
        datasets = loader.list_datasets()
        
        assert isinstance(datasets, list)
        assert "sycophancy" in datasets
        assert "refusal" in datasets
    
    def test_error_message_includes_available_datasets(self):
        """Test error message includes list of available datasets."""
        loader = DataLoader()
        with pytest.raises(DataLoadError) as exc_info:
            loader.load("fake_dataset")
        
        error_msg = str(exc_info.value)
        assert "Available" in error_msg


class TestEvalDataLoader:
    """Tests for EvalDataLoader class."""
    
    def test_initialization(self):
        """Test EvalDataLoader initializes correctly."""
        loader = EvalDataLoader()
        assert loader is not None
    
    def test_load_invalid_dataset(self):
        """Test EvalDataLoader raises DataLoadError for invalid dataset."""
        loader = EvalDataLoader()
        with pytest.raises(DataLoadError) as exc_info:
            loader.load("nonexistent_test_dataset")
        
        assert "Unknown dataset" in str(exc_info.value)
    
    def test_get_config(self):
        """Test get_config returns correct config."""
        loader = EvalDataLoader()
        cfg = loader.get_config("csqa")
        
        assert isinstance(cfg, TestDatasetConfig)
        assert cfg.evaluator == "logit"
    
    def test_list_datasets(self):
        """Test list_datasets returns available datasets."""
        loader = EvalDataLoader()
        datasets = loader.list_datasets()
        
        assert isinstance(datasets, list)
        assert "csqa" in datasets
        assert "gms8k" in datasets


class TestUnifiedRegistry:
    """Tests for unified dataset registries."""
    
    def test_train_registry_has_sycophancy(self):
        """Test TRAIN_DATASET_REGISTRY has sycophancy."""
        assert "sycophancy" in TRAIN_DATASET_REGISTRY
        cfg = TRAIN_DATASET_REGISTRY["sycophancy"]
        assert isinstance(cfg, TrainDatasetConfig)
        assert cfg.schema == "sycophancy_personas"
    
    def test_train_registry_has_refusal(self):
        """Test TRAIN_DATASET_REGISTRY has refusal."""
        assert "refusal" in TRAIN_DATASET_REGISTRY
        cfg = TRAIN_DATASET_REGISTRY["refusal"]
        assert isinstance(cfg, TrainDatasetConfig)
    
    def test_train_registry_path_property(self):
        """Test TrainDatasetConfig.path returns valid Path."""
        cfg = TRAIN_DATASET_REGISTRY["sycophancy"]
        assert isinstance(cfg.path, Path)
        assert "sycophancy" in str(cfg.path)
    
    def test_test_registry_has_csqa(self):
        """Test TEST_DATASET_REGISTRY has csqa."""
        assert "csqa" in TEST_DATASET_REGISTRY
        cfg = TEST_DATASET_REGISTRY["csqa"]
        assert isinstance(cfg, TestDatasetConfig)
        assert cfg.evaluator == "logit"
    
    def test_test_registry_has_refusal(self):
        """Test TEST_DATASET_REGISTRY has refusal_ab."""
        assert "refusal_ab" in TEST_DATASET_REGISTRY
        cfg = TEST_DATASET_REGISTRY["refusal_ab"]
        assert cfg.evaluator == "logit"
    
    def test_test_registry_path_property(self):
        """Test TestDatasetConfig.path returns valid Path."""
        cfg = TEST_DATASET_REGISTRY["csqa"]
        assert isinstance(cfg.path, Path)


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent
