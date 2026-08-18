"""
Unit tests for the configuration system.

Tests:
- ModelConfig creation and validation
- ExtractorConfig creation and validation
- SteerConfig creation and validation
- PipelineConfig loading and validation
"""

import pytest
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from Steering.config import (
    ModelConfig, 
    ExtractorConfig, 
    SteerConfig,
    PostProcessConfig,
    PipelineConfig,
)
from Steering.data import TRAIN_DATASET_REGISTRY, TEST_DATASET_REGISTRY


class TestModelConfig:
    """Tests for ModelConfig dataclass."""
    
    def test_default_values(self):
        """Test ModelConfig has correct default values."""
        config = ModelConfig()
        assert config.name == "google/gemma-2-2b"
        assert config.device == "cuda"
        assert config.dtype == "bfloat16"
    
    def test_custom_values(self):
        """Test ModelConfig accepts custom values."""
        config = ModelConfig(
            name="meta-llama/Llama-3.2-3B-Instruct",
            device="cuda:1",
            dtype="float16"
        )
        assert config.name == "meta-llama/Llama-3.2-3B-Instruct"
        assert config.device == "cuda:1"
        assert config.dtype == "float16"
    
    def test_get_dtype_bfloat16(self):
        """Test get_dtype returns correct torch dtype for bfloat16."""
        import torch
        config = ModelConfig(dtype="bfloat16")
        assert config.get_dtype() == torch.bfloat16
    
    def test_get_dtype_float16(self):
        """Test get_dtype returns correct torch dtype for float16."""
        import torch
        config = ModelConfig(dtype="float16")
        assert config.get_dtype() == torch.float16
    
    def test_get_dtype_float32(self):
        """Test get_dtype returns correct torch dtype for float32."""
        import torch
        config = ModelConfig(dtype="float32")
        assert config.get_dtype() == torch.float32
    
    def test_to_dict(self):
        """Test ModelConfig can be converted to dict."""
        config = ModelConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "name" in d
        assert "device" in d
        assert "dtype" in d


class TestExtractorConfig:
    """Tests for ExtractorConfig dataclass."""
    
    def test_valid_method_caa(self):
        """Test ExtractorConfig accepts valid CAA method."""
        config = ExtractorConfig(method="CAA", layer=14)
        assert config.method == "CAA"
        assert config.layer == 14
    
    def test_valid_method_angular(self):
        """Test ExtractorConfig accepts valid ANGULAR method."""
        config = ExtractorConfig(method="ANGULAR", layer=14)
        assert config.method == "ANGULAR"
    
    def test_default_batch_size(self):
        """Test ExtractorConfig has correct default batch_size."""
        config = ExtractorConfig(method="CAA", layer=14)
        assert config.batch_size == 8

    def test_weightsteer_extractor_params(self):
        """Test ExtractorConfig has correct defaults and scoped fields for WEIGHTSTEER."""
        config = ExtractorConfig(
            method="WEIGHTSTEER",
            layer=14,
            weight_steer_lr=0.0001,
            weight_steer_epochs=5,
            weight_steer_target_modules=["q_proj"],
            weight_steer_lora_r=16,
            weight_steer_lora_alpha=32,
            weight_steer_lora_dropout=0.1,
        )
        assert config.method == "WEIGHTSTEER"
        assert config.weight_steer_lr == 0.0001
        assert config.weight_steer_epochs == 5
        assert config.weight_steer_target_modules == ["q_proj"]
        assert config.weight_steer_lora_r == 16
        assert config.weight_steer_lora_alpha == 32
        assert config.weight_steer_lora_dropout == 0.1
        
        # Test to_dict with method_scoped=True
        scoped_dict = config.to_dict(method_scoped=True)
        assert "weight_steer_lr" in scoped_dict
        assert "weight_steer_epochs" in scoped_dict
        assert "weight_steer_target_modules" in scoped_dict
        assert "weight_steer_lora_r" in scoped_dict
        assert "weight_steer_lora_alpha" in scoped_dict
        assert "weight_steer_lora_dropout" in scoped_dict
        assert "top_k" not in scoped_dict  # Since WEIGHTSTEER is dense (not SAE)


class TestSteerConfig:
    """Tests for SteerConfig dataclass."""
    
    def test_valid_creation(self):
        """Test SteerConfig can be created with valid values."""
        config = SteerConfig(method="CAA", layer=14, coeff=2.0)
        assert config.method == "CAA"
        assert config.layer == 14
        assert config.coeff == 2.0
    
    def test_negative_coeff_allowed(self):
        """Test SteerConfig allows negative coefficients."""
        config = SteerConfig(method="CAA", layer=14, coeff=-1.5)
        assert config.coeff == -1.5
    
    def test_zero_coeff_allowed(self):
        """Test SteerConfig allows zero coefficient (baseline)."""
        config = SteerConfig(method="CAA", layer=14, coeff=0.0)
        assert config.coeff == 0.0


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""
    
    def test_load_from_file(self, sample_config_file):
        """Test PipelineConfig can be loaded from file."""
        config = PipelineConfig.load(sample_config_file)
        assert config.name == "test_experiment"
        assert config.extractor.method == "CAA"
        assert config.steer.coeff == 2.0
    
    def test_to_dict(self, sample_config_file):
        """Test PipelineConfig can be converted to dict."""
        config = PipelineConfig.load(sample_config_file)
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "name" in d
        assert "extractor" in d
        assert "steer" in d
    
    def test_load_nonexistent_file(self, tmp_path):
        """Test PipelineConfig raises error for nonexistent file."""
        with pytest.raises(Exception):  # FileNotFoundError or ConfigError
            PipelineConfig.load(tmp_path / "nonexistent.json")
    
    def test_load_invalid_json(self, tmp_path):
        """Test PipelineConfig raises error for invalid JSON."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{ invalid json }")
        
        with pytest.raises(Exception):  # JSONDecodeError or ConfigError
            PipelineConfig.load(bad_file)

    def test_post_process_defaults(self, sample_config_file):
        """Test PipelineConfig has post_process defaults when omitted."""
        config = PipelineConfig.load(sample_config_file)
        assert isinstance(config.post_process, PostProcessConfig)
        assert config.post_process.enabled is False
        assert config.post_process.use_classifier is False
        assert config.post_process.scale == 1.0
        assert config.post_process.negative is False

    def test_post_process_from_dict(self):
        """Test PipelineConfig parses nested post_process block."""
        config = PipelineConfig.from_dict(
            {
                "name": "pp-test",
                "model": {"name": "google/gemma-2-2b", "device": "cpu", "dtype": "float32"},
                "extractor": {"method": "CAA", "layer": 14},
                "steer": {"method": "CAA", "layer": 14, "coeff": 1.0},
                "train_dataset": "sycophancy",
                "test_dataset": "csqa",
                "post_process": {
                    "enabled": True,
                    "source": "org/glp-repo",
                    "noise_rate": 0.4,
                    "num_timesteps": 12,
                    "use_classifier": True,
                    "scale": 1.5,
                    "negative": True,
                    "classifier_guidance_start_step": 2,
                },
            }
        )
        assert config.post_process.enabled is True
        assert config.post_process.source == "org/glp-repo"
        assert config.post_process.noise_rate == 0.4
        assert config.post_process.num_timesteps == 12
        assert config.post_process.use_classifier is True
        assert config.post_process.scale == 1.5
        assert config.post_process.negative is True
        assert config.post_process.classifier_guidance_start_step == 2


class TestDatasetRegistries:
    """Tests for dataset registry configurations."""
    
    def test_train_registry_has_sycophancy(self):
        """Test TRAIN_DATASET_REGISTRY contains sycophancy."""
        assert "sycophancy" in TRAIN_DATASET_REGISTRY
    
    def test_train_registry_has_refusal(self):
        """Test TRAIN_DATASET_REGISTRY contains refusal."""
        assert "refusal" in TRAIN_DATASET_REGISTRY
    
    def test_test_registry_has_csqa(self):
        """Test TEST_DATASET_REGISTRY contains csqa."""
        assert "csqa" in TEST_DATASET_REGISTRY
    
    def test_test_registry_has_refusal(self):
        """Test TEST_DATASET_REGISTRY contains refusal variants."""
        assert "refusal_ab" in TEST_DATASET_REGISTRY or "refusal_open" in TEST_DATASET_REGISTRY


class TestFinetuneConfig:
    """Tests for FinetuneConfig dataclass."""
    
    def test_default_values(self):
        from Steering.finetune.config import FinetuneConfig
        config = FinetuneConfig()
        assert config.name == "finetune"
        assert config.save_vector is None
        assert config.load_vector is None
        assert config.output == "./Results/finetune"

    def test_custom_method_target_modules(self):
        from Steering.finetune.config import FinetuneConfig
        config = FinetuneConfig.from_dict({
            "method": {
                "target_modules": ["up_proj", "down_proj"]
            }
        })
        assert list(config.method.target_modules) == ["up_proj", "down_proj"]

    def test_custom_values(self):
        from Steering.finetune.config import FinetuneConfig
        config = FinetuneConfig(
            save_vector="./Vector/FINETUNE/custom_save",
            load_vector="./Vector/FINETUNE/custom_load",
        )
        assert config.save_vector == "./Vector/FINETUNE/custom_save"
        assert config.load_vector == "./Vector/FINETUNE/custom_load"

    def test_to_dict_from_dict(self):
        from Steering.finetune.config import FinetuneConfig
        config = FinetuneConfig(
            save_vector="./Vector/FINETUNE/custom_save",
            load_vector="./Vector/FINETUNE/custom_load",
        )
        d = config.to_dict(include_none=True)
        assert d["save_vector"] == "./Vector/FINETUNE/custom_save"
        assert d["load_vector"] == "./Vector/FINETUNE/custom_load"
        
        config2 = FinetuneConfig.from_dict(d)
        assert config2.save_vector == "./Vector/FINETUNE/custom_save"
        assert config2.load_vector == "./Vector/FINETUNE/custom_load"
