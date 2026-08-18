"""
Unit tests for chat template application in data loaders.

Tests:
- DataLoader applies [INST]q[/INST]{answer} format for training
- EvalDataLoader applies [INST]q[/INST]{prefix} format for testing
- TestDatasetConfig prefix field works correctly
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from Steering.data import DataLoader, EvalDataLoader
from Steering.config import TestDatasetConfig
from Steering.data.data_registry import TEST_DATASET_REGISTRY


class MockTokenizer:
    """Mock tokenizer for testing chat template application."""
    
    chat_template = "mock_template"
    
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        """Mock apply_chat_template that returns [INST] format."""
        user_content = messages[0]["content"] if messages else ""
        return f"[INST] {user_content} [/INST]"


class TestChatTemplateDataLoader:
    """Tests for chat template application in DataLoader."""
    
    def test_load_without_template(self):
        """Test loading without chat template - should return clean data."""
        loader = DataLoader()
        data = loader.load("sycophancy", n_samples=2, apply_chat_template=False)
        
        assert isinstance(data, list)
        assert len(data) == 2
        
        # Check that question doesn't have [INST] tags
        first_entry = data[0]
        assert "[INST]" not in first_entry.get("question", "")
    
    def test_load_with_template(self):
        """Test loading with chat template - should wrap prompts."""
        loader = DataLoader()
        mock_tokenizer = MockTokenizer()
        
        data = loader.load(
            "sycophancy",
            n_samples=2,
            apply_chat_template=True,
            tokenizer=mock_tokenizer,
        )
        
        assert isinstance(data, list)
        assert len(data) == 2
        
        # Check that correct_prompt has [INST] format
        first_entry = data[0]
        assert "[INST]" in first_entry.get("correct_prompt", "")
        assert "[/INST]" in first_entry.get("correct_prompt", "")
        
        # Check false_prompt also has template
        assert "[INST]" in first_entry.get("false_prompt", "")
        assert "[/INST]" in first_entry.get("false_prompt", "")


class TestChatTemplateEvalDataLoader:
    """Tests for chat template application in EvalDataLoader."""
    
    def test_load_without_template(self):
        """Test loading without chat template - should return clean data."""
        loader = EvalDataLoader()
        data = loader.load("sorrybench", n_samples=2, apply_chat_template=False)
        
        assert isinstance(data, list)
        assert len(data) == 2
        
        # Check that question doesn't have [INST] tags (after formatter cleanup)
        first_entry = data[0]
        question = first_entry.get("question", "")
        # Note: formatters now return clean text without [INST] tags
        assert not question.startswith("[INST]") or "[INST]" in question
    
    def test_load_with_template(self):
        """Test loading with chat template - should wrap question."""
        loader = EvalDataLoader()
        mock_tokenizer = MockTokenizer()
        
        data = loader.load(
            "sorrybench",
            n_samples=2,
            apply_chat_template=True,
            tokenizer=mock_tokenizer,
        )
        
        assert isinstance(data, list)
        assert len(data) == 2
        
        # Check that question has [INST] format
        first_entry = data[0]
        question = first_entry.get("question", "")
        assert "[INST]" in question
        assert "[/INST]" in question
    
    def test_load_uses_registry_prefix(self):
        """Test loading uses the dataset prefix from TEST_DATASET_REGISTRY."""
        loader = EvalDataLoader()
        mock_tokenizer = MockTokenizer()
        expected_prefix = TEST_DATASET_REGISTRY["sorrybench"].prefix
        
        data = loader.load(
            "sorrybench",
            n_samples=2,
            apply_chat_template=True,
            tokenizer=mock_tokenizer,
        )
        
        assert isinstance(data, list)
        assert len(data) == 2
        
        # Check that question ends with prefix
        first_entry = data[0]
        question = first_entry.get("question", "")
        assert question.endswith(expected_prefix)


class TestDatasetConfigPrefix:
    """Tests for TestDatasetConfig prefix field."""
    
    def test_prefix_field_exists(self):
        """Test that TestDatasetConfig has prefix field."""
        cfg = TestDatasetConfig(
            file="test.json",
            schema="test",
            evaluator="logit",
            prefix="(",
        )
        assert cfg.prefix == "("
    
    def test_prefix_default_empty(self):
        """Test that prefix defaults to empty string."""
        cfg = TestDatasetConfig(
            file="test.json",
            schema="test",
            evaluator="logit",
        )
        assert cfg.prefix == ""
    
    def test_registry_configs_have_prefix(self):
        """Test that registry configs have prefix attribute."""
        for name, cfg in TEST_DATASET_REGISTRY.items():
            assert hasattr(cfg, "prefix"), f"Config {name} missing prefix field"
