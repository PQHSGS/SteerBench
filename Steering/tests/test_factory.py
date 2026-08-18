"""
Unit tests for the Steering module API.

Tests:
- list_methods function
- SAE_METHODS constant
- create_extractor / create_steer_model factories
- Evaluator functions (from Steering.eval_pipeline module)
"""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from Steering import (
    list_methods,
    SAE_METHODS,
)
from Steering.pipeline import SteeringPipeline
from Steering.evaluators import EVALUATOR_MAP


def list_evaluators():
    """Wrapper for SteeringPipeline.list_evaluators()."""
    return SteeringPipeline.list_evaluators()


def create_evaluator(name: str, device: str):
    """Create an evaluator instance by name."""
    if name not in EVALUATOR_MAP:
        raise KeyError(f"Unknown evaluator: {name}")
    return EVALUATOR_MAP[name](device=device)

class TestListMethods:
    """Tests for list_methods function."""
    
    def test_returns_list(self):
        methods = list_methods()
        assert isinstance(methods, list)
    
    def test_has_caa(self):
        methods = list_methods()
        assert "CAA" in methods
    
    def test_has_cast(self):
        methods = list_methods()
        assert "CAST" in methods
    
    def test_has_angular(self):
        methods = list_methods()
        assert "ANGULAR" in methods
    
    def test_has_sae_methods(self):
        methods = list_methods()
        assert "SAS" in methods
        assert "SPARE" in methods
        assert "SRE" in methods
    
    def test_no_duplicates(self):
        methods = list_methods()
        assert len(methods) == len(set(methods))


class TestListEvaluators:
    """Tests for list_evaluators function."""
    
    def test_returns_list(self):
        evaluators = list_evaluators()
        assert isinstance(evaluators, list)
    
    def test_has_multiple_choice(self):
        evaluators = list_evaluators()
        assert "multiple_choice" in evaluators
    
    def test_has_refusal(self):
        evaluators = list_evaluators()
        assert "refusal" in evaluators


class TestSAEMethods:
    """Tests for SAE_METHODS constant."""
    
    def test_is_set(self):
        assert isinstance(SAE_METHODS, set)
    
    def test_contains_sas(self):
        assert "SAS" in SAE_METHODS
    
    def test_contains_spare(self):
        assert "SPARE" in SAE_METHODS
    
    def test_contains_sre(self):
        assert "SRE" in SAE_METHODS
    
    def test_not_contains_caa(self):
        assert "CAA" not in SAE_METHODS
    
    def test_not_contains_cast(self):
        assert "CAST" not in SAE_METHODS
    
    def test_not_contains_angular(self):
        assert "ANGULAR" not in SAE_METHODS


class TestGetMethodInfo:
    """Tests for method info via list_methods()."""
    
    def test_list_methods_returns_list(self):
        methods = list_methods()
        assert isinstance(methods, list)
    
    def test_list_methods_has_expected_count(self):
        methods = list_methods()
        assert len(methods) >= 12  # grows as methods are added
    
    def test_caa_in_list_methods(self):
        methods = list_methods()
        assert "CAA" in methods
    
    def test_sas_in_list_methods(self):
        methods = list_methods()
        assert "SAS" in methods


class TestCreateEvaluator:
    """Tests for create_evaluator function."""
    
    def test_create_multiple_choice_evaluator(self):
        evaluator = create_evaluator("multiple_choice", "cpu")
        assert evaluator is not None
        assert hasattr(evaluator, 'check')  # evaluator instances have .check()
    
    def test_create_refusal_evaluator(self):
        evaluator = create_evaluator("refusal", "cpu")
        assert evaluator is not None
        assert hasattr(evaluator, 'check')  # evaluator instances have .check()
    
    def test_invalid_evaluator_type_raises(self):
        with pytest.raises((KeyError, ValueError)):
            create_evaluator("nonexistent_evaluator", "cpu")
