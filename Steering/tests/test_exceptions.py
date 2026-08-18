"""
Unit tests for the exception system.

Tests:
- Exception hierarchy
- Exception messages
- Exception inheritance
"""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from Steering.exceptions import (
    SteeringError,
    ConfigError,
    DataLoadError,
    ModelError,
    ExtractionError,
    EvaluationError,
)


class TestExceptionHierarchy:
    """Tests for exception inheritance hierarchy."""
    
    def test_steering_error_is_exception(self):
        """Test SteeringError inherits from Exception."""
        assert issubclass(SteeringError, Exception)
    
    def test_config_error_inherits_steering_error(self):
        """Test ConfigError inherits from SteeringError."""
        assert issubclass(ConfigError, SteeringError)
    
    def test_data_load_error_inherits_steering_error(self):
        """Test DataLoadError inherits from SteeringError."""
        assert issubclass(DataLoadError, SteeringError)
    
    def test_model_error_inherits_steering_error(self):
        """Test ModelError inherits from SteeringError."""
        assert issubclass(ModelError, SteeringError)
    
    def test_extraction_error_inherits_steering_error(self):
        """Test ExtractionError inherits from SteeringError."""
        assert issubclass(ExtractionError, SteeringError)
    
    def test_evaluation_error_inherits_steering_error(self):
        """Test EvaluationError inherits from SteeringError."""
        assert issubclass(EvaluationError, SteeringError)


class TestExceptionInstantiation:
    """Tests for exception instantiation and messages."""
    
    def test_steering_error_with_message(self):
        """Test SteeringError can be raised with message."""
        with pytest.raises(SteeringError) as exc_info:
            raise SteeringError("Test steering error message")
        
        assert "Test steering error message" in str(exc_info.value)
    
    def test_config_error_with_message(self):
        """Test ConfigError can be raised with message."""
        with pytest.raises(ConfigError) as exc_info:
            raise ConfigError("Invalid configuration")
        
        assert "Invalid configuration" in str(exc_info.value)
    
    def test_data_load_error_with_message(self):
        """Test DataLoadError can be raised with message."""
        with pytest.raises(DataLoadError) as exc_info:
            raise DataLoadError("Failed to load dataset")
        
        assert "Failed to load dataset" in str(exc_info.value)
    
    def test_model_error_with_message(self):
        """Test ModelError can be raised with message."""
        with pytest.raises(ModelError) as exc_info:
            raise ModelError("Failed to load model")
        
        assert "Failed to load model" in str(exc_info.value)
    
    def test_extraction_error_with_message(self):
        """Test ExtractionError can be raised with message."""
        with pytest.raises(ExtractionError) as exc_info:
            raise ExtractionError("Extraction failed")
        
        assert "Extraction failed" in str(exc_info.value)
    
    def test_evaluation_error_with_message(self):
        """Test EvaluationError can be raised with message."""
        with pytest.raises(EvaluationError) as exc_info:
            raise EvaluationError("Evaluation failed")
        
        assert "Evaluation failed" in str(exc_info.value)


class TestExceptionCatching:
    """Tests for catching exceptions at different levels."""
    
    def test_catch_config_error_as_steering_error(self):
        """Test ConfigError can be caught as SteeringError."""
        caught = False
        try:
            raise ConfigError("Config error")
        except SteeringError:
            caught = True
        
        assert caught
    
    def test_catch_data_load_error_as_steering_error(self):
        """Test DataLoadError can be caught as SteeringError."""
        caught = False
        try:
            raise DataLoadError("Data error")
        except SteeringError:
            caught = True
        
        assert caught
    
    def test_catch_model_error_as_steering_error(self):
        """Test ModelError can be caught as SteeringError."""
        caught = False
        try:
            raise ModelError("Model error")
        except SteeringError:
            caught = True
        
        assert caught
    
    def test_catch_all_steering_errors(self):
        """Test all custom errors can be caught with SteeringError."""
        exceptions = [
            ConfigError("config"),
            DataLoadError("data"),
            ModelError("model"),
            ExtractionError("extraction"),
            EvaluationError("evaluation"),
        ]
        
        for exc in exceptions:
            caught = False
            try:
                raise exc
            except SteeringError:
                caught = True
            
            assert caught, f"Failed to catch {type(exc).__name__} as SteeringError"
