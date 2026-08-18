"""
Custom exceptions for the Steering Vector Benchmark.

This module provides a hierarchy of exceptions for better error handling
and debugging. All exceptions inherit from SteeringError for easy catching.

Exception Hierarchy:
    SteeringError (base)
    ├── ConfigError
    ├── DataLoadError
    ├── ModelError
    ├── ExtractionError
    └── EvaluationError

Usage:
    from Steering.exceptions import ConfigError, DataLoadError
    
    try:
        config = load_config(path)
    except ConfigError as e:
        logger.error(f"Invalid configuration: {e}")
"""


class SteeringError(Exception):
    """
    Base exception for all steering operations.
    
    All custom exceptions in this module inherit from this class,
    making it easy to catch any steering-related error.
    
    Example:
        try:
            # Any steering operation
            pass
        except SteeringError as e:
            logger.error(f"Steering operation failed: {e}")
    """
    pass


class ConfigError(SteeringError):
    """
    Configuration validation or loading error.
    
    Raised when:
    - Config file not found or invalid JSON/YAML
    - Required config fields are missing
    - Config values are invalid (e.g., unknown method, negative layer)
    - Config schema validation fails
    
    Example:
        if method not in VALID_METHODS:
            raise ConfigError(f"Unknown method '{method}'. Valid: {VALID_METHODS}")
    """
    pass


class DataLoadError(SteeringError):
    """
    Dataset loading or parsing error.
    
    Raised when:
    - Dataset file not found
    - Dataset key not in registry
    - Invalid variant specified
    - Data parsing/formatting fails
    - Schema formatter not found
    """
    pass


class ModelError(SteeringError):
    """
    Model loading or inference error.
    
    Raised when:
    - Model download fails
    - Model loading fails (OOM, missing weights)
    - Tokenization fails
    - Forward pass fails
    - Hook registration fails
    
    Example:
        try:
            model = HookedTransformer.from_pretrained(name)
        except Exception as e:
            raise ModelError(f"Failed to load model '{name}': {e}") from e
    """
    pass


class ExtractionError(SteeringError):
    """
    Steering vector extraction error.
    
    Raised when:
    - Activation collection fails
    - Vector computation fails
    - SAE encoding/decoding fails
    - Feature selection fails
    
    Example:
        if len(target_data) == 0:
            raise ExtractionError("No target data provided for extraction")
    """
    pass


class EvaluationError(SteeringError):
    """
    Evaluation pipeline error.
    
    Raised when:
    - Evaluator not found
    - Generation fails
    - Scoring fails
    - Result saving fails
    
    Example:
        if evaluator_type not in EVALUATORS:
            raise EvaluationError(f"Unknown evaluator: {evaluator_type}")
    """
    pass


class SAEError(SteeringError):
    """
    SAE (Sparse Autoencoder) specific error.
    
    Raised when:
    - SAE loading fails
    - SAE not found for model/layer combination
    - SAE encoding/decoding fails
    - Feature dimension mismatch
    
    Example:
        if layer not in available_layers:
            raise SAEError(f"No SAE available for layer {layer}")
    """
    pass


class ValidationError(ConfigError):
    """
    Specific validation error for configs and inputs.
    
    Provides detailed information about what failed validation
    and what the expected values should be.
    
    Attributes:
        field: The field that failed validation
        value: The invalid value
        expected: Description of expected value
    """
    
    def __init__(self, message: str, field: str = None, value=None, expected: str = None):
        super().__init__(message)
        self.field = field
        self.value = value
        self.expected = expected
    
    def __str__(self):
        msg = super().__str__()
        details = []
        if self.field:
            details.append(f"field={self.field}")
        if self.value is not None:
            details.append(f"value={self.value}")
        if self.expected:
            details.append(f"expected={self.expected}")
        if details:
            msg += f" ({', '.join(details)})"
        return msg
