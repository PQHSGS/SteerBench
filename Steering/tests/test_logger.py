"""
Unit tests for the logging system.

Tests:
- Logger setup
- Log levels
- Colored output
"""

import pytest
import logging
from pathlib import Path
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from Steering.logger import setup_logger


class TestSetupLogger:
    """Tests for setup_logger function."""
    
    def test_returns_logger(self):
        """Test setup_logger returns a Logger instance."""
        logger = setup_logger("test_logger")
        assert isinstance(logger, logging.Logger)
    
    def test_logger_name(self):
        """Test logger has correct name."""
        logger = setup_logger("my_test_name")
        assert logger.name == "my_test_name"
    
    def test_default_level_info(self):
        """Test default log level is INFO."""
        logger = setup_logger("test_info_level")
        assert logger.level == logging.INFO
    
    def test_custom_level_debug(self):
        """Test setting DEBUG level."""
        logger = setup_logger("test_debug_level", level="DEBUG")
        assert logger.level == logging.DEBUG
    
    def test_custom_level_warning(self):
        """Test setting WARNING level."""
        logger = setup_logger("test_warning_level", level="WARNING")
        assert logger.level == logging.WARNING
    
    def test_custom_level_error(self):
        """Test setting ERROR level."""
        logger = setup_logger("test_error_level", level="ERROR")
        assert logger.level == logging.ERROR
    
    def test_logger_has_handler(self):
        """Test logger has at least one handler."""
        logger = setup_logger("test_handler")
        assert len(logger.handlers) > 0
    
    def test_can_log_info(self, capsys):
        """Test logger can log INFO messages."""
        logger = setup_logger("test_log_info")
        logger.info("Test info message")
        # Just checking no exception is raised
    
    def test_can_log_warning(self, capsys):
        """Test logger can log WARNING messages."""
        logger = setup_logger("test_log_warning")
        logger.warning("Test warning message")
        # Just checking no exception is raised
    
    def test_can_log_error(self, capsys):
        """Test logger can log ERROR messages."""
        logger = setup_logger("test_log_error")
        logger.error("Test error message")
        # Just checking no exception is raised
    
    def test_file_logging(self, tmp_path):
        """Test logger can write to file."""
        log_file = tmp_path / "test.log"
        logger = setup_logger("test_file_log", log_file=log_file)
        logger.info("Test file logging message")
        
        # Check file was created and has content
        assert log_file.exists()
        content = log_file.read_text()
        assert "Test file logging message" in content
