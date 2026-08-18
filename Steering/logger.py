"""
Centralized logging configuration for the Steering Vector Benchmark.

This module provides a unified logging system with proper formatting,
log levels, and optional file output.

Usage:
    from Steering.logger import setup_logger
    
    logger = setup_logger(__name__)
    logger.info("Loading model...")
    logger.warning("No HF_TOKEN found")
    logger.error("Failed to load dataset")
"""

import logging
import sys
from pathlib import Path
from typing import Optional


# ANSI color codes for terminal output
class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    LEVEL_COLORS = {
        logging.DEBUG: Colors.GRAY,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.MAGENTA,
    }
    
    def __init__(self, fmt: str, datefmt: str = None, use_colors: bool = True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors
    
    def format(self, record: logging.LogRecord) -> str:
        if self.use_colors and sys.stdout.isatty():
            color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)
            record.levelname = f"{color}{record.levelname}{Colors.RESET}"
            record.name = f"{Colors.CYAN}{record.name}{Colors.RESET}"
        return super().format(record)


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[Path] = None,
    use_colors: bool = True,
) -> logging.Logger:
    """
    Setup structured logger with console and optional file output.
    
    Args:
        name: Logger name (usually __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
        use_colors: Whether to use colored output in terminal
    
    Returns:
        Configured logger instance
    
    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Model loaded successfully")
        2025-01-06 10:58:35 | Steering.pipeline | INFO | Model loaded successfully
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    # Format string
    fmt = "%(asctime)s | %(name)-35s | %(levelname)-8s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    
    formatter = ColoredFormatter(fmt, datefmt, use_colors=use_colors)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Optional file handler (no colors)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(fmt, datefmt)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger





# Module-level logger for this file
_logger = setup_logger("Steering.logger")



