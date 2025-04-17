"""Logging utilities for the pipeline."""

import logging
import os
import sys
from datetime import datetime

def setup_logger(name, log_file=None, level=logging.INFO):
    """Set up and return a logger with the given name and configuration.
    
    Args:
        name: Name of the logger
        log_file: Path to the log file. If None, only console output is used.
        level: Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory if it doesn't exist
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Default logger for the pipeline
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
default_log_path = os.path.join("logs", f"pipeline_{timestamp}.log")
logger = setup_logger("dubbing_pipeline", default_log_path) 