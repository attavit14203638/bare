#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for TCD-SegFormer model.

This module provides common utility functions used throughout the codebase,
including random seed setting, logging setup, and parameter counting.
"""

import os
import json
import random
import numpy as np
import torch
import logging
from typing import Dict, Any, Optional, Union, Callable

# Standardized logger name used throughout the codebase
LOGGER_NAME = "BARE"

# Configure the root logger at module import time
def _configure_root_logger():
    """
    Configure the root logger with console handler.
    This ensures that there's always at least a console handler
    even if setup_logging() isn't called explicitly.
    """
    root_logger = logging.getLogger(LOGGER_NAME)
    
    # Only configure if handlers aren't already set up
    if not root_logger.handlers:
        root_logger.setLevel(logging.INFO)
        
        # Create console handler with formatter
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        root_logger.addHandler(console_handler)
        
        # Prevent propagation to the root logger to avoid duplicate logs
        root_logger.propagate = False

# Initialize logger when module is imported
_configure_root_logger()

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(output_dir: str, log_level: int = logging.INFO, 
                 file_log_level: Optional[int] = None,
                 filename: str = "training.log",
                 formatter: Optional[logging.Formatter] = None) -> logging.Logger:
    """
    Set up logging with both console and file handlers. 
    This function manages handlers properly to avoid duplicates.
    
    Args:
        output_dir: Directory to save log file
        log_level: Logging level for console handler
        file_log_level: Logging level for file handler (defaults to log_level if None)
        filename: Name of the log file
        formatter: Custom formatter (uses default if None)
        
    Returns:
        Logger
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the logger
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(min(log_level, file_log_level or log_level))
    
    # Use default formatter if none provided
    if formatter is None:
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Clear existing handlers if any exist (to avoid duplicates)
    # Save file handlers path to prevent recreating the same file handler
    existing_file_paths = []
    handlers_to_remove = []
    for handler in logger.handlers:
        # Mark console handlers for removal
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handlers_to_remove.append(handler)
        # Keep track of file paths for file handlers
        elif isinstance(handler, logging.FileHandler):
            existing_file_paths.append(handler.baseFilename)
            # If we're creating a file handler with the same path, mark for removal
            if os.path.join(output_dir, filename) == handler.baseFilename:
                handlers_to_remove.append(handler)
    
    # Remove marked handlers
    for handler in handlers_to_remove:
        logger.removeHandler(handler)
    
    # Create and add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file path
    log_file_path = os.path.join(output_dir, filename)
    
    # Create and add file handler if not already existing
    if log_file_path not in existing_file_paths:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(file_log_level if file_log_level is not None else log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Ensure propagation is disabled to avoid duplicate logs
    logger.propagate = False
    
    return logger

def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """
    Get the current learning rate from the optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    
    return 0.0


def get_logger() -> logging.Logger:
    """
    Get the standardized logger used throughout the codebase.
    This is useful when a function needs a logger but doesn't have one passed as a parameter.
    
    Returns:
        The standardized TCD-SegFormer logger
    """
    return logging.getLogger(LOGGER_NAME)


def log_or_print(
    message: str, 
    logger: Optional[logging.Logger] = None, 
    level: int = logging.INFO, 
    is_notebook: bool = False
) -> None:
    """
    Logs a message using the provided logger or falls back to the standardized logger.
    Only falls back to print statements in notebook environments.
    
    This function centralizes logging across the codebase.

    Args:
        message: The message to log or print.
        logger: An optional logging.Logger instance.
        level: The logging level to use (e.g., logging.INFO, logging.WARNING).
        is_notebook: Flag indicating if running in a notebook environment.
    """
    # If no logger is provided, use the standardized logger
    if logger is None and not is_notebook:
        logger = get_logger()
    
    if logger:
        # Use the appropriate logging level
        if level == logging.INFO:
            logger.info(message)
        elif level == logging.WARNING:
            logger.warning(message)
        elif level == logging.ERROR:
            logger.error(message)
        elif level == logging.DEBUG:
            logger.debug(message)
        else:
            # Fallback to generic log() method for custom levels
            logger.log(level, message)
    elif is_notebook:
        # For notebooks, include level name for warnings and errors
        if level >= logging.WARNING:
            print(f"{logging.getLevelName(level)}: {message}")
        else:
            print(message)
