#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Custom exceptions for TCD-SegFormer.

This module defines custom exceptions to provide consistent
error handling throughout the codebase.
"""

from utils import LOGGER_NAME

class TCDSegformerError(Exception):
    """Base exception for all TCD-SegFormer errors."""
    pass


class ConfigurationError(TCDSegformerError):
    """Exception raised for errors in the configuration."""
    pass


class DatasetError(TCDSegformerError):
    """Exception raised for errors in dataset handling."""
    pass


class InvalidSampleError(DatasetError):
    """Exception raised when a dataset sample is invalid."""
    pass


class EmptyMaskError(InvalidSampleError):
    """Exception raised when a mask is empty or has a single class."""
    
    def __init__(self, unique_values=None, sample_idx=None, message=None):
        if message is None:
            message = f"Sample {sample_idx} has only {len(unique_values)} unique values in mask: {unique_values}"
        self.unique_values = unique_values
        self.sample_idx = sample_idx
        super().__init__(message)


class ShapeMismatchError(InvalidSampleError):
    """Exception raised when image and mask shapes don't match."""
    
    def __init__(self, image_shape=None, mask_shape=None, message=None):
        if message is None:
            message = f"Image shape {image_shape} doesn't match mask shape {mask_shape}"
        self.image_shape = image_shape
        self.mask_shape = mask_shape
        super().__init__(message)


class ClassImbalanceError(InvalidSampleError):
    """Exception raised when a mask has severe class imbalance."""
    
    def __init__(self, class_percentages=None, threshold=99.0, sample_idx=None, message=None):
        if message is None:
            message = f"Sample {sample_idx} has severe class imbalance (threshold: {threshold}%): {class_percentages}"
        self.class_percentages = class_percentages
        self.threshold = threshold
        self.sample_idx = sample_idx
        super().__init__(message)


class ModelError(TCDSegformerError):
    """Exception raised for errors in model handling."""
    pass


class WeightLoadingError(ModelError):
    """Exception raised when loading model weights fails."""
    
    def __init__(self, model_name=None, original_error=None, message=None):
        if message is None:
            message = f"Failed to load weights for model {model_name}: {str(original_error)}"
        self.model_name = model_name
        self.original_error = original_error
        super().__init__(message)


class InferenceError(ModelError):
    """Exception raised during model inference."""
    pass


class InvalidInputShapeError(InferenceError):
    """Exception raised when input shape is invalid for the model."""
    
    def __init__(self, input_shape=None, expected_shape=None, message=None):
        if message is None:
            message = f"Invalid input shape {input_shape}, expected {expected_shape}"
        self.input_shape = input_shape
        self.expected_shape = expected_shape
        super().__init__(message)


class TrainingError(TCDSegformerError):
    """Exception raised during model training."""
    pass


class OptimizerError(TrainingError):
    """Exception raised for errors in optimizer configuration."""
    pass


class SchedulerError(TrainingError):
    """Exception raised for errors in scheduler configuration."""
    pass


class GradientError(TrainingError):
    """Exception raised for gradient-related errors during training."""
    
    def __init__(self, gradient_norm=None, message=None):
        if message is None:
            message = f"Gradient norm is abnormal: {gradient_norm}"
        self.gradient_norm = gradient_norm
        super().__init__(message)


class FileError(TCDSegformerError):
    """Exception raised for file-related errors."""
    pass


class FileNotFoundError(FileError):
    """Exception raised when a file is not found."""
    pass


class InvalidFileFormatError(FileError):
    """Exception raised when a file has an invalid format."""
    
    def __init__(self, file_path=None, expected_format=None, message=None):
        if message is None:
            message = f"Invalid file format for {file_path}, expected {expected_format}"
        self.file_path = file_path
        self.expected_format = expected_format
        super().__init__(message)


class HubError(TCDSegformerError):
    """Exception raised for Hugging Face Hub related errors."""
    pass


class APIError(TCDSegformerError):
    """Exception raised for API-related errors."""
    pass


class ExternalLibraryError(TCDSegformerError):
    """Exception raised for errors in external libraries."""
    
    def __init__(self, library_name=None, original_error=None, message=None):
        if message is None:
            message = f"Error in external library {library_name}: {str(original_error)}"
        self.library_name = library_name
        self.original_error = original_error
        super().__init__(message)


def handle_dataset_error(func=None, *, logger=None):
    """
    Decorator to handle dataset-related errors consistently.
    
    Args:
        func: Function to decorate
        logger: Optional logger instance for logging errors. If None, will use
               the standardized logger from utils.get_logger().
        
    Returns:
        Wrapped function with error handling
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get logger if not provided
            nonlocal logger
            if logger is None:
                from utils import get_logger
                try:
                    logger = get_logger()
                except Exception as logger_error:
                    # Fallback to prevent errors if logging setup fails
                    import logging
                    logger = logging.getLogger(LOGGER_NAME)
                    logger.warning(f"Error getting standard logger: {logger_error}. Using fallback.")
            
            try:
                return func(*args, **kwargs)
            except InvalidSampleError as e:
                # Log the error and return a fallback sample
                logger.warning(f"Dataset error (InvalidSample): {str(e)}")
                return create_fallback_sample()
            except DatasetError as e:
                # Log the error and re-raise
                logger.error(f"Dataset error: {str(e)}")
                raise
            except Exception as e:
                # Wrap unknown errors in a DatasetError
                logger.error(f"Unexpected error in dataset processing: {str(e)}", exc_info=True)
                raise DatasetError(f"Unexpected error: {str(e)}") from e
        return wrapper
    
    # Handle case where decorator is used with or without arguments
    if func is not None:
        return decorator(func)
    return decorator


def create_fallback_sample():
    """
    Create a fallback sample for error recovery.
    
    Returns:
        Dictionary with fallback data
    """
    import torch
    import numpy as np
    
    # Create a simple dummy sample with one foreground region
    h, w = 64, 64
    
    # Create a dummy image (black with a white square)
    dummy_image = np.zeros((3, h, w), dtype=np.float32)
    dummy_image[:, 20:40, 20:40] = 1.0
    
    # Create a dummy mask (background with one foreground region)
    dummy_mask = np.zeros((h, w), dtype=np.int64)
    dummy_mask[25:35, 25:35] = 1
    
    return {
        'pixel_values': torch.tensor(dummy_image),
        'labels': torch.tensor(dummy_mask)
    }
