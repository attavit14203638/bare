#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Class weight computation utilities for TCD-SegFormer model.

This module provides a robust function for computing class weights based on
actual pixel distribution across the entire dataset, ensuring accurate handling
of class imbalance in segmentation tasks.
"""

import numpy as np
import torch
import logging
from typing import Tuple, Optional
from datasets import Dataset

from image_utils import ensure_mask_shape
from utils import get_logger

# Setup module logger
logger = get_logger()


def compute_class_pixel_distribution(dataset, num_labels, mask_feature="annotation", binary=True, logger=None):
    """
    Compute class weights and pixel distribution over the entire dataset (all samples).
    Args:
        dataset: HuggingFace Dataset object (e.g., dataset_dict["train"])
        num_labels: Number of classes (e.g., 2 for background/trees)
        mask_feature: Name of the mask field in the dataset
        binary: Whether to treat as binary segmentation (0/1)
        logger: Logger for output (optional)
    Returns:
        class_weights: torch.Tensor of shape (num_labels,)
        class_counts: np.ndarray of pixel counts per class
    """
    if logger is None:
        logger = get_logger()

    class_counts = np.zeros(num_labels, dtype=np.int64)
    logger.info(f"Counting pixels for {len(dataset)} samples to compute class weights...")
    for i in range(len(dataset)):
        mask = np.array(dataset[i][mask_feature])
        expected_shape = (mask.shape[0], mask.shape[1]) if len(mask.shape) >= 2 else (1, 1)
        mask = ensure_mask_shape(mask, expected_shape, binary=binary)
        if len(mask.shape) > 2:
            logger.warning(f"Mask shape {mask.shape} still has > 2 dimensions after processing (sample {i})")
        for class_idx in range(num_labels):
            class_counts[class_idx] += np.sum(mask == class_idx)
    total_pixels = np.sum(class_counts)
    logger.info(f"Total pixels: {total_pixels}")
    for class_idx in range(num_labels):
        percentage = (class_counts[class_idx] / total_pixels) * 100 if total_pixels > 0 else 0.0
        logger.info(f"Class {class_idx}: {class_counts[class_idx]} pixels ({percentage:.2f}%)")
    # Compute weights (inverse frequency)
    if np.any(class_counts == 0):
        logger.warning(f"Some classes have zero pixels: {class_counts}. Using uniform class weights.")
        weights = np.ones(num_labels)
    else:
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * num_labels
    class_weights = torch.tensor(weights, dtype=torch.float32)
    logger.info(f"Computed class weights: {class_weights.tolist()}")
    return class_weights, class_counts
