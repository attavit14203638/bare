#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation metrics for TCD-SegFormer model.

This module centralizes the computation of evaluation metrics to ensure
consistent results across different parts of the codebase.
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn import metrics as skmetrics
from scipy.ndimage import binary_dilation, generate_binary_structure

# Import centralized logger name
from utils import LOGGER_NAME, get_logger

# Setup module logger
logger = get_logger()

try:
    from skimage import measure
except ImportError:
    measure = None
    logger.warning("scikit-image not found. Connectivity/shape metrics will not be available. Install with: pip install scikit-image")


from exceptions import ShapeMismatchError, InvalidInputShapeError
from image_utils import resize_mask


def calculate_metrics(
    preds: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    ignore_index: int = 255,
    num_classes: Optional[int] = None,
    metrics_list: Optional[List[str]] = None,
    class_metrics: bool = False
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for semantic segmentation.
    
    Args:
        preds: Predicted segmentation maps (N, H, W) or (H, W)
        labels: Ground truth segmentation maps (N, H, W) or (H, W)
        ignore_index: Index to ignore in evaluation
        num_classes: Number of classes (inferred from data if None)
        metrics_list: List of metrics to calculate (if None, calculates all)
        class_metrics: Whether to return metrics per class
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Convert to numpy arrays if they're torch tensors
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Add batch dimension if needed
    if len(preds.shape) == 2:
        preds = preds[np.newaxis, ...]
    if len(labels.shape) == 2:
        labels = labels[np.newaxis, ...]
    
    # Check if shapes match
    if preds.shape != labels.shape:
        logger.info(f"Predictions shape {preds.shape} doesn't match labels shape {labels.shape}. Resizing predictions.")
        
        # Resize predictions to match labels
        resized_preds = np.zeros_like(labels)
        for i in range(preds.shape[0]):
            resized_preds[i] = resize_mask(preds[i], labels.shape[1:])
        
        preds = resized_preds
    
    # Determine which metrics to calculate
    all_metrics = [
        'pixel_accuracy',
        'iou_class_1', 'dice_class_1', 'precision_class_1', 'recall_class_1', 'f1_score_class_1',
        'boundary_iou_class_1',
        'mean_solidity', 'object_count_diff'
    ]
    if metrics_list is None:
        # Default list now focuses on pixel accuracy and implicitly class 1 metrics
        metrics_list = ['pixel_accuracy'] # Removed mean metrics, F1, boundary_iou from default
    else:
        # Ensure all requested metrics are valid (check against the new all_metrics)
        valid_requested_metrics = []
        for metric in metrics_list:
            if metric in all_metrics:
                valid_requested_metrics.append(metric)
            # Allow requesting old mean metrics, but issue warning? Or just ignore?
            # Let's ignore for now, as they won't be calculated.
            elif metric not in ['mean_iou', 'mean_dice', 'mean_precision', 'mean_recall', 'f1_score', 'boundary_iou']:
                 logger.warning(f"Unknown or unsupported metric '{metric}' requested. Ignoring.")
        metrics_list = valid_requested_metrics

    # Note: Boundary IoU for class 1 will be calculated if class 1 exists, regardless of metrics_list
    # Class metrics flag still controls if metrics for *other* classes are returned.

    # Get valid mask (ignoring ignore_index)
    valid_mask = labels != ignore_index
    
    # Calculate metrics
    result_metrics = {}
    
    # Flatten arrays for global metrics
    preds_valid = preds[valid_mask]
    labels_valid = labels[valid_mask]
    
    # Determine number of classes
    if num_classes is None:
        unique_classes = set(np.unique(labels_valid))
        if ignore_index in unique_classes:
            unique_classes.remove(ignore_index)
        num_classes = len(unique_classes)
    
    # Print for debugging
    logger.debug(f"Unique values in predictions: {np.unique(preds_valid)}")
    logger.debug(f"Unique values in labels: {np.unique(labels_valid)}")
    
    # Calculate pixel accuracy
    if 'pixel_accuracy' in metrics_list:
        pixel_accuracy = np.mean(preds_valid == labels_valid)
        result_metrics['pixel_accuracy'] = pixel_accuracy
    
    # Calculate per-class metrics
    class_iou = []
    class_dice = []
    class_precision = []
    class_recall = []
    tp_class_1, fp_class_1, fn_class_1 = 0, 0, 0 # Store Class 1 TP/FP/FN for F1

    unique_labels = np.unique(labels_valid)

    # Calculate class 1 existence check outside the loop for efficiency
    class_1_exists = (num_classes > 1 and 1 in unique_labels)

    for cls in range(num_classes):
        # Skip if this class doesn't exist in the ground truth
        if cls not in unique_labels and cls != ignore_index:
            if class_metrics:
                result_metrics[f'iou_class_{cls}'] = 0.0
                result_metrics[f'dice_class_{cls}'] = 0.0
                result_metrics[f'precision_class_{cls}'] = 0.0
                result_metrics[f'recall_class_{cls}'] = 0.0
            continue
        
        # Get binary masks for this class
        pred_mask = preds[valid_mask] == cls
        label_mask = labels[valid_mask] == cls
        
        # Calculate intersection and union
        intersection = np.logical_and(pred_mask, label_mask).sum()
        union = np.logical_or(pred_mask, label_mask).sum()
        
        # Calculate true positives, false positives, and false negatives
        tp = intersection
        fp = pred_mask.sum() - tp
        fn = label_mask.sum() - tp
        
        # Calculate IoU
        iou = intersection / union if union > 0 else 0.0

        if class_metrics or (cls == 1 and class_1_exists): # Store if class_metrics=True or if it's class 1
            result_metrics[f'iou_class_{cls}'] = iou

        # Calculate Dice coefficient
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        if class_metrics or (cls == 1 and class_1_exists):
            result_metrics[f'dice_class_{cls}'] = dice

        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if class_metrics or (cls == 1 and class_1_exists):
            result_metrics[f'precision_class_{cls}'] = precision
            result_metrics[f'recall_class_{cls}'] = recall

        # Store TP, FP, FN for class 1 if this is class 1
        if cls == 1 and class_1_exists:
            tp_class_1 = tp
            fp_class_1 = fp
            fn_class_1 = fn

    # Calculate F1 score specifically for class 1 after the loop
    if class_1_exists:
        f1_class_1 = 2 * tp_class_1 / (2 * tp_class_1 + fp_class_1 + fn_class_1) if (2 * tp_class_1 + fp_class_1 + fn_class_1) > 0 else 0.0
        result_metrics['f1_score_class_1'] = f1_class_1

    # Calculate Boundary IoU
    # Calculate class 1 specifically if it exists, remove mean calculation
    if class_1_exists:
        pred_mask_cls1 = preds == 1
        label_mask_cls1 = labels == 1
        biou_cls1 = calculate_boundary_iou(pred_mask_cls1, label_mask_cls1, dilation_ratio=0.02)
        result_metrics['boundary_iou_class_1'] = biou_cls1
    elif num_classes > 1: # If class 1 *could* exist but isn't in this batch
        result_metrics['boundary_iou_class_1'] = 0.0

    # Calculate for other classes only if class_metrics is True
    if class_metrics:
        for cls in range(num_classes):
            if cls == 1: continue # Already calculated
            if cls not in unique_labels and cls != ignore_index:
                 result_metrics[f'boundary_iou_class_{cls}'] = 0.0
                 continue

            pred_mask_cls = preds == cls
            label_mask_cls = labels == cls
            biou = calculate_boundary_iou(pred_mask_cls, label_mask_cls, dilation_ratio=0.02)
            result_metrics[f'boundary_iou_class_{cls}'] = biou

    # Calculate Connectivity/Shape Metrics if skimage is available
    if measure is not None:
        if 'mean_solidity' in metrics_list:
            # Calculate per-class solidity and average
            solidity_values = []
            for cls in range(num_classes):
                 if cls not in unique_labels or cls == 0: continue # Skip background
                 pred_mask_cls = preds == cls # Use prediction mask for solidity
                 # label_mask_cls = labels == cls
                 # Calculate solidity on the predicted mask
                 solidity = calculate_mean_solidity(pred_mask_cls) # Calculate on Prediction
                 solidity_values.append(solidity)
                 if class_metrics: result_metrics[f'solidity_class_{cls}'] = solidity
            result_metrics['mean_solidity'] = np.mean(solidity_values) if solidity_values else 0.0

        if 'object_count_diff' in metrics_list:
            # Calculate per-class object count difference and average absolute difference
            count_diff_values = []
            for cls in range(num_classes):
                 if cls not in unique_labels or cls == 0: continue # Skip background
                 pred_mask_cls = preds == cls
                 label_mask_cls = labels == cls
                 # Calculate difference for this class (pred_count - gt_count)
                 count_diff = calculate_object_count_difference(pred_mask_cls, label_mask_cls)
                 count_diff_values.append(count_diff)
                 if class_metrics: result_metrics[f'object_count_diff_class_{cls}'] = count_diff
            # Report the mean of the absolute differences
            result_metrics['object_count_diff'] = np.mean(np.abs(count_diff_values)) if count_diff_values else 0.0
    elif any(m in metrics_list for m in ['mean_solidity', 'object_count_diff']):
         logger.warning("scikit-image not installed. Skipping shape/connectivity metrics.")


    return result_metrics


def calculate_boundary_iou(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    dilation_ratio: float = 0.02,
    ignore_index: int = 255
) -> float:
    """
    Calculate Boundary IoU metric.

    Args:
        pred_mask: Binary prediction mask (H, W) or (N, H, W) for a single class.
        gt_mask: Binary ground truth mask (H, W) or (N, H, W) for a single class.
        dilation_ratio: Ratio of image diagonal to determine dilation amount.
        ignore_index: Value in gt_mask to ignore (relevant if gt_mask is not binary).

    Returns:
        Boundary IoU score (float).
    """
    # Ensure boolean masks
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    # Handle ignore index in gt_mask if necessary (though typically expects binary)
    if ignore_index is not None:
        valid_gt_mask = gt_mask != ignore_index
        gt_mask = np.logical_and(gt_mask, valid_gt_mask)
        # Apply same validity mask to prediction? Depends on definition.
        # Assuming pred_mask is already only for the target class vs background.

    # Add batch dimension if needed
    if pred_mask.ndim == 2:
        pred_mask = pred_mask[np.newaxis, ...]
    if gt_mask.ndim == 2:
        gt_mask = gt_mask[np.newaxis, ...]

    if pred_mask.shape != gt_mask.shape:
         # Attempt resize if shapes mismatch (e.g., during evaluation)
         logger.info(f"Boundary IoU: pred shape {pred_mask.shape} != gt shape {gt_mask.shape}. Resizing pred.")
         resized_preds = np.zeros_like(gt_mask, dtype=bool)
         for i in range(pred_mask.shape[0]):
             # Use nearest neighbor for boolean masks
             resized_preds[i] = resize_mask(pred_mask[i], gt_mask.shape[1:], order=0).astype(bool)
         pred_mask = resized_preds

    batch_size, h, w = gt_mask.shape
    total_boundary_iou = 0.0
    valid_samples = 0

    # Define connectivity structure for dilation (8-connectivity)
    struct = generate_binary_structure(2, 2)

    # Calculate dilation amount based on image diagonal
    dilation_pixels = int(max(1, np.sqrt(h**2 + w**2) * dilation_ratio))

    for i in range(batch_size):
        gt = gt_mask[i]
        pred = pred_mask[i]

        # Dilate masks
        gt_dilated = binary_dilation(gt, structure=struct, iterations=dilation_pixels)
        pred_dilated = binary_dilation(pred, structure=struct, iterations=dilation_pixels)

        # Get boundary regions
        gt_boundary = gt_dilated & (~gt)
        pred_boundary = pred_dilated & (~pred)

        # Calculate intersection and union of boundaries
        intersection = np.sum(gt_boundary & pred_boundary)
        union = np.sum(gt_boundary | pred_boundary)

        # Calculate Boundary IoU for this sample
        if union == 0:
            # If both boundaries are empty (e.g., empty masks), IoU is 1 if masks match, 0 otherwise
            boundary_iou = 1.0 if np.all(gt == pred) else 0.0
        else:
            boundary_iou = intersection / union

        total_boundary_iou += boundary_iou
        valid_samples += 1

    # Return average Boundary IoU across the batch
    return total_boundary_iou / valid_samples if valid_samples > 0 else 0.0


# --- Shape/Connectivity Metrics ---

def calculate_mean_solidity(pred_mask: np.ndarray) -> float:
    """
    Calculate the mean solidity of predicted objects in a binary mask.
    Requires scikit-image.

    Args:
        pred_mask: Binary prediction mask (N, H, W) or (H, W).

    Returns:
        Mean solidity of predicted objects, or 0.0 if no objects or skimage not installed.
    """
    if measure is None: return 0.0
    if pred_mask.ndim == 2: pred_mask = pred_mask[np.newaxis, ...]

    total_solidity = 0.0
    object_count = 0
    for i in range(pred_mask.shape[0]):
        # Label connected components in the prediction mask
        labeled_pred_mask, num_pred_labels = measure.label(pred_mask[i], connectivity=2, return_num=True)
        if num_pred_labels > 0:
            props = measure.regionprops(labeled_pred_mask)
            for prop in props:
                # Add solidity if area is > 0 (should always be true for labeled regions)
                if prop.area > 0:
                    total_solidity += prop.solidity
                    object_count += 1 # Count each valid object

    # Return mean solidity across all predicted objects in the batch
    return total_solidity / object_count if object_count > 0 else 0.0


def calculate_object_count_difference(pred_mask: np.ndarray, gt_mask: np.ndarray) -> int:
    """
    Calculate the difference in the number of connected objects between prediction and ground truth.
    Requires scikit-image.

    Args:
        pred_mask: Binary prediction mask (N, H, W) or (H, W).
        gt_mask: Binary ground truth mask (N, H, W) or (H, W).

    Returns:
        Difference in object count (pred_count - gt_count), or 0 if skimage not installed.
    """
    if measure is None: return 0
    if pred_mask.ndim == 2: pred_mask = pred_mask[np.newaxis, ...]
    if gt_mask.ndim == 2: gt_mask = gt_mask[np.newaxis, ...]

    pred_count_total = 0
    gt_count_total = 0
    for i in range(pred_mask.shape[0]):
        _, pred_num = measure.label(pred_mask[i], connectivity=2, return_num=True)
        _, gt_num = measure.label(gt_mask[i], connectivity=2, return_num=True)
        pred_count_total += pred_num
        gt_count_total += gt_num

    return pred_count_total - gt_count_total

# Define error categories for multi-class segmentation
# Base categories for binary and multi-class compatibility
BASE_ERROR_CATEGORIES = {
    0: "Correct Background",    # Predicted BG, True BG (TN)
    1: "Correct Class",         # Predicted Class X, True Class X (TP) - includes all correct foreground classes
    2: "False Positive",        # Predicted FG, True BG (FP)
    3: "False Negative",        # Predicted BG, True FG (FN)
    4: "Misclassification",     # Predicted Class X, True Class Y (where X,Y > 0 and X≠Y)
}

# This will be populated dynamically in categorize_errors based on num_classes
# Basic error categories for compatibility with existing code
ERROR_CATEGORIES = BASE_ERROR_CATEGORIES.copy()

def categorize_errors(
    preds: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
    detailed_mode: bool = False
) -> np.ndarray:
    """
    Categorizes prediction errors pixel-wise for multi-class segmentation.

    Args:
        preds: Predicted segmentation maps (N, H, W) or (H, W)
        labels: Ground truth segmentation maps (N, H, W) or (H, W)
        num_classes: Number of classes
        ignore_index: Index to ignore in labels
        detailed_mode: If True, generates detailed confusion codes for each class pair.
                      If False, uses the basic categorization scheme (compatible with binary).

    Returns:
        An error map (N, H, W) or (H, W) with integer codes representing error categories.
        Pixels with ignore_index in labels will have value `ignore_index`.
    """
    # Ensure numpy arrays
    if isinstance(preds, torch.Tensor): preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
    
    # Validate inputs
    if num_classes < 2:
        logger.warning(f"Invalid num_classes: {num_classes}. Must be at least 2. Forcing num_classes=2.")
        num_classes = 2

    # Setup error categories based on number of classes and mode
    global ERROR_CATEGORIES
    ERROR_CATEGORIES = BASE_ERROR_CATEGORIES.copy()
    
    # In detailed mode, add specific misclassification codes for each class pair
    if detailed_mode and num_classes > 2:
        # Start custom error codes after the base ones
        next_code = max(BASE_ERROR_CATEGORIES.keys()) + 1
        
        for true_cls in range(num_classes):
            for pred_cls in range(num_classes):
                if true_cls != pred_cls:  # Only for misclassifications
                    # Format: "Class X -> Class Y"
                    # Background class (0) is handled separately in base categories 2 & 3
                    if true_cls == 0 or pred_cls == 0:
                        continue
                        
                    # Create a new error code and category for this class pair
                    code = next_code
                    ERROR_CATEGORIES[code] = f"Class {true_cls} -> Class {pred_cls}"
                    next_code += 1
    
    # Create error map initialized with ignore_index
    error_map = np.full_like(labels, ignore_index, dtype=int)

    # Create mask for valid (non-ignored) pixels
    valid_mask = labels != ignore_index

    # --- Apply categorization logic within the valid mask ---

    # 1. Correct Predictions
    correct_mask = (preds == labels) & valid_mask
    # Correct Background (TN)
    error_map[correct_mask & (labels == 0)] = 0
    # Correct Foreground for each class (TP)
    error_map[correct_mask & (labels > 0)] = 1

    # 2. Incorrect Predictions
    incorrect_mask = (preds != labels) & valid_mask
    
    # False Positive (Predicted FG, True BG)
    error_map[incorrect_mask & (preds > 0) & (labels == 0)] = 2
    
    # False Negative (Predicted BG, True FG)
    error_map[incorrect_mask & (preds == 0) & (labels > 0)] = 3
    
    # Confusion between foreground classes
    fg_confusion_mask = incorrect_mask & (preds > 0) & (labels > 0)
    
    if detailed_mode and num_classes > 2:
        # Detailed multi-class confusion - assign specific error codes for each class pair
        for true_cls in range(1, num_classes):  # Skip background (0)
            for pred_cls in range(1, num_classes):  # Skip background (0)
                if true_cls != pred_cls:
                    # Find the error code for this class pair
                    pair_key = None
                    for code, desc in ERROR_CATEGORIES.items():
                        if desc == f"Class {true_cls} -> Class {pred_cls}":
                            pair_key = code
                            break
                    
                    if pair_key is not None:
                        # Apply the specific error code for this confusion
                        specific_mask = fg_confusion_mask & (labels == true_cls) & (preds == pred_cls)
                        error_map[specific_mask] = pair_key
    else:
        # Simple mode - all foreground confusions use the same code
        error_map[fg_confusion_mask] = 4

    return error_map


def calculate_error_statistics(error_map: np.ndarray, ignore_index: int = 255) -> Dict[str, float]:
    """
    Calculates the percentage of pixels belonging to each error category.

    Args:
        error_map: The error map generated by `categorize_errors`.
        ignore_index: Index to ignore in the error map.

    Returns:
        Dictionary mapping error category names to their percentage.
    """
    stats = {}
    valid_pixels = error_map != ignore_index
    total_valid_pixels = np.sum(valid_pixels)

    if total_valid_pixels == 0:
        return {f"error_perc_{name.lower().replace(' ', '_').replace('->', 'to')}": 0.0 
                for name in ERROR_CATEGORIES.values()}

    for code, name in ERROR_CATEGORIES.items():
        count = np.sum(error_map[valid_pixels] == code)
        # Convert names like "Class 1 -> Class 2" to "class_1_to_class_2" for keys
        key = f"error_perc_{name.lower().replace(' ', '_').replace('->', 'to')}"
        stats[key] = (count / total_valid_pixels) * 100

    # Calculate aggregate statistics for easier interpretation
    if len(ERROR_CATEGORIES) > len(BASE_ERROR_CATEGORIES):
        # Add a summary of all misclassifications if we're in detailed mode
        total_misclass = sum(stats.get(f"error_perc_{name.lower().replace(' ', '_').replace('->', 'to')}", 0)
                           for code, name in ERROR_CATEGORIES.items() 
                           if code > 4)  # Codes above 4 are detailed misclassifications
        stats["error_perc_all_misclassifications"] = total_misclass

    return stats


# --- End Error Analysis ---


def calculate_confusion_matrix(
    preds: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    num_classes: int,
    ignore_index: int = 255,
    normalize: bool = False
) -> np.ndarray:
    """
    Calculate confusion matrix for semantic segmentation.
    
    Args:
        preds: Predicted segmentation maps (N, H, W) or (H, W)
        labels: Ground truth segmentation maps (N, H, W) or (H, W)
        num_classes: Number of classes
        ignore_index: Index to ignore in evaluation
        normalize: Whether to normalize confusion matrix by row (true label)
        
    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    # Convert to numpy arrays if they're torch tensors
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Flatten arrays
    if len(preds.shape) > 1:
        preds = preds.reshape(-1)
    if len(labels.shape) > 1:
        labels = labels.reshape(-1)
    
    # Create mask for valid pixels
    mask = labels != ignore_index
    preds = preds[mask]
    labels = labels[mask]
    
    # Calculate confusion matrix
    confusion_matrix = skmetrics.confusion_matrix(
        labels, preds, labels=range(num_classes)
    )
    
    # Normalize if requested
    if normalize:
        confusion_matrix = confusion_matrix.astype(np.float32)
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        confusion_matrix = np.divide(
            confusion_matrix, row_sums,
            out=np.zeros_like(confusion_matrix, dtype=np.float32),
            where=row_sums != 0
        )
    
    return confusion_matrix


def calculate_per_image_metrics(
    preds: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    ignore_index: int = 255,
    metrics_list: Optional[List[str]] = None
) -> List[Dict[str, float]]:
    """
    Calculate metrics for each image in the batch.
    
    Args:
        preds: Predicted segmentation maps (N, H, W)
        labels: Ground truth segmentation maps (N, H, W)
        ignore_index: Index to ignore in evaluation
        metrics_list: List of metrics to calculate
        
    Returns:
        List of metric dictionaries for each image
    """
    # Convert to numpy arrays if they're torch tensors
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Ensure inputs have batch dimension
    if len(preds.shape) == 2:
        preds = preds[np.newaxis, ...]
    if len(labels.shape) == 2:
        labels = labels[np.newaxis, ...]
    
    # Check if batch sizes match
    if preds.shape[0] != labels.shape[0]:
        raise ShapeMismatchError(preds.shape, labels.shape, 
                                "Batch sizes don't match for per-image metrics")
    
    # Calculate metrics for each image
    batch_size = preds.shape[0]
    per_image_metrics = []
    
    for i in range(batch_size):
        metrics = calculate_metrics(
            preds[i], labels[i],
            ignore_index=ignore_index,
            metrics_list=metrics_list
        )
        per_image_metrics.append(metrics)
    
    return per_image_metrics


def find_worst_samples(
    preds: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    images: Optional[Union[np.ndarray, torch.Tensor]] = None,
    metric: str = 'iou_class_1', # Changed default metric
    num_samples: int = 5,
    ignore_index: int = 255
) -> List[Tuple[float, int, Union[np.ndarray, None], np.ndarray, np.ndarray]]:
    """
    Find the worst performing samples based on a specific metric.
    
    Args:
        preds: Predicted segmentation maps (N, H, W)
        labels: Ground truth segmentation maps (N, H, W)
        images: Original images (N, C, H, W) or (N, H, W, C) (optional)
        metric: Metric to use for ranking ('mean_iou', 'mean_dice', etc.)
        num_samples: Number of worst samples to return
        ignore_index: Index to ignore in evaluation
        
    Returns:
        List of tuples (metric_value, sample_index, image, pred, label) for the worst samples
    """
    # Convert to numpy arrays if they're torch tensors
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if images is not None and isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    
    # Ensure inputs have batch dimension
    if len(preds.shape) == 2:
        preds = preds[np.newaxis, ...]
    if len(labels.shape) == 2:
        labels = labels[np.newaxis, ...]
    
    # Calculate metrics for each sample
    # Ensure the requested metric (now defaulting to iou_class_1) is calculated
    per_sample_metrics = calculate_per_image_metrics(
        preds, labels, ignore_index=ignore_index, metrics_list=[metric], class_metrics=True # Force class_metrics=True if needed for class_1 metrics
    )

    # Get metric values
    metric_values = []
    for m in per_sample_metrics:
        if metric in m:
            metric_values.append(m[metric])
        else:
            # Handle case where metric (e.g., iou_class_1) wasn't calculated for a sample
            # (e.g., class 1 not present in that specific label)
            # Assign a 'good' value (like 1.0 for IoU) so it's not ranked worst,
            # or handle as needed. For simplicity, let's use a placeholder like -1.0,
            # assuming normal metrics >= 0.
            # Using a large negative number for sorting might be problematic if the metric can be negative.
            # Let's use a placeholder like -1.0, assuming normal metrics >= 0.
            logger.warning(f"Metric '{metric}' not found for a sample in find_worst_samples. Assigning placeholder value -1.0.")
            metric_values.append(-1.0)

    # Find indices of worst samples (lower score is worse)
    worst_indices = np.argsort(metric_values)[:num_samples]
    
    # Create result list
    worst_samples = []
    for idx in worst_indices:
        if images is not None:
            sample_image = images[idx]
        else:
            sample_image = None
        
        worst_samples.append((
            metric_values[idx],
            idx,
            sample_image,
            preds[idx],
            labels[idx]
        ))
    
    return worst_samples


def metric_names_to_pretty(metric_name: str) -> str:
    """
    Convert metric name to pretty format for display.
    
    Args:
        metric_name: Original metric name
        
    Returns:
        Pretty formatted metric name
    """
    # Define metric name mapping
    name_map = {
        'pixel_accuracy': 'Overall Accuracy',
        'iou_class_1': 'Tree IoU',
        'dice_class_1': 'Tree Dice',
        'precision_class_1': 'Tree Precision',
        'recall_class_1': 'Tree Recall',
        'f1_score_class_1': 'Tree F1 Score',
        'boundary_iou_class_1': 'Tree Boundary IoU',
        'loss': 'Loss',
        'mean_solidity': 'Mean Solidity',
        'object_count_diff': 'Object Count Diff'
    }

    # Handle other per-class metrics if class_metrics=True
    if metric_name.startswith('iou_class_') and metric_name not in name_map:
        class_num = metric_name.split('_')[-1]
        return f'Class {class_num} IoU'
    if metric_name.startswith('dice_class_') and metric_name not in name_map:
        class_num = metric_name.split('_')[-1]
        return f'Class {class_num} Dice'
    if metric_name.startswith('precision_class_') and metric_name not in name_map:
        class_num = metric_name.split('_')[-1]
        return f'Class {class_num} Precision'
    if metric_name.startswith('recall_class_') and metric_name not in name_map:
        class_num = metric_name.split('_')[-1]
        return f'Class {class_num} Recall'
    # Note: No generic per-class F1 score implemented, only class 1
    if metric_name.startswith('boundary_iou_class_') and metric_name not in name_map:
        class_num = metric_name.split('_')[-1]
        return f'Class {class_num} Boundary IoU'
    if metric_name.startswith('solidity_class_') and metric_name not in name_map:
         class_num = metric_name.split('_')[-1]
         return f'Class {class_num} Solidity'

    # Return pretty name if in mapping, otherwise capitalize words
    if metric_name in name_map:
        return name_map[metric_name]
    else:
        return ' '.join(word.capitalize() for word in metric_name.split('_'))
