#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization utilities for TCD-SegFormer model.

This module provides functions for visualizing segmentation results,
training progress, model confidence, class activation maps, and other aspects
of the model, centralizing functionality that was previously scattered
across multiple files.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import io # Added for in-memory buffer
import matplotlib.gridspec as gridspec
from typing import Dict, Tuple, List, Optional, Union, Any, Callable
from PIL import Image
import logging
from torch.nn import functional as F
import cv2 # Needed for CAM visualization overlay
import seaborn as sns # Added for confusion matrix heatmap
import copy # Import copy module
from scipy.ndimage import binary_dilation, generate_binary_structure # Added for boundary visualization
from utils import LOGGER_NAME, get_logger
from torch.optim import Optimizer # Added for LR schedule plot
from torch.optim.lr_scheduler import _LRScheduler # Added for LR schedule plot

# Set up module logger
logger = get_logger()

# Grad-CAM related imports
try:
    from pytorch_grad_cam import GradCAM # Using GradCAM as the base
    from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRAD_CAM_AVAILABLE = True
except ImportError:
    GRAD_CAM_AVAILABLE = False
    logger.warning("pytorch-grad-cam not installed. CAM/Grad-CAM functionality will not be available. Install with: pip install grad-cam")

# Moved from image_utils.py
def create_colormap(num_classes: int) -> ListedColormap:
    """
    Create colormap for visualization.

    Args:
        num_classes: Number of classes

    Returns:
        Matplotlib colormap
    """
    if num_classes <= 10:
        # Use qualitative colormap for small number of classes
        cmap = plt.cm.get_cmap("tab10", num_classes)
    elif num_classes <= 20:
         # Use larger colormap for more classes
        cmap = plt.cm.get_cmap("tab20", num_classes)
    else:
        # Fallback for very large number of classes
        cmap = plt.cm.get_cmap("viridis", num_classes)

    return cmap

# Moved from image_utils.py
def tensor_to_image(
    tensor: torch.Tensor,
    denormalize: bool = True,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> np.ndarray:
    """
    Convert a PyTorch tensor to a numpy image.

    Args:
        tensor: Input tensor (C, H, W) or (B, C, H, W)
        denormalize: Whether to denormalize the tensor
        mean: Mean values for denormalization (ImageNet default if None)
        std: Standard deviation values for denormalization (ImageNet default if None)

    Returns:
        Numpy image (H, W, C) in uint8 format [0, 255]
    """
    # Default ImageNet normalization parameters
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    # Make a copy to avoid modifying the original tensor
    tensor = tensor.clone().detach().cpu()

    # Remove batch dimension if present
    if len(tensor.shape) == 4:
        tensor = tensor[0]

    # Get tensor as numpy array
    image = tensor.permute(1, 2, 0).numpy()

    # Denormalize if requested
    if denormalize:
        mean_np = np.array(mean).reshape(1, 1, 3)
        std_np = np.array(std).reshape(1, 1, 3)
        image = image * std_np + mean_np

    # Clip values to [0, 1]
    image = np.clip(image, 0, 1)

    # Convert to uint8 [0, 255]
    image = (image * 255).astype(np.uint8)

    return image

# Moved from image_utils.py
def visualize_segmentation(
    image: np.ndarray,
    segmentation_map: np.ndarray,
    id2label: Optional[Dict[int, str]] = None,
    alpha: float = 0.4,
    colormap: Optional[Union[str, ListedColormap]] = None
) -> np.ndarray:
    """
    Visualize segmentation map overlaid on the image.

    Args:
        image: Input image (H, W, 3)
        segmentation_map: Segmentation map (H, W)
        id2label: Mapping from class indices to class names
        alpha: Transparency of the segmentation map
        colormap: Optional custom colormap

    Returns:
        Visualization image (H, W, 3)
    """
    # DIAGNOSTIC: Log input dimensions
    logger.warning(f"visualize_segmentation: image shape: {image.shape}, segmentation_map shape: {segmentation_map.shape}")
    
    # --- REVISED MODIFICATION STARTS HERE ---
    # Pre-process segmentation_map if it's 3D but should be 2D (H,W)
    if segmentation_map.ndim == 3:
        logger.warning(f"Segmentation map received with 3 dimensions: {segmentation_map.shape}. Attempting to convert to 2D (H,W).")
        num_channels = segmentation_map.shape[-1]

        if num_channels == 1:
            logger.info(f"Segmentation map shape {segmentation_map.shape} is (H,W,1). Squeezing the last dimension.")
            segmentation_map = segmentation_map.squeeze(axis=-1)
            logger.info(f"Squeezed segmentation_map to shape: {segmentation_map.shape}")
        elif num_channels > 1:
            # Assuming one-hot encoded if multiple channels (e.g., H, W, C)
            # This was confirmed by the user for the (H,W,3) case.
            logger.info(
                f"Segmentation map shape {segmentation_map.shape} has {num_channels} channels. "
                "Assuming it's one-hot encoded. Converting to 2D label map using argmax along the last axis."
            )
            segmentation_map = np.argmax(segmentation_map, axis=-1)
            logger.info(f"Converted one-hot segmentation_map to 2D label map with shape: {segmentation_map.shape}")
        else: # num_channels == 0 or other unexpected cases
            logger.error(
                f"Segmentation map is 3D with shape {segmentation_map.shape}, but has an unexpected number of channels ({num_channels}). "
                "Cannot automatically convert to 2D. Errors may follow."
            )
    # --- REVISED MODIFICATION ENDS HERE ---

    # Ensure image is RGB with values in [0, 255]
    if image.dtype != np.uint8:
        # Check if image is float [0, 1]
        if image.max() <= 1.0 and image.min() >= 0.0:
             image = (image * 255).astype(np.uint8)
        else:
             # Assume it's some other range, try to normalize (this might be risky)
             logger.warning("visualize_segmentation received non-uint8 image not in [0,1] range. Attempting normalization.")
             img_min, img_max = image.min(), image.max()
             if img_max > img_min:
                 image = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
             else:
                 image = np.zeros_like(image, dtype=np.uint8) # Fallback to black

    # Make a copy to avoid modifying the original image
    vis = image.copy()

    # Determine number of classes
    num_classes = (
        len(id2label) if id2label
        else int(np.max(segmentation_map)) + 1 if np.any(segmentation_map) else 1 # Handle empty mask case
    )

    # Create colormap
    if colormap is None:
        cmap = create_colormap(num_classes)
    elif isinstance(colormap, str):
        cmap = plt.cm.get_cmap(colormap, num_classes)
    else:
        cmap = colormap

    # Handle shape mismatch
    if image.shape[:2] != segmentation_map.shape:
        logger.warning(
            f"Image shape {image.shape[:2]} doesn't match segmentation map shape {segmentation_map.shape}. "
            "Resizing segmentation map using nearest neighbor."
        )
        # Add more detailed logging about the mismatch
        logger.warning(f"Detailed shape info - Image: {image.shape}, Segmentation map: {segmentation_map.shape}")
        
        # Comprehensive resizing with multiple fallback methods
        try:
            # METHOD 1: Use PIL for resizing (generally most reliable for segmentation masks)
            from PIL import Image as PILImage
            logger.info(f"Attempting PIL resize from {segmentation_map.shape} to {(image.shape[1], image.shape[0])}")
            mask_img = PILImage.fromarray(segmentation_map.astype(np.uint8))
            mask_img = mask_img.resize((image.shape[1], image.shape[0]), resample=PILImage.NEAREST)
            segmentation_map = np.array(mask_img)
            logger.info(f"Resized segmentation map using PIL. New shape: {segmentation_map.shape}")
            
            # Verify resize was successful
            if image.shape[:2] != segmentation_map.shape:
                raise RuntimeError(f"PIL resize failed to match target dimensions - Result: {segmentation_map.shape}, Target: {image.shape[:2]}")
                
        except Exception as e_pil:
            logger.error(f"PIL resize failed: {e_pil}. Trying OpenCV...")
            
            try:
                # METHOD 2: Try using OpenCV with exact target dimensions
                import cv2
                logger.info(f"Attempting OpenCV resize to exact dimensions {(image.shape[1], image.shape[0])}")
                segmentation_map = cv2.resize(
                    segmentation_map.astype(np.uint8), 
                    (image.shape[1], image.shape[0]), 
                    interpolation=cv2.INTER_NEAREST
                )
                logger.info(f"Resized segmentation map using OpenCV. New shape: {segmentation_map.shape}")
                
                # Verify OpenCV resize worked
                if image.shape[:2] != segmentation_map.shape:
                    raise RuntimeError(f"OpenCV resize failed: Result shape {segmentation_map.shape} != Target {image.shape[:2]}")
                    
            except Exception as e_cv2:
                logger.error(f"OpenCV resize failed: {e_cv2}. Trying PyTorch...")
                
                try:
                    # METHOD 3: PyTorch interpolation as a last resort
                    logger.info(f"Attempting PyTorch interpolate with size={image.shape[:2]}")
                    mask_tensor = torch.from_numpy(segmentation_map).long().unsqueeze(0).unsqueeze(0) # B=1, C=1, H, W
                    resized_tensor = F.interpolate(mask_tensor.float(), size=image.shape[:2], mode='nearest')
                    segmentation_map = resized_tensor.squeeze().long().numpy()
                    logger.info(f"Resized segmentation map using PyTorch. New shape: {segmentation_map.shape}")
                    
                    # Final verification
                    if image.shape[:2] != segmentation_map.shape:
                        logger.error(f"PyTorch resize also failed! Creating empty mask as last resort.")
                        segmentation_map = np.zeros(image.shape[:2], dtype=np.uint8)
                        
                except Exception as e_torch:
                    logger.error(f"All resize methods failed! {e_torch}. Creating empty mask of correct size.")
                    segmentation_map = np.zeros(image.shape[:2], dtype=np.uint8)

    # Create mask for visualization (skip class 0, typically background)
    for class_idx in range(1, num_classes):
        mask = segmentation_map == class_idx
        if np.any(mask):
            # Get color for this class
            try:
                color = np.array(cmap(class_idx)[:3]) * 255
            except IndexError:
                 logger.warning(f"Colormap index {class_idx} out of bounds for colormap size. Using default color.")
                 # Fallback color (e.g., magenta) if index is out of bounds
                 color = np.array([255, 0, 255])

            # Apply color with alpha blending - ensure proper type conversion
            vis[mask] = ((1 - alpha) * vis[mask].astype(np.float32) + 
                         alpha * color.reshape(1, 3).astype(np.float32)).astype(np.uint8)

    return vis


# Moved from image_utils.py
def create_pseudocolor(
    mask: np.ndarray,
    colormap: str = 'viridis'
) -> np.ndarray:
    """
    Create pseudocolor visualization of a mask.

    Args:
        mask: Input mask (H, W)
        colormap: Matplotlib colormap name

    Returns:
        Pseudocolor visualization (H, W, 3)
    """
    # Normalize mask to [0, 1]
    if mask.dtype == np.uint8 and mask.max() > 1: # Check if it's likely class indices > 1
        # If mask contains class indices, normalize based on max index
        mask_max = mask.max()
        if mask_max > 0:
            normalized_mask = mask.astype(np.float32) / mask_max
        else:
            normalized_mask = np.zeros_like(mask, dtype=np.float32)
    elif mask.dtype == np.uint8: # Assume binary 0/255
        normalized_mask = mask.astype(np.float32) / 255.0
    else: # Assume float mask, potentially not in [0, 1]
        normalized_mask = mask.astype(np.float32)
        mask_min = normalized_mask.min()
        mask_max = normalized_mask.max()
        if mask_max > mask_min:
            normalized_mask = (normalized_mask - mask_min) / (mask_max - mask_min)
        else:
             normalized_mask = np.zeros_like(mask, dtype=np.float32) # Handle constant mask

    # Apply colormap
    cmap = plt.cm.get_cmap(colormap)
    pseudocolor = cmap(normalized_mask)

    # Convert to uint8 [0, 255]
    pseudocolor = (pseudocolor[:, :, :3] * 255).astype(np.uint8)

    return pseudocolor

# Moved from image_utils.py
def decode_segmentation_mask(
    mask: np.ndarray,
    id2label: Dict[int, str],
    ignore_index: int = 255
) -> Tuple[np.ndarray, List[Tuple[str, np.ndarray]]]: # Return color as numpy array
    """
    Decode segmentation mask for visualization.

    Args:
        mask: Segmentation mask (H, W)
        id2label: Dictionary mapping class IDs to class names
        ignore_index: Index to ignore in mask

    Returns:
        Tuple of (decoded mask (H, W, 3), legend items List[Tuple[name, color_array]])
    """
    # Create RGB mask
    rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)

    # Determine number of classes
    num_classes = len(id2label)
    cmap = create_colormap(num_classes)

    # Create legend items
    legend_items = []

    # Color each class
    for class_id, class_name in id2label.items():
        # Skip ignored index
        if class_id == ignore_index:
            continue

        # Get color for this class
        try:
            color = np.array(cmap(class_id)[:3]) * 255
        except IndexError:
             logger.warning(f"Colormap index {class_id} out of bounds for colormap size. Using default color.")
             color = np.array([255, 0, 255]) # Fallback color

        # Apply color to pixels of this class
        class_mask = mask == class_id
        if np.any(class_mask):
            rgb_mask[class_mask] = color.astype(np.uint8)
            legend_items.append((class_name, color / 255.0)) # Store color as float [0,1]

    return rgb_mask, legend_items

def plot_segmentation_comparison(
    image: Union[np.ndarray, torch.Tensor],
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
    alpha: float = 0.5,
    save_path: Optional[str] = None,
    id2label: Optional[Dict[int, str]] = None
) -> plt.Figure:
    """
    Plot comparison between original image, prediction, and target.

    Args:
        image: Original image (H, W, 3) or (3, H, W)
        pred: Predicted segmentation mask (H, W)
        target: Target segmentation mask (H, W)
        title: Title for the figure
        figsize: Figure size
        alpha: Transparency for segmentation overlay
        save_path: Path to save the figure (if None, doesn't save)
        id2label: Dictionary mapping class indices to class names

    Returns:
        Matplotlib figure
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(image, torch.Tensor):
        image = tensor_to_image(image)
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Ensure image is in HWC format
    if len(image.shape) == 3 and image.shape[0] == 3:
        # Convert from CHW to HWC
        image = np.transpose(image, (1, 2, 0))

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Plot prediction
    pred_vis = visualize_segmentation(image, pred, id2label=id2label, alpha=alpha)
    axes[1].imshow(pred_vis)
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    # Plot target
    target_vis = visualize_segmentation(image, target, id2label=id2label, alpha=alpha)
    axes[2].imshow(target_vis)
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")

    # Set figure title
    if title:
        fig.suptitle(title, fontsize=16)

    # Adjust layout
    plt.tight_layout()

    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig) # Add plt.close()
    return fig


def visualize_class_distribution(
    class_distribution: Dict[int, float],
    id2label: Dict[int, str],
    title: str = 'Class Distribution in Dataset',
    figsize: Tuple[int, int] = (8, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots the class distribution as a bar chart.

    Args:
        class_distribution: Dictionary mapping class IDs to their percentage.
        id2label: Dictionary mapping class IDs to class names.
        title: Title for the plot.
        figsize: Figure size.
        save_path: Path to save the figure.

    Returns:
        Matplotlib figure.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    classes = sorted(class_distribution.keys())
    class_names = [id2label.get(cls, f"Class {cls}") for cls in classes]
    percentages = [class_distribution[cls] for cls in classes]

    bars = ax.bar(class_names, percentages, color='skyblue')
    ax.set_title(title)
    ax.set_ylabel('Percentage of Pixels')
    ax.set_xlabel('Class')
    ax.tick_params(axis='x', rotation=45, ha='right')

    # Add value labels above bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.1f}%", va='bottom', ha='center') # Adjust position slightly

    plt.ylim(0, 105) # Ensure space for labels above 100% if needed
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # Return the figure object directly, let caller handle closing if needed when plot_return=True
    plt.close(fig) # Close figure after saving or if not saving
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    normalize: bool = False,
    title: str = 'Confusion Matrix',
    cmap=plt.cm.Blues,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots the confusion matrix using seaborn heatmap.

    Args:
        cm: Confusion matrix array (output of sklearn.metrics.confusion_matrix).
        class_names: List of class names for labels.
        normalize: Whether to normalize the matrix by row (true label).
        title: Title for the plot.
        cmap: Colormap for the heatmap.
        figsize: Figure size.
        save_path: Path to save the figure.

    Returns:
        Matplotlib figure.
    """
    if normalize:
        # Avoid division by zero if a row sum is 0
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        # Use np.errstate to temporarily ignore invalid division warnings
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_normalized = cm.astype('float') / row_sums
        cm_normalized = np.nan_to_num(cm_normalized) # Replace NaN with 0
        cm_to_plot = cm_normalized
        fmt = '.2f' # Format for normalized values
        print("Normalized confusion matrix")
    else:
        cm_to_plot = cm
        fmt = 'd' # Format for integer counts
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_to_plot, annot=True, fmt=fmt, cmap=cmap, xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=True)

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # Return the figure object directly, let caller handle closing if needed when plot_return=True
    plt.close(fig) # Close figure after saving or if not saving
    return fig


def plot_augmented_samples(
    original_sample: Dict[str, Any],
    transform: Callable,
    num_augmented: int = 4,
    figsize: Tuple[int, int] = (15, 8),
    save_path: Optional[str] = None,
    id2label: Optional[Dict[int, str]] = None,
    alpha: float = 0.5
) -> plt.Figure:
    """
    Visualizes the effect of augmentation transforms on a single sample.

    Args:
        original_sample: A dictionary containing the original 'image' and 'mask' (PIL Images or numpy arrays).
        transform: The augmentation transform function to apply.
        num_augmented: Number of augmented versions to generate and display.
        figsize: Figure size.
        save_path: Path to save the figure.
        id2label: Dictionary mapping class indices to class names for mask visualization.
        alpha: Transparency for segmentation overlay.

    Returns:
        Matplotlib figure.
    """
    # Ensure input is in the expected format (PIL Image) for transforms
    orig_image = original_sample['image']
    orig_mask = original_sample['mask']
    if isinstance(orig_image, np.ndarray):
        orig_image = Image.fromarray(orig_image)
    if isinstance(orig_mask, np.ndarray):
        orig_mask = Image.fromarray(orig_mask.astype(np.uint8), mode='L')

    # Create figure
    num_cols = num_augmented + 1 # Original + Augmented versions
    fig, axes = plt.subplots(2, num_cols, figsize=figsize) # Row 0: Images, Row 1: Masks/Overlays

    # --- Plot Original Sample ---
    # Original Image
    axes[0, 0].imshow(np.array(orig_image))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # Original Mask Overlay
    orig_mask_np = np.array(orig_mask)
    orig_overlay = visualize_segmentation(np.array(orig_image), orig_mask_np, id2label=id2label, alpha=alpha)
    axes[1, 0].imshow(orig_overlay)
    axes[1, 0].set_title("Original Mask")
    axes[1, 0].axis("off")

    # --- Plot Augmented Samples ---
    for i in range(num_augmented):
        # Apply transform
        # Important: Ensure transform receives copies if it modifies inplace
        augmented_sample = transform({'image': orig_image.copy(), 'mask': orig_mask.copy()})
        aug_image = augmented_sample['image']
        aug_mask = augmented_sample['mask']

        # Convert back to numpy for display if needed
        if isinstance(aug_image, Image.Image):
            aug_image_np = np.array(aug_image)
        else:
             aug_image_np = aug_image # Assume numpy or tensor convertible
             if isinstance(aug_image_np, torch.Tensor):
                 aug_image_np = tensor_to_image(aug_image_np)

        if isinstance(aug_mask, Image.Image):
            aug_mask_np = np.array(aug_mask)
        else:
             aug_mask_np = aug_mask
             if isinstance(aug_mask_np, torch.Tensor):
                 aug_mask_np = aug_mask_np.cpu().numpy()


        # Augmented Image
        axes[0, i + 1].imshow(aug_image_np)
        axes[0, i + 1].set_title(f"Augmented {i+1}")
        axes[0, i + 1].axis("off")

        # Augmented Mask Overlay
        aug_overlay = visualize_segmentation(aug_image_np, aug_mask_np, id2label=id2label, alpha=alpha)
        axes[1, i + 1].imshow(aug_overlay)
        axes[1, i + 1].set_title(f"Augmented Mask {i+1}")
        axes[1, i + 1].axis("off")

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig) # Add plt.close()
    return fig


def plot_worst_predictions(
    worst_samples: List[Tuple[float, int, Union[np.ndarray, None], np.ndarray, np.ndarray]],
    figsize: Tuple[int, int] = (20, 4),
    metric_name: str = "IoU",
    save_dir: Optional[str] = None,
    id2label: Optional[Dict[int, str]] = None
) -> List[plt.Figure]:
    """
    Plot worst predictions for visual inspection.

    Args:
        worst_samples: List of tuples (metric_value, sample_index, image, pred, label)
        figsize: Base figure size per sample
        metric_name: Name of the metric used for ranking
        save_dir: Directory to save figures (if None, doesn't save)
        id2label: Dictionary mapping class indices to class names

    Returns:
        List of Matplotlib figures
    """
    figures = []

    for i, (metric_value, sample_idx, image, pred, label) in enumerate(worst_samples):
        # Create figure for this sample
        title = f"Sample #{sample_idx} - {metric_name}: {metric_value:.4f}"
        
        # Handle case where image is not provided
        if image is None:
            # Create a default image (a gray canvas)
            h, w = pred.shape
            image = np.ones((h, w, 3), dtype=np.uint8) * 128
        
        # Plot comparison
        fig = plot_segmentation_comparison(
            image=image,
            pred=pred,
            target=label,
            title=title,
            figsize=figsize,
            id2label=id2label
        )
        
        figures.append(fig)
        
        # Save figure if requested
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"worst_{i+1}_sample_{sample_idx}.png")
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.close(fig) # Always close the figure after saving or if not saving
    
    # Return an empty list as figures are closed internally now
    return [] 


def plot_dataset_samples(
    dataset,
    num_samples: int = 5,
    figsize: Tuple[int, int] = (15, 5 * 5),
    alpha: float = 0.5,
    save_path: Optional[str] = None,
    id2label: Optional[Dict[int, str]] = None,
    random_state: Optional[int] = None
) -> plt.Figure:
    """
    Plot sample images from a dataset for inspection.

    Args:
        dataset: PyTorch Dataset with 'pixel_values' and 'labels' keys
        num_samples: Number of samples to plot
        figsize: Figure size
        alpha: Transparency for segmentation overlay
        save_path: Path to save the figure (if None, doesn't save)
        id2label: Dictionary mapping class indices to class names
        random_state: Random state for reproducible sampling

    Returns:
        Matplotlib figure
    """
    # Set random state if provided
    if random_state is not None:
        np.random.seed(random_state)
    
    # Sample indices
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        
        # Get image and mask
        image = sample['pixel_values']
        mask = sample['labels']
        
        # Convert to numpy arrays
        if isinstance(image, torch.Tensor):
            image = tensor_to_image(image)
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        
        # Create subplots for this sample
        gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gridspec.GridSpec(num_samples, 1)[i])
        
        # Plot original image
        ax_image = fig.add_subplot(gs[0])
        ax_image.imshow(image)
        ax_image.set_title(f"Sample {idx} - Image")
        ax_image.axis("off")
        
        # Plot mask
        ax_mask = fig.add_subplot(gs[1])
        mask_vis = create_pseudocolor(mask, colormap='viridis')
        ax_mask.imshow(mask_vis)
        ax_mask.set_title(f"Sample {idx} - Mask")
        ax_mask.axis("off")
        
        # Plot overlay
        ax_overlay = fig.add_subplot(gs[2])
        overlay = visualize_segmentation(image, mask, id2label=id2label, alpha=alpha)
        ax_overlay.imshow(overlay)
        ax_overlay.set_title(f"Sample {idx} - Overlay")
        ax_overlay.axis("off")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig) # Add plt.close()
    return fig


def plot_training_metrics(
    metrics_history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    smoothing_window: int = 1
) -> plt.Figure:
    """
    Plot training metrics over time.

    Args:
        metrics_history: Dictionary mapping metric names to lists of values
        figsize: Figure size
        save_path: Path to save the figure (if None, doesn't save)
        smoothing_window: Window size for moving average smoothing

    Returns:
        Matplotlib figure
    """
    # Create figure with subplots
    num_metrics = len(metrics_history)
    fig, axes = plt.subplots(num_metrics, 1, figsize=figsize, sharex=True)
    
    # If there's only one metric, wrap axes in a list for consistent indexing
    if num_metrics == 1:
        axes = [axes]
    
    # Plot each metric
    for i, (metric_name, values) in enumerate(metrics_history.items()):
        ax = axes[i]
        
        # Apply smoothing if requested
        if smoothing_window > 1 and len(values) >= smoothing_window:
            smoothed_values = []
            for j in range(len(values) - smoothing_window + 1):
                smoothed_values.append(np.mean(values[j:j+smoothing_window]))
            
            # Plot original values as light line
            ax.plot(range(len(values)), values, alpha=0.3, color='gray')
            
            # Plot smoothed values as bold line
            ax.plot(
                range(smoothing_window - 1, len(values)), 
                smoothed_values, 
                linewidth=2,
                label=f"{metric_name} (smoothed)"
            )
        else:
            # Plot original values only
            ax.plot(range(len(values)), values, linewidth=2)
        
        # Set title and labels
        ax.set_title(f"{metric_name} vs. Steps")
        ax.set_ylabel(metric_name)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Show legend if smoothing is applied
        if smoothing_window > 1 and len(values) >= smoothing_window:
            ax.legend()
    
    # Set common x-label
    axes[-1].set_xlabel("Steps")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.close(fig) # Add plt.close()
    return fig


def prepare_and_visualize_augmentations(
    dataset,
    sample_idx: int,
    transform,
    save_path: str,
    num_augmented: int = 4,
    logger = None,
    id2label: Optional[Dict[int, str]] = None
) -> bool:
    """
    Prepares and visualizes the effect of augmentation on a dataset sample.
    This function handles extracting the original image and mask from a dataset
    and applying the visualizations.
    
    Args:
        dataset: A dataset with access to original samples via dataset.dataset property
        sample_idx: The index of the sample to visualize
        transform: The augmentation transform to apply
        save_path: Path to save the visualization
        num_augmented: Number of augmented versions to generate and visualize
        logger: Optional logger for logging messages
        id2label: Optional dictionary mapping class indices to class names
        
    Returns:
        Boolean indicating success
    """
    try:
        # Access the original image and mask (before image processor transforms)
        original_item = dataset.dataset[sample_idx]
        original_image = original_item["image"]
        
        # Handle TiffImageFile objects by converting to numpy array if needed
        if hasattr(original_image, 'mode'):  # It's a PIL Image object
            original_image = np.array(original_image)
        
        # Get the mask using the dataset's mask feature name
        mask_feature = getattr(dataset, 'mask_feature', 'annotation')
        original_mask = np.array(original_item[mask_feature])
        
        # Apply basic mask preprocessing if the dataset provides it
        if hasattr(dataset, 'preprocess_mask'):
            original_mask = dataset.preprocess_mask(original_mask)
        
        # Log the visualization attempt
        if logger:
            logger.info(f"Visualizing augmentations for sample {sample_idx}...")
        
        # Generate the visualization using the existing plot function
        plot_augmented_samples(
            original_sample={'image': original_image, 'mask': original_mask},
            transform=transform,
            num_augmented=num_augmented,
            save_path=save_path,
            id2label=id2label
        )
        
        # Log success
        if logger:
            logger.info(f"Augmentation visualization saved to {save_path}")
        
        return True
    except Exception as e:
        # Log the error
        if logger:
            logger.error(f"Error visualizing augmentations for sample {sample_idx}: {e}")
        return False


def plot_loss_reduction(
    loss_history: List[float],
    steps: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (12, 6),
    smoothing_window: int = 5,
    highlight_significant: bool = True,
    significance_threshold: float = 0.1,
    show_velocity: bool = True,
    save_path: Optional[str] = None,
    title: str = "Loss Reduction Analysis",
    cmap: str = "viridis"
) -> plt.Figure:
    """
    Specialized visualization for loss reduction during training.
    
    Args:
        loss_history: List of loss values over training
        steps: List of step/epoch numbers (if None, uses sequential indices)
        figsize: Figure size
        smoothing_window: Window size for moving average smoothing
        highlight_significant: Whether to highlight significant drops in loss
        significance_threshold: Threshold for significant loss reduction (relative change)
        show_velocity: Whether to show rate of loss reduction
        save_path: Path to save the figure (if None, doesn't save)
        title: Figure title
        cmap: Colormap for velocity curve
        
    Returns:
        Matplotlib figure
    """
    if not loss_history:
        raise ValueError("Loss history must not be empty")
    
    # Create steps if not provided
    if steps is None:
        steps = list(range(len(loss_history)))
    
    # Apply smoothing to loss values
    smoothed_loss = None
    if smoothing_window > 1 and len(loss_history) >= smoothing_window:
        smoothed_loss = []
        for i in range(len(loss_history) - smoothing_window + 1):
            smoothed_loss.append(np.mean(loss_history[i:i+smoothing_window]))
    
    # Calculate loss reduction velocity (negative of first derivative)
    velocity = None
    if show_velocity and len(loss_history) > 1:
        # Use smoothed values for velocity if available
        values_for_velocity = smoothed_loss if smoothed_loss else loss_history
        # Only calculate velocity where we have smoothed values
        steps_for_velocity = steps[smoothing_window-1:] if smoothed_loss else steps
        
        # Calculate differences between adjacent values
        raw_velocity = [-1 * (values_for_velocity[i] - values_for_velocity[i-1]) 
                      for i in range(1, len(values_for_velocity))]
        
        # Normalize to [0, 1] for colormapping
        min_v, max_v = min(raw_velocity), max(raw_velocity)
        if min_v != max_v:
            velocity = [(v - min_v) / (max_v - min_v) for v in raw_velocity]
        else:
            velocity = [0.5] * len(raw_velocity)  # Default to middle value if all velocities are the same
            
        # Skip the first step for alignment
        steps_for_velocity = steps_for_velocity[1:]
    
    # Find significant drops in loss
    significant_points = []
    if highlight_significant and len(loss_history) > 1:
        values_to_check = smoothed_loss if smoothed_loss else loss_history
        steps_to_check = steps[smoothing_window-1:] if smoothed_loss else steps
        
        for i in range(1, len(values_to_check)):
            relative_change = (values_to_check[i-1] - values_to_check[i]) / values_to_check[i-1]
            if relative_change > significance_threshold:
                significant_points.append((
                    steps_to_check[i],     # Step
                    values_to_check[i],    # Loss value
                    relative_change        # Relative reduction
                ))
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Plot original loss values as a light gray line
    ax1.plot(steps, loss_history, alpha=0.4, color='gray', linewidth=1, label='Raw Loss')
    
    # Plot smoothed loss if available
    if smoothed_loss:
        ax1.plot(
            steps[smoothing_window-1:], 
            smoothed_loss, 
            color='blue', 
            linewidth=2.5, 
            label=f'Smoothed Loss (window={smoothing_window})'
        )
    
    # Plot velocity as a colored line if requested
    if show_velocity and velocity:
        # Create a colormap for velocity
        cmap_obj = plt.cm.get_cmap(cmap)
        
        # Plot line segments with color based on velocity
        for i in range(len(steps_for_velocity) - 1):
            x = [steps_for_velocity[i], steps_for_velocity[i+1]]
            # Use smoothed loss values if available, otherwise use original
            y_vals = smoothed_loss if smoothed_loss else loss_history
            # Adjust indices to align with velocity
            idx_offset = (smoothing_window) if smoothed_loss else 1
            y = [y_vals[i+idx_offset-1], y_vals[i+idx_offset]]
            
            ax1.plot(x, y, color=cmap_obj(velocity[i]), linewidth=3)
        
        # Add colorbar for velocity
        sm = plt.cm.ScalarMappable(cmap=cmap_obj)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax1)
        cbar.set_label('Loss Reduction Rate (higher is better)')
    
    # Mark significant drops
    if highlight_significant and significant_points:
        steps_sig = [p[0] for p in significant_points]
        loss_sig = [p[1] for p in significant_points]
        ax1.scatter(steps_sig, loss_sig, color='red', s=100, zorder=5, 
                   marker='o', label='Significant Drops')
        
        # Add annotations for significant points
        for step, loss_val, rel_change in significant_points:
            ax1.annotate(
                f"{rel_change:.1%}↓",
                (step, loss_val),
                textcoords="offset points",
                xytext=(0, -15),
                ha='center',
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8)
            )
    
    # Configure the main axis
    ax1.set_title(title, fontsize=14, pad=20)
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Loss Value', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis to log scale if range is large enough
    if max(loss_history) / (min(loss_history) + 1e-10) > 10:
        ax1.set_yscale('log')
        ax1.set_ylabel('Loss Value (log scale)', fontsize=12)
    
    ax1.legend(loc='upper right')
    
    # Add descriptive text
    total_reduction = (loss_history[0] - loss_history[-1]) / loss_history[0]
    final_loss = loss_history[-1]
    
    description = (
        f"Total Loss Reduction: {total_reduction:.2%}\n"
        f"Final Loss: {final_loss:.4f}\n"
        f"Initial Loss: {loss_history[0]:.4f}"
    )
    
    # Position the text box in figure coords
    fig.text(0.02, 0.02, description, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.close(fig) # Add plt.close()
    return fig


def plot_accuracy_gain(
    metric_history: Dict[str, List[float]],
    steps: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (14, 8),
    smoothing_window: int = 5,
    highlight_improvements: bool = True,
    improvement_threshold: float = 0.05,
    metrics_to_plot: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Accuracy Metrics Improvement Analysis",
    ascending_metrics: Optional[List[str]] = None
) -> plt.Figure:
    """
    Specialized visualization for accuracy metric improvements during training.
    
    Args:
        metric_history: Dictionary mapping metric names to lists of values
        steps: List of step/epoch numbers (if None, uses sequential indices)
        figsize: Figure size
        smoothing_window: Window size for moving average smoothing
        highlight_improvements: Whether to highlight significant improvements
        improvement_threshold: Threshold for significant improvement (absolute change)
        metrics_to_plot: List of metrics to plot (if None, plots all in metric_history)
        save_path: Path to save the figure (if None, doesn't save)
        title: Figure title
        ascending_metrics: List of metrics where higher values are better (default: all)
        
    Returns:
        Matplotlib figure
    """
    if not metric_history:
        raise ValueError("Metric history must not be empty")
    
    # Determine which metrics to plot
    if metrics_to_plot is None:
        metrics_to_plot = list(metric_history.keys())
    else:
        # Ensure all requested metrics exist in the history
        for metric in metrics_to_plot:
            if metric not in metric_history:
                raise ValueError(f"Metric '{metric}' not found in metric history")
    
    # Determine which metrics should be ascending (higher is better)
    if ascending_metrics is None:
        # By default, assume all metrics should be ascending
        ascending_metrics = list(metrics_to_plot)
    
    # Create steps if not provided
    if steps is None:
        # Find the longest metric list to determine the number of steps
        max_len = max(len(metric_history[m]) for m in metrics_to_plot)
        steps = list(range(max_len))
    
    # Determine number of metrics and create figure
    num_metrics = len(metrics_to_plot)
    num_rows = (num_metrics + 1) // 2  # 2 metrics per row, with an extra row if odd
    fig, axes = plt.subplots(num_rows, min(2, num_metrics), figsize=figsize)
    
    # Handle case of single metric (make axes iterable)
    if num_metrics == 1:
        axes = np.array([axes])
    
    # Flatten axes array for easier indexing
    if num_rows > 1 and num_metrics > 1:
        axes = axes.flatten()
    
    # Process each metric
    for i, metric_name in enumerate(metrics_to_plot):
        ax = axes[i] if num_metrics > 1 else axes
        values = metric_history[metric_name]
        
        # Ensure steps and values align in length
        metric_steps = steps[:len(values)]
        
        # Determine if this metric should be ascending or descending
        is_ascending = metric_name in ascending_metrics
        
        # Apply smoothing
        smoothed_values = None
        if smoothing_window > 1 and len(values) >= smoothing_window:
            smoothed_values = []
            for j in range(len(values) - smoothing_window + 1):
                smoothed_values.append(np.mean(values[j:j+smoothing_window]))
        
        # Find significant improvements
        significant_points = []
        if highlight_improvements and len(values) > 1:
            check_values = smoothed_values if smoothed_values else values
            check_steps = metric_steps[smoothing_window-1:] if smoothed_values else metric_steps
            
            for j in range(1, len(check_values)):
                change = check_values[j] - check_values[j-1]
                # For descending metrics, negate the change
                if not is_ascending:
                    change = -change
                
                if change > improvement_threshold:
                    significant_points.append((
                        check_steps[j],    # Step
                        check_values[j],   # Value
                        change             # Absolute improvement
                    ))
        
        # Plot raw values with lower alpha
        ax.plot(metric_steps, values, alpha=0.4, color='gray', linewidth=1, label='Raw Values')
        
        # Plot smoothed values if available
        if smoothed_values:
            smooth_color = 'green' if is_ascending else 'red'
            ax.plot(
                metric_steps[smoothing_window-1:], 
                smoothed_values, 
                color=smooth_color, 
                linewidth=2.5, 
                label=f'Smoothed (window={smoothing_window})'
            )
        
        # Mark significant improvements
        if highlight_improvements and significant_points:
            imp_steps = [p[0] for p in significant_points]
            imp_values = [p[1] for p in significant_points]
            marker_color = 'green' if is_ascending else 'red'
            ax.scatter(imp_steps, imp_values, color=marker_color, s=80, zorder=5, 
                      marker='^' if is_ascending else 'v', label='Significant Changes')
            
            # Add annotations for significant points
            for step, value, change in significant_points:
                ax.annotate(
                    f"+{change:.3f}" if is_ascending else f"-{-change:.3f}",
                    (step, value),
                    textcoords="offset points",
                    xytext=(0, 10 if is_ascending else -15),
                    ha='center',
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=marker_color, alpha=0.7)
                )
        
        # Configure the axis
        pretty_name = metric_name.replace('_', ' ').title()
        ax.set_title(f"{pretty_name} {'Improvement' if is_ascending else 'Reduction'}", fontsize=12)
        ax.set_xlabel('Steps')
        ax.set_ylabel(pretty_name)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        ax.legend(loc='best', fontsize=8)
        
        # Add overall improvement text
        if len(values) > 1:
            overall_change = values[-1] - values[0]
            if not is_ascending:
                overall_change = -overall_change
            
            change_text = f"Overall {'Improvement' if overall_change >= 0 else 'Decline'}: {abs(overall_change):.4f}"
            final_value_text = f"Final: {values[-1]:.4f}"
            
            # Add text box with improvement stats
            textbox_props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7)
            text = f"{change_text}\n{final_value_text}"
            ax.text(0.05, 0.05, text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='bottom', bbox=textbox_props)
    
    # Hide any unused subplots
    for j in range(num_metrics, len(axes) if isinstance(axes, np.ndarray) else 1):
        if isinstance(axes, np.ndarray):
            axes[j].set_visible(False)
    
    # Add overall title
    fig.suptitle(title, fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.close(fig) # Add plt.close()
    return fig


def plot_prediction_confidence(
    image: Union[np.ndarray, torch.Tensor],
    prediction: Union[np.ndarray, torch.Tensor],
    confidence_map: Optional[Union[np.ndarray, torch.Tensor]] = None,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 5),
    colormap: str = 'viridis',
    save_path: Optional[str] = None,
    title: Optional[str] = "Prediction Confidence Visualization"
) -> np.ndarray:
    """
    Visualize the confidence of predictions made by the model.

    Args:
        image: Original image (H, W, 3) or (3, H, W)
        prediction: Predicted segmentation mask (H, W) or logits (C, H, W)
        confidence_map: Optional pre-computed confidence map (H, W)
        class_names: List of class names for visualization
        figsize: Figure size
        colormap: Colormap for confidence visualization
        save_path: Path to save the figure (if None, doesn't save)
        title: Title for the figure

    Returns:
        Numpy array representing the rendered visualization image (H, W, 3 or 4).
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(image, torch.Tensor):
        image = tensor_to_image(image)
    if len(image.shape) == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))

    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu()

    # Compute confidence map if not provided
    if confidence_map is None:
        if isinstance(prediction, torch.Tensor):
            if len(prediction.shape) == 3:
                if prediction.shape[0] > 1:
                    if torch.min(prediction) < 0 or torch.max(prediction) > 1:
                        probs = F.softmax(prediction, dim=0)
                    else:
                        probs = prediction
                    confidence_map = torch.max(probs, dim=0)[0].numpy()
                    prediction = torch.argmax(prediction, dim=0).numpy()
                else:
                    if torch.min(prediction) < 0 or torch.max(prediction) > 1:
                        probs = torch.sigmoid(prediction)
                    else:
                        probs = prediction
                    confidence_map = probs[0].numpy()
                    prediction = (probs[0] > 0.5).numpy().astype(np.uint8)
            else:
                prediction = prediction.numpy()
                confidence_map = np.ones_like(prediction, dtype=np.float32)
        else:
            confidence_map = np.ones_like(prediction, dtype=np.float32)
    elif isinstance(confidence_map, torch.Tensor):
        confidence_map = confidence_map.detach().cpu().numpy()

    if isinstance(prediction, torch.Tensor):
        prediction = prediction.numpy()

    # Ensure prediction and confidence map match image dimensions
    target_size = image.shape[:2] # H, W

    # Resize prediction if necessary
    if prediction.shape != target_size:
        logger.warning(f"Resizing prediction in plot_prediction_confidence from {prediction.shape} to {target_size}.")
        # Ensure prediction is integer type for nearest interpolation, but interpolate as float
        pred_tensor = torch.from_numpy(prediction).unsqueeze(0).unsqueeze(0).float() # B, C, H, W
        resized_pred = F.interpolate(pred_tensor, size=target_size, mode='nearest')
        prediction = resized_pred.squeeze().long().numpy() # H, W

    # Resize confidence map if necessary
    if confidence_map.shape != target_size:
        logger.warning(f"Resizing confidence_map in plot_prediction_confidence from {confidence_map.shape} to {target_size}.")
        conf_tensor = torch.from_numpy(confidence_map).unsqueeze(0).unsqueeze(0).float() # B, C, H, W
        # Use bilinear for smoother confidence map resizing
        resized_conf = F.interpolate(conf_tensor, size=target_size, mode='bilinear', align_corners=False)
        confidence_map = resized_conf.squeeze().numpy() # H, W

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    if class_names:
        pred_vis = visualize_segmentation(image, prediction, id2label={i: name for i, name in enumerate(class_names)})
        axes[1].imshow(pred_vis)
    else:
        pred_vis = create_pseudocolor(prediction)
        axes[1].imshow(pred_vis)
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    conf_img = axes[2].imshow(confidence_map, cmap=colormap, vmin=0, vmax=1)
    axes[2].set_title("Confidence")
    axes[2].axis("off")
    cbar = fig.colorbar(conf_img, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("Confidence Score")

    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    # Always return a NumPy array (not a Figure)
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=fig.dpi, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    img_arr = np.array(img)
    buf.close()

    # Type check for safety
    if not isinstance(img_arr, np.ndarray):
        raise RuntimeError("plot_prediction_confidence did not return a numpy.ndarray as expected.")
    return img_arr


def plot_error_analysis_map(
    image: Union[np.ndarray, torch.Tensor],
    prediction: Union[np.ndarray, torch.Tensor],
    ground_truth: Union[np.ndarray, torch.Tensor],
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None,
    title: Optional[str] = "Error Analysis"
) -> plt.Figure:
    """
    Visualize the errors in prediction compared to ground truth.
    
    Args:
        image: Original image (H, W, 3) or (3, H, W)
        prediction: Predicted segmentation mask (H, W)
        ground_truth: Ground truth segmentation mask (H, W)
        class_names: List of class names for visualization
        figsize: Figure size
        save_path: Path to save the figure (if None, doesn't save)
        title: Title for the figure
        
    Returns:
        Matplotlib figure
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(image, torch.Tensor):
        image = tensor_to_image(image)
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()
    
    # Ensure image is in HWC format
    if len(image.shape) == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    
    # Create error map (0: correct, 1: false positive, 2: false negative)
    error_map = np.zeros_like(prediction)
    error_map[(prediction != ground_truth) & (prediction > 0)] = 1  # False positive
    error_map[(prediction != ground_truth) & (ground_truth > 0)] = 2  # False negative
    
    # Create a custom colormap for error visualization
    colors = [(0, 1, 0, 0.0),  # Transparent for correct predictions
              (1, 0, 0, 0.7),  # Red for false positives
              (0, 0, 1, 0.7)]  # Blue for false negatives
    cmap = ListedColormap(colors)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Plot ground truth
    if class_names:
        gt_vis = visualize_segmentation(image, ground_truth, id2label={i: name for i, name in enumerate(class_names)})
        axes[1].imshow(gt_vis)
    else:
        # Create a pseudocolor visualization of the ground truth
        gt_vis = create_pseudocolor(ground_truth)
        axes[1].imshow(gt_vis)
    
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")
    
    # Plot error map overlaid on image
    axes[2].imshow(image)
    error_vis = axes[2].imshow(error_map, cmap=cmap, vmin=0, vmax=2)
    axes[2].set_title("Error Map")
    axes[2].axis("off")
    
    # Add legend for error types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='False Positive'),
        Patch(facecolor='blue', alpha=0.7, label='False Negative')
    ]
    axes[2].legend(handles=legend_elements, loc='lower right', framealpha=0.7)
    
    # Set figure title if provided
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.close(fig)  # Close the figure
    return fig


# Alias for visualize_class_distribution to match import name
def plot_class_distribution(
    class_distribution: Dict[int, float],
    id2label: Dict[int, str],
    title: str = 'Class Distribution in Dataset',
    figsize: Tuple[int, int] = (8, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Alias for visualize_class_distribution function.
    
    See visualize_class_distribution for full documentation.
    """
    return visualize_class_distribution(
        class_distribution=class_distribution,
        id2label=id2label,
        title=title,
        figsize=figsize,
        save_path=save_path
    )


def plot_training_progress_dashboard(
    metrics_history: Dict[str, List[float]],
    loss_key: str = 'loss',
    accuracy_keys: Optional[List[str]] = None,
    steps: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (15, 10),
    smoothing_window: int = 5,
    include_derivatives: bool = True,
    save_path: Optional[str] = None,
    title: str = "Training Progress Dashboard",
    style: str = "default"
) -> plt.Figure:
    """
    Create a comprehensive dashboard visualization of training progress with loss reduction and accuracy gains.
    
    Args:
        metrics_history: Dictionary mapping metric names to lists of values
        loss_key: Key for loss values in metrics_history
        accuracy_keys: List of keys for accuracy metrics (if None, uses all non-loss keys)
        steps: List of step/epoch numbers (if None, uses sequential indices)
        figsize: Figure size
        smoothing_window: Window size for moving average smoothing
        include_derivatives: Whether to show rate of change for metrics
        save_path: Path to save the figure (if None, doesn't save)
        title: Dashboard title
        style: Visualization style ('default', 'dark', 'light', 'paper')
        
    Returns:
        Matplotlib figure
    """
    if not metrics_history:
        raise ValueError("Metrics history must not be empty")
    
    # Ensure loss key exists
    if loss_key not in metrics_history:
        raise ValueError(f"Loss key '{loss_key}' not found in metrics history")
    
    # Set matplotlib style if specified
    if style == 'dark':
        plt.style.use('dark_background')
    elif style == 'paper':
        plt.style.use('seaborn-paper')
    elif style == 'light':
        plt.style.use('seaborn-whitegrid')
    
    # Determine accuracy keys if not provided
    if accuracy_keys is None:
        accuracy_keys = [k for k in metrics_history.keys() if k != loss_key]
    
    # Create steps if not provided
    if steps is None:
        # Find the longest metric list
        max_len = max(len(metrics_history[m]) for m in metrics_history.keys())
        steps = list(range(max_len))
    
    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(4, 4, figure=fig)
    
    # 1. Create loss plot (larger panel on top)
    ax_loss = fig.add_subplot(gs[0:2, :])
    loss_values = metrics_history[loss_key]
    
    # Plot raw loss
    ax_loss.plot(steps[:len(loss_values)], loss_values, alpha=0.4, color='gray', 
               linewidth=1, label='Raw Loss')
    
    # Apply smoothing to loss
    if smoothing_window > 1 and len(loss_values) >= smoothing_window:
        smoothed_loss = []
        for i in range(len(loss_values) - smoothing_window + 1):
            smoothed_loss.append(np.mean(loss_values[i:i+smoothing_window]))
        
        # Plot smoothed loss
        ax_loss.plot(
            steps[smoothing_window-1:len(loss_values)], 
            smoothed_loss, 
            color='blue', 
            linewidth=2.5, 
            label=f'Smoothed Loss (window={smoothing_window})'
        )
        
        # Plot loss reduction rate (derivative) if requested
        if include_derivatives and len(smoothed_loss) > 1:
            # Create a second y-axis for the derivative
            ax_loss_der = ax_loss.twinx()
            
            # Calculate derivative (negative for reduction rate - higher is better)
            loss_derivatives = [-1 * (smoothed_loss[i] - smoothed_loss[i-1]) 
                             for i in range(1, len(smoothed_loss))]
            
            # Plot derivative as a faint area
            ax_loss_der.fill_between(
                steps[smoothing_window:len(loss_values)],
                0,
                loss_derivatives,
                alpha=0.2,
                color='green' if loss_derivatives[-1] > 0 else 'red',
                label='Loss Reduction Rate'
            )
            
            # Configure derivative axis
            ax_loss_der.set_ylabel('Loss Reduction Rate', color='green' if loss_derivatives[-1] > 0 else 'red')
            ax_loss_der.tick_params(axis='y', colors='green' if loss_derivatives[-1] > 0 else 'red')
            
            # Add to legend
            lines_der, labels_der = ax_loss_der.get_legend_handles_labels()
            lines_loss, labels_loss = ax_loss.get_legend_handles_labels()
            ax_loss.legend(lines_loss + lines_der, labels_loss + labels_der, loc='upper right')
        else:
            ax_loss.legend(loc='upper right')
    else:
        ax_loss.legend(loc='upper right')
    
    # Configure loss axis
    ax_loss.set_title(f"{loss_key.replace('_', ' ').title()} Reduction", fontsize=14)
    ax_loss.set_xlabel('Steps')
    ax_loss.set_ylabel(f"{loss_key.replace('_', ' ').title()}")
    ax_loss.grid(True, linestyle='--', alpha=0.7)
    
    # Add summary text for loss
    total_reduction = (loss_values[0] - loss_values[-1]) / (loss_values[0] + 1e-10)
    loss_text = (
        f"Total Reduction: {total_reduction:.2%}\n"
        f"Final Value: {loss_values[-1]:.4f}"
    )
    text_box = dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8)
    ax_loss.text(0.02, 0.05, loss_text, transform=ax_loss.transAxes, fontsize=10,
                bbox=text_box)
    
    # Y-axis log scale for loss if appropriate
    if max(loss_values) / (min(loss_values) + 1e-10) > 10:
        ax_loss.set_yscale('log')
    
    # 2. Create accuracy metrics plots (smaller panels below)
    accuracy_axes = []
    for i, metric_key in enumerate(accuracy_keys):
        # Skip if metric doesn't exist in history
        if metric_key not in metrics_history:
            continue
            
        # Create subplot in bottom grid
        row, col = (i // 2) + 2, (i % 2) * 2
        # Span 2 columns for each accuracy plot
        ax = fig.add_subplot(gs[row, col:col+2])
        accuracy_axes.append(ax)
        
        metric_values = metrics_history[metric_key]
        
        # Plot raw values
        ax.plot(steps[:len(metric_values)], metric_values, alpha=0.4, color='gray', 
              linewidth=1, label='Raw Values')
        
        # Apply smoothing
        if smoothing_window > 1 and len(metric_values) >= smoothing_window:
            smoothed_values = []
            for j in range(len(metric_values) - smoothing_window + 1):
                smoothed_values.append(np.mean(metric_values[j:j+smoothing_window]))
            
            # Plot smoothed values
            ax.plot(
                steps[smoothing_window-1:len(metric_values)], 
                smoothed_values, 
                color='green', 
                linewidth=2, 
                label=f'Smoothed (window={smoothing_window})'
            )
            
            # Plot improvement rate (derivative) if requested
            if include_derivatives and len(smoothed_values) > 1:
                # Create a second y-axis for the derivative
                ax_der = ax.twinx()
                
                # Calculate derivative (higher is better for accuracy)
                derivatives = [smoothed_values[j] - smoothed_values[j-1] 
                             for j in range(1, len(smoothed_values))]
                
                # Plot derivative as a faint area
                ax_der.fill_between(
                    steps[smoothing_window:len(metric_values)],
                    0,
                    derivatives,
                    alpha=0.2,
                    color='green' if derivatives[-1] > 0 else 'red',
                    label='Improvement Rate'
                )
                
                # Configure derivative axis
                ax_der.set_ylabel('Rate', color='green' if derivatives[-1] > 0 else 'red', fontsize=8)
                ax_der.tick_params(axis='y', labelsize=8, colors='green' if derivatives[-1] > 0 else 'red')
                
                # Add to legend
                lines_der, labels_der = ax_der.get_legend_handles_labels()
                lines_acc, labels_acc = ax.get_legend_handles_labels()
                ax.legend(lines_acc + lines_der, labels_acc + labels_der, loc='lower right', fontsize=8)
            else:
                ax.legend(loc='lower right', fontsize=8)
        else:
            ax.legend(loc='lower right', fontsize=8)
        
        # Configure accuracy axis
        pretty_name = metric_key.replace('_', ' ').title()
        ax.set_title(f"{pretty_name} Improvement", fontsize=12)
        ax.set_xlabel('Steps')
        ax.set_ylabel(pretty_name)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add descriptive text for this metric
        if len(metric_values) > 1:
            initial_value = metric_values[0]
            final_value = metric_values[-1]
            overall_change = final_value - initial_value
            
            # Create text annotation
            change_text = f"Change: {overall_change:+.4f} ({(overall_change/max(abs(initial_value), 1e-10))*100:+.1f}%)"
            ax.text(0.05, 0.95, change_text, transform=ax.transAxes, 
                   fontsize=10, va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add overall title
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
    plt.close(fig) # Add plt.close()
    return fig


def visualize_boundary_iou_components(
    image_np: np.ndarray,
    gt_mask_np: np.ndarray, # Should be binary (0 or 1)
    pred_mask_np: np.ndarray, # Should be binary (0 or 1)
    dilation_pixels: int = 5, # Or calculate based on dilation_ratio and image size
    alpha: float = 0.7,
    colors: dict = None
) -> np.ndarray:
    """
    Visualizes True Positive, False Positive, and False Negative boundary pixels.

    Args:
        image_np: Original image (H, W, 3), uint8.
        gt_mask_np: Ground truth binary mask (H, W), bool or uint8.
        pred_mask_np: Predicted binary mask (H, W), bool or uint8.
        dilation_pixels: Number of pixels for dilation to define boundary band.
        alpha: Transparency for overlaid boundaries.
        colors: Dictionary for TP, FP, FN boundary colors.
                e.g., {'tp': [255, 255, 0], 'fp': [255, 0, 0], 'fn': [0, 255, 0]}

    Returns:
        Visualization image (H, W, 3), uint8.
    """
    if colors is None:
        colors = {
            'tp': np.array([0, 255, 0], dtype=np.uint8),    # Green: Correct boundaries
            'fp': np.array([255, 255, 0], dtype=np.uint8),  # Yellow: False boundaries
            'fn': np.array([255, 0, 0], dtype=np.uint8)     # Red: Missed boundaries
        }

    # Ensure masks are boolean
    gt_mask_bool = gt_mask_np.astype(bool)
    pred_mask_bool = pred_mask_np.astype(bool)

    # Define connectivity structure for dilation
    struct = generate_binary_structure(2, 2) # 8-connectivity

    # Dilate masks
    gt_dilated = binary_dilation(gt_mask_bool, structure=struct, iterations=dilation_pixels)
    pred_dilated = binary_dilation(pred_mask_bool, structure=struct, iterations=dilation_pixels)

    # Get boundary regions (pixels in dilated mask but not in original mask)
    gt_boundary = gt_dilated & (~gt_mask_bool)
    pred_boundary = pred_dilated & (~pred_mask_bool)

    # Identify TP, FP, FN boundary pixels
    tp_boundary = gt_boundary & pred_boundary
    fp_boundary = pred_boundary & (~gt_boundary)
    fn_boundary = gt_boundary & (~pred_boundary)

    # Create visualization on a copy of the original image
    vis_image = image_np.copy()

    # Overlay boundaries
    # Apply FN first, then FP, then TP, so TP is most prominent if overlaps occur
    for mask, color_key in [(fn_boundary, 'fn'), (fp_boundary, 'fp'), (tp_boundary, 'tp')]:
        if np.any(mask):
            color = colors[color_key]
            # Blend color with the image
            vis_image[mask] = (
                (1 - alpha) * vis_image[mask].astype(np.float32) +
                alpha * color.astype(np.float32)
            ).astype(np.uint8)

    return vis_image


def plot_prediction_comparison_with_confidence(
    image: Union[np.ndarray, torch.Tensor],
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    confidence_map: Optional[Union[np.ndarray, torch.Tensor]] = None,
    figsize: Tuple[int, int] = (20, 5),
    alpha: float = 0.5,
    save_path: Optional[str] = None,
    id2label: Optional[Dict[int, str]] = None,
    colormap: str = 'viridis',
    title: Optional[str] = "Prediction vs Ground Truth vs Confidence"
) -> plt.Figure:
    """
    Visualize original image, prediction, ground truth, and confidence map side by side.

    Args:
        image: Original image (H, W, 3) or (3, H, W)
        pred: Predicted segmentation mask (H, W)
        target: Ground truth segmentation mask (H, W)
        confidence_map: Confidence map (H, W)
        figsize: Figure size
        alpha: Transparency for segmentation overlays
        save_path: Path to save the figure (if None, doesn't save)
        id2label: Dictionary mapping class indices to class names
        colormap: Colormap for confidence visualization
        title: Title for the figure

    Returns:
        Matplotlib figure
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(image, torch.Tensor):
        image = tensor_to_image(image)
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if confidence_map is not None and isinstance(confidence_map, torch.Tensor):
        confidence_map = confidence_map.detach().cpu().numpy()

    # Ensure image is in HWC format
    if len(image.shape) == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Prediction overlay
    pred_vis = visualize_segmentation(image, pred, id2label=id2label, alpha=alpha)
    axes[1].imshow(pred_vis)
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    # Ground truth overlay
    target_vis = visualize_segmentation(image, target, id2label=id2label, alpha=alpha)
    axes[2].imshow(target_vis)
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")

    # Confidence map
    conf_img = axes[3].imshow(confidence_map, cmap=colormap, vmin=0, vmax=1)
    axes[3].set_title("Confidence Map")
    axes[3].axis("off")
    cbar = fig.colorbar(conf_img, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.set_label("Confidence Score")

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig


def plot_learning_rate_schedule(
    scheduler: _LRScheduler,
    num_training_steps: int,
    optimizer: Optimizer,
    title: str = "Learning Rate Schedule",
    figsize: Tuple[int, int] = (10, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots the learning rate schedule over the training steps.

    Args:
        scheduler: The learning rate scheduler instance.
        num_training_steps: Total number of training steps.
        optimizer: The optimizer associated with the scheduler.
        title: Title for the plot.
        figsize: Figure size.
        save_path: Path to save the figure.

    Returns:
        Matplotlib figure.
    """
    # Store initial states using deepcopy
    initial_optimizer_state = copy.deepcopy(optimizer.state_dict())
    initial_scheduler_state = copy.deepcopy(scheduler.state_dict())

    # Simulate the scheduler steps using copies
    lrs = []
    # Create temporary optimizer and scheduler instances for simulation
    # Note: Creating a new optimizer might be complex if it depends on model parameters.
    # Instead, we'll manipulate the state copies directly.
    # We need a temporary optimizer instance to get the LR, but we won't step it.
    temp_optimizer = copy.deepcopy(optimizer)
    temp_scheduler = copy.deepcopy(scheduler)

    # Load the initial states into the temporary instances
    temp_optimizer.load_state_dict(initial_optimizer_state)
    temp_scheduler.load_state_dict(initial_scheduler_state)

    for step in range(num_training_steps):
        # Get LR from the temporary optimizer's param group
        lrs.append(temp_optimizer.param_groups[0]['lr'])
        # Step the temporary scheduler
        temp_scheduler.step()
        # Update the temporary optimizer's LR based on the temporary scheduler
        # This mimics how the training loop updates the LR
        temp_optimizer.param_groups[0]['lr'] = temp_scheduler.get_last_lr()[0]


    # The original optimizer and scheduler states remain unchanged.

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(num_training_steps), lrs)
    ax.set_title(title)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Learning Rate")
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig) # Close figure after saving or if not saving
    return fig


# =====================================================================================
# ENHANCED VISUALIZATION FUNCTIONS FOR BINARY TREE CROWN SEGMENTATION
# =====================================================================================

def visualize_segmentation_enhanced(
    image: np.ndarray,
    segmentation_map: np.ndarray,
    id2label: Optional[Dict[int, str]] = None,
    alpha: float = 0.6,
    use_natural_colors: bool = True,
    high_contrast: bool = True
) -> np.ndarray:
    """
    Enhanced segmentation visualization with improved colors and contrast for binary tree crown segmentation.

    Args:
        image: Input image (H, W, 3)
        segmentation_map: Segmentation map (H, W)
        id2label: Mapping from class indices to class names
        alpha: Transparency of the segmentation map
        use_natural_colors: Use natural colors (green for vegetation)
        high_contrast: Use high contrast colors for better visibility

    Returns:
        Visualization image (H, W, 3)
    """
    # Handle 3D segmentation maps
    if segmentation_map.ndim == 3:
        if segmentation_map.shape[-1] == 1:
            segmentation_map = segmentation_map.squeeze(axis=-1)
        elif segmentation_map.shape[-1] > 1:
            segmentation_map = np.argmax(segmentation_map, axis=-1)

    # Ensure image is uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    # Handle size mismatch
    if image.shape[:2] != segmentation_map.shape:
        from PIL import Image as PILImage
        mask_img = PILImage.fromarray(segmentation_map.astype(np.uint8))
        mask_img = mask_img.resize((image.shape[1], image.shape[0]), resample=PILImage.NEAREST)
        segmentation_map = np.array(mask_img)

    vis = image.copy()
    num_classes = len(id2label) if id2label else int(np.max(segmentation_map)) + 1

    # Enhanced color schemes for binary tree crown segmentation
    if use_natural_colors and num_classes == 2:
        # Natural colors optimized for tree crown segmentation
        if high_contrast:
            # High contrast green for better visibility on various backgrounds
            tree_color = np.array([0, 220, 0], dtype=np.uint8)  # Bright green
        else:
            # More natural forest green
            tree_color = np.array([34, 139, 34], dtype=np.uint8)  # Forest green
        
        # Apply tree crown visualization
        tree_mask = segmentation_map == 1
        if np.any(tree_mask):
            vis[tree_mask] = ((1 - alpha) * vis[tree_mask].astype(np.float32) + 
                             alpha * tree_color.astype(np.float32)).astype(np.uint8)
    else:
        # Fallback to original behavior for multi-class
        cmap = create_colormap(num_classes)
        for class_idx in range(1, num_classes):
            mask = segmentation_map == class_idx
            if np.any(mask):
                try:
                    color = np.array(cmap(class_idx)[:3]) * 255
                except IndexError:
                    color = np.array([255, 0, 255])  # Fallback magenta
                
                vis[mask] = ((1 - alpha) * vis[mask].astype(np.float32) + 
                            alpha * color.astype(np.float32)).astype(np.uint8)

    return vis


def visualize_error_decomposition(
    image: np.ndarray,
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    alpha: float = 0.7,
    use_natural_colors: bool = True
) -> np.ndarray:
    """
    Create enhanced error decomposition visualization showing TP, FP, FN regions.

    Args:
        image: Original image (H, W, 3)
        prediction: Predicted mask (H, W)
        ground_truth: Ground truth mask (H, W)
        alpha: Transparency for error overlays
        use_natural_colors: Use intuitive colors for different error types

    Returns:
        Error visualization image (H, W, 3)
    """
    # Ensure inputs are the right shape and type
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    # Handle size mismatches
    if image.shape[:2] != prediction.shape:
        from PIL import Image as PILImage
        pred_img = PILImage.fromarray(prediction.astype(np.uint8))
        pred_img = pred_img.resize((image.shape[1], image.shape[0]), resample=PILImage.NEAREST)
        prediction = np.array(pred_img)
    
    if image.shape[:2] != ground_truth.shape:
        from PIL import Image as PILImage
        gt_img = PILImage.fromarray(ground_truth.astype(np.uint8))
        gt_img = gt_img.resize((image.shape[1], image.shape[0]), resample=PILImage.NEAREST)
        ground_truth = np.array(gt_img)

    vis = image.copy()

    # Create binary masks for analysis
    pred_bool = prediction.astype(bool)
    gt_bool = ground_truth.astype(bool)

    # Define error regions
    true_positive = pred_bool & gt_bool       # Correctly predicted trees
    false_positive = pred_bool & (~gt_bool)   # Predicted trees where there are none
    false_negative = (~pred_bool) & gt_bool   # Missed trees

    if use_natural_colors:
        # Updated colors per user requirements:
        # Green for correct trees, Yellow for false alarms, Red for missed trees
        tp_color = np.array([0, 255, 0], dtype=np.uint8)      # Green: Correct trees
        fp_color = np.array([255, 255, 0], dtype=np.uint8)    # Yellow: False alarms  
        fn_color = np.array([255, 0, 0], dtype=np.uint8)      # Red: Missed trees
    else:
        # Standard error visualization colors
        tp_color = np.array([0, 255, 0], dtype=np.uint8)      # Green
        fp_color = np.array([255, 255, 0], dtype=np.uint8)    # Yellow
        fn_color = np.array([255, 0, 0], dtype=np.uint8)      # Red

    # Apply error visualizations with different alpha for clarity
    error_masks = [
        (false_negative, fn_color, alpha),      # Missed trees (most critical)
        (false_positive, fp_color, alpha * 0.8), # False alarms
        (true_positive, tp_color, alpha * 0.6)   # Correct (least intrusive)
    ]

    for mask, color, mask_alpha in error_masks:
        if np.any(mask):
            vis[mask] = ((1 - mask_alpha) * vis[mask].astype(np.float32) + 
                        mask_alpha * color.astype(np.float32)).astype(np.uint8)

    return vis


def plot_enhanced_segmentation_analysis(
    image: Union[np.ndarray, torch.Tensor],
    prediction: Union[np.ndarray, torch.Tensor],
    ground_truth: Union[np.ndarray, torch.Tensor],
    sample_id: str = "Sample",
    figsize: Tuple[int, int] = (20, 10),
    id2label: Optional[Dict[int, str]] = None,
    include_boundary_analysis: bool = True,
    include_error_decomposition: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive enhanced segmentation analysis with multiple visualization panels.

    Args:
        image: Original image
        prediction: Predicted segmentation mask
        ground_truth: Ground truth segmentation mask
        sample_id: Identifier for the sample
        figsize: Figure size
        id2label: Dictionary mapping class indices to names
        include_boundary_analysis: Include boundary IoU visualization
        include_error_decomposition: Include TP/FP/FN error analysis
        save_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    # Convert tensors to numpy
    if isinstance(image, torch.Tensor):
        image = tensor_to_image(image)
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()

    # Ensure image is in HWC format
    if len(image.shape) == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))

    # Determine layout based on what's included
    num_panels = 3  # Always have: Original, GT Enhanced, Pred Enhanced
    if include_error_decomposition:
        num_panels += 1
    if include_boundary_analysis:
        num_panels += 1

    # Create figure with appropriate layout
    if num_panels <= 3:
        fig, axes = plt.subplots(1, num_panels, figsize=figsize)
    elif num_panels == 4:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
    else:  # 5 panels
        fig, axes = plt.subplots(2, 3, figsize=(figsize[0], figsize[1] * 0.8))
        axes = axes.flatten()

    panel_idx = 0

    # Panel 1: Original Image
    axes[panel_idx].imshow(image)
    axes[panel_idx].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[panel_idx].axis('off')
    panel_idx += 1

    # Panel 2: Enhanced Ground Truth
    gt_vis = visualize_segmentation_enhanced(
        image.copy(), ground_truth, id2label=id2label, 
        alpha=0.6, use_natural_colors=True, high_contrast=True
    )
    axes[panel_idx].imshow(gt_vis)
    axes[panel_idx].set_title("Ground Truth\n(Enhanced Visualization)", fontsize=14, fontweight='bold')
    axes[panel_idx].axis('off')
    panel_idx += 1

    # Panel 3: Enhanced Prediction
    pred_vis = visualize_segmentation_enhanced(
        image.copy(), prediction, id2label=id2label,
        alpha=0.6, use_natural_colors=True, high_contrast=True
    )
    axes[panel_idx].imshow(pred_vis)
    axes[panel_idx].set_title("Prediction\n(Enhanced Visualization)", fontsize=14, fontweight='bold')
    axes[panel_idx].axis('off')
    panel_idx += 1

    # Panel 4: Error Decomposition (if requested)
    if include_error_decomposition:
        error_vis = visualize_error_decomposition(
            image.copy(), prediction, ground_truth, 
            alpha=0.7, use_natural_colors=True
        )
        axes[panel_idx].imshow(error_vis)
        axes[panel_idx].set_title("Error Analysis\n(Green: TP, Red: FP, Yellow: FN)", 
                                fontsize=14, fontweight='bold')
        axes[panel_idx].axis('off')
        panel_idx += 1

    # Panel 5: Boundary Analysis (if requested)
    if include_boundary_analysis:
        boundary_vis = visualize_boundary_iou_components(
            image.copy(), ground_truth, prediction,
            dilation_pixels=5, alpha=0.8
        )
        axes[panel_idx].imshow(boundary_vis)
        axes[panel_idx].set_title("Boundary Analysis\n(Green: TP, Yellow: FP, Red: FN)", 
                                fontsize=14, fontweight='bold')
        axes[panel_idx].axis('off')
        panel_idx += 1

    # Hide unused panels if any
    for i in range(panel_idx, len(axes)):
        axes[i].set_visible(False)

    # Add overall title
    fig.suptitle(f"{sample_id} - Enhanced Segmentation Analysis", 
                fontsize=16, fontweight='bold', y=0.95)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig


def create_confidence_visualization(
    image: np.ndarray,
    prediction_logits: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    alpha: float = 0.6,
    colormap: str = 'RdYlGn'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create confidence-based visualization for binary segmentation.

    Args:
        image: Original image (H, W, 3)
        prediction_logits: Raw prediction logits or probabilities (H, W) or (2, H, W)
        threshold: Threshold for binary prediction
        alpha: Transparency for confidence overlay
        colormap: Colormap for confidence visualization

    Returns:
        Tuple of (confidence_overlay, confidence_map)
    """
    if isinstance(prediction_logits, torch.Tensor):
        prediction_logits = prediction_logits.detach().cpu().numpy()

    # Handle different input formats
    if len(prediction_logits.shape) == 3 and prediction_logits.shape[0] == 2:
        # Softmax probabilities for binary case
        probs = torch.softmax(torch.from_numpy(prediction_logits), dim=0).numpy()
        confidence_map = np.max(probs, axis=0)  # Max probability
    elif len(prediction_logits.shape) == 2:
        # Single channel - assume sigmoid output or raw scores
        if prediction_logits.min() >= 0 and prediction_logits.max() <= 1:
            # Already probabilities
            probs = prediction_logits
        else:
            # Apply sigmoid to get probabilities
            probs = 1 / (1 + np.exp(-prediction_logits))
        
        # For binary case, confidence is distance from decision boundary
        confidence_map = np.maximum(probs, 1 - probs)
    else:
        raise ValueError(f"Unexpected prediction_logits shape: {prediction_logits.shape}")

    # Ensure image is uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

    # Resize confidence map if needed
    if image.shape[:2] != confidence_map.shape:
        from PIL import Image as PILImage
        conf_img = PILImage.fromarray((confidence_map * 255).astype(np.uint8))
        conf_img = conf_img.resize((image.shape[1], image.shape[0]), resample=PILImage.BILINEAR)
        confidence_map = np.array(conf_img) / 255.0

    # Create confidence overlay
    vis = image.copy()
    
    # Apply colormap to confidence
    cmap = plt.cm.get_cmap(colormap)
    confidence_colored = cmap(confidence_map)[:, :, :3]  # Remove alpha channel
    confidence_colored = (confidence_colored * 255).astype(np.uint8)

    # Blend with original image
    confidence_overlay = ((1 - alpha) * vis.astype(np.float32) + 
                         alpha * confidence_colored.astype(np.float32)).astype(np.uint8)

    return confidence_overlay, confidence_map
