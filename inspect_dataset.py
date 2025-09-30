#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to inspect TCD dataset samples and verify segmentation masks.
This script also examines raw annotations from the TCD dataset before any processing.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
from typing import Union, Optional
from datasets import load_dataset
from transformers import SegformerImageProcessor
from dataset import TCDDataset, create_augmentation_transform, load_and_shuffle_dataset 

from visualization import (plot_augmented_samples, visualize_class_distribution, 
                           prepare_and_visualize_augmentations, visualize_segmentation, 
                           create_pseudocolor)

from config import Config
from PIL import Image

from utils import LOGGER_NAME, get_logger, log_or_print
from exceptions import (
    DatasetError, InvalidSampleError, EmptyMaskError, 
    ShapeMismatchError, ClassImbalanceError
)

# Set up module logger
logger = get_logger()

def examine_raw_annotations(dataset_name="restor/tcd", num_samples=3, save_dir="./raw_annotations", seed=None, enhanced_vis=True, dataset_dict=None, logger=None, is_notebook=False):
    """
    Examine raw annotations from the dataset before any processing.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace Hub
        num_samples: Number of samples to examine
        save_dir: Directory to save visualizations
        seed: Random seed (only used if dataset_dict is None)
        enhanced_vis: Use enhanced visualization techniques
        dataset_dict: Pre-loaded and shuffled dataset (if provided, dataset_name is ignored)
        logger: Optional logger for logging messages
        is_notebook: Whether running in a notebook environment
    """
    # Create seed-specific save directory to compare results across seeds
    random_seed = 42 if seed is None else seed
    seed_dir = os.path.join(save_dir, f"seed_{random_seed}")
    os.makedirs(seed_dir, exist_ok=True)
    
    log_or_print(f"\n{'='*20} EXAMINING RAW ANNOTATIONS WITH SEED {random_seed} {'='*20}", logger, logging.INFO, is_notebook)
    
    try:
        # If dataset_dict is provided, use it (pre-shuffled) instead of loading a new one
        if dataset_dict is not None:
            dataset = dataset_dict
            log_or_print(f"Using provided pre-shuffled dataset for raw annotations (no additional shuffling)", logger, logging.INFO, is_notebook)
        else:
            # Load dataset using the helper function from dataset.py
            dataset = load_and_shuffle_dataset(dataset_name, seed=random_seed)
            log_or_print(f"Loaded and shuffled dataset with seed {random_seed}", logger, logging.INFO, is_notebook)
        
        # Check if train split exists
        if "train" not in dataset:
            raise DatasetError(f"Dataset {dataset_name} does not have a 'train' split")
        
        # Check if annotation feature exists
        if "annotation" not in dataset["train"].features:
            raise DatasetError(f"Dataset {dataset_name} does not have an 'annotation' feature")
    except Exception as e:
        if isinstance(e, DatasetError):
            log_or_print(f"Dataset error: {str(e)}", logger, logging.ERROR, is_notebook)
        else:
            log_or_print(f"Failed to load dataset: {str(e)}", logger, logging.ERROR, is_notebook)
        raise
    
    # Examine samples
    for i in range(min(num_samples, len(dataset["train"]))):
        try:
            # Get sample
            sample = dataset["train"][i]
            
            # Get image and annotation
            image = sample["image"]
            annotation = sample["annotation"]

            # Convert to numpy arrays
            image_np = np.array(image)
            annotation_np = np.array(annotation)
            
            # Log information
            log_or_print(f"Sample {i}:", logger, logging.INFO, is_notebook)
            log_or_print(f"  Image shape: {image_np.shape}", logger, logging.INFO, is_notebook)
            log_or_print(f"  Image mode: {image.mode if hasattr(image, 'mode') else 'Unknown'}", logger, logging.INFO, is_notebook)
            log_or_print(f"  Annotation shape: {annotation_np.shape}", logger, logging.INFO, is_notebook)
            log_or_print(f"  Annotation dtype: {annotation_np.dtype}", logger, logging.INFO, is_notebook)
            log_or_print(f"  Unique annotation values: {np.unique(annotation_np)}", logger, logging.INFO, is_notebook)
            
            # Count annotation values
            unique_values = np.unique(annotation_np)
            total_pixels = annotation_np.size
            
            # Check for empty or single-value masks
            if len(unique_values) <= 1:
                log_or_print(f"  WARNING: Sample {i} has only {len(unique_values)} unique value(s) in annotation", 
                          logger, logging.WARNING, is_notebook)
            
            log_or_print(f"  Annotation value counts:", logger, logging.INFO, is_notebook)
            for value in unique_values:
                count = np.sum(annotation_np == value)
                percentage = (count / total_pixels) * 100
                log_or_print(f"    Value {value}: {count} pixels ({percentage:.2f}%)", logger, logging.INFO, is_notebook)
        
            # Prepare annotations for visualization 
            try:
                if len(annotation_np.shape) > 2:
                    # If it's a 3D array (like RGB), convert to 2D for visualization
                    annotation_np_2d = annotation_np[:,:,0]  # Take first channel to preserve values
                else:
                    annotation_np_2d = annotation_np
                    
                # Create binary version
                binary_annotation = (annotation_np > 0).astype(np.int32)
                if len(binary_annotation.shape) > 2:
                    binary_annotation = binary_annotation[:,:,0]
    
                # Visualize
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                # Display image
                axes[0].imshow(image_np)
                axes[0].set_title(f"Image (Seed {random_seed}, Sample {i})")
                axes[0].axis("off")
                
                # Display raw annotation with enhanced colors to show value differences
                if enhanced_vis:
                    im = axes[1].imshow(annotation_np_2d, cmap='plasma')
                    axes[1].set_title(f"Raw Annotation (Seed {random_seed})")
                    # Add small colorbar to show the range of values
                    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
                else:
                    axes[1].imshow(annotation_np_2d, cmap='viridis')
                    axes[1].set_title(f"Raw Annotation (Seed {random_seed})")
                axes[1].axis("off")
                
                # Display binary annotation
                axes[2].imshow(binary_annotation, cmap='gray')
                axes[2].set_title("Binary Annotation (always [0,1])")
                axes[2].axis("off")
                
                plt.tight_layout()
                # Save only to seed-specific directory with consistent naming
                plt.savefig(os.path.join(seed_dir, f"seed_{random_seed}_sample_{i}.png"), dpi=300, bbox_inches="tight")
                plt.close()
                
                # Create enhanced visualization of raw annotation values with colorbar
                fig, ax = plt.subplots(figsize=(10, 10))
                im = ax.imshow(annotation_np_2d, cmap='plasma')
                ax.set_title(f"Raw Annotation Values (Seed {random_seed}, Sample {i})")
                ax.axis("off")
                
                # Add a colorbar to show the value mapping
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label("Annotation Value")
                
                plt.tight_layout()
                plt.savefig(os.path.join(seed_dir, f"seed_{random_seed}_raw_annotation_{i}.png"), dpi=300, bbox_inches="tight")
                plt.close(fig) # Close this figure
                
                # Save binary annotation as image
                binary_img = Image.fromarray((binary_annotation * 255).astype(np.uint8))
                binary_img.save(os.path.join(seed_dir, f"seed_{random_seed}_binary_annotation_{i}.png"))
                
                # If enhanced visualization is enabled, create a pseudocolor representation
                if enhanced_vis:
                    try:
                        # Create a pseudocolor binary visualization based on the seed
                        hsv = np.zeros((binary_annotation.shape[0], binary_annotation.shape[1], 3), dtype=np.float32)
                        
                        # Use the seed to determine the hue (color)
                        hue = (random_seed % 360) / 360.0  # Convert seed to a value between 0-1
                        
                        # Set the hue (color) based on seed, saturation to max (1.0), 
                        # and value/brightness based on whether it's foreground (1.0) or background (0.2)
                        hsv[..., 0] = hue  # Hue
                        hsv[..., 1] = 1.0  # Saturation
                        hsv[..., 2] = binary_annotation * 0.8 + 0.2  # Value (brightness)
                        
                        # Convert HSV to RGB for display
                        import matplotlib.colors as mcolors
                        rgb = mcolors.hsv_to_rgb(hsv)
                        
                        # Save the pseudocolor representation
                        plt.figure(figsize=(10, 10))
                        plt.imshow(rgb)
                        plt.title(f"Pseudocolor Binary (Seed {random_seed}, Sample {i})")
                        plt.axis('off')
                        plt.tight_layout()
                        plt.savefig(os.path.join(seed_dir, f"seed_{random_seed}_pseudocolor_binary_{i}.png"), dpi=300, bbox_inches="tight")
                        plt.close(plt.gcf()) # Close the current figure
                    except Exception as e:
                        log_or_print(f"Error creating pseudocolor visualization for sample {i}: {str(e)}", 
                                  logger, logging.WARNING, is_notebook)
            except Exception as vis_e:
                log_or_print(f"Visualization error for sample {i}: {str(vis_e)}", 
                          logger, logging.WARNING, is_notebook)
                # Try to save a basic fallback visualization
                try:
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(image_np)
                    plt.title(f"Image (Sample {i})")
                    plt.axis('off')
                    plt.subplot(1, 2, 2)
                    plt.imshow(annotation_np_2d, cmap='gray')
                    plt.title(f"Annotation (Sample {i}) - Fallback")
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(os.path.join(seed_dir, f"seed_{random_seed}_sample_{i}_fallback.png"), 
                               dpi=300, bbox_inches="tight")
                    plt.close()
                except Exception:
                    log_or_print(f"Failed to create fallback visualization for sample {i}", 
                              logger, logging.ERROR, is_notebook)
                    
        except Exception as e:
            log_or_print(f"Error processing sample {i}: {str(e)}", logger, logging.ERROR, is_notebook)
            # Continue with next sample

def examine_dataset_statistics(dataset, num_samples=100, save_dir="./dataset_statistics", logger=None, is_notebook=False):
    """
    Generate comprehensive statistics about the dataset.
    
    Args:
        dataset: A TCDDataset instance
        num_samples: Number of samples to analyze
        save_dir: Directory to save statistics and visualizations
        logger: Optional logger for logging messages
        is_notebook: Whether running in a notebook environment
        
    Returns:
        Dictionary of statistics
    """
    os.makedirs(save_dir, exist_ok=True)
    
    log_or_print(f"Analyzing dataset statistics using {min(num_samples, len(dataset))} samples...",
               logger, logging.INFO, is_notebook)
    
    stats = {
        "class_distribution": {},
        "mask_sizes": [],
        "mask_quality": {
            "empty_masks": 0,
            "single_class_masks": 0,
            "imbalanced_masks": 0,
            "valid_masks": 0
        },
        "class_percentages": {},
        "dataset_size": len(dataset)
    }
    
    # Class counts across all analyzed samples
    total_pixel_counts = {}
    
    start_time = time.time()
    
    for i in range(min(num_samples, len(dataset))):
        try:
            sample = dataset[i]
            labels = sample['labels']
            
            # Convert to numpy if it's a tensor
            if isinstance(labels, torch.Tensor):
                labels_np = labels.numpy()
            else:
                labels_np = np.array(labels)
            
            # Get unique labels and counts
            unique_labels, counts = np.unique(labels_np, return_counts=True)
            
            # Record mask size
            stats["mask_sizes"].append(labels_np.size)
            
            # Check mask quality
            if len(unique_labels) == 0:
                stats["mask_quality"]["empty_masks"] += 1
            elif len(unique_labels) == 1:
                stats["mask_quality"]["single_class_masks"] += 1
            else:
                # Check for class imbalance
                total_pixels = labels_np.size
                max_class_percentage = (np.max(counts) / total_pixels) * 100
                
                if max_class_percentage > 99.0:  # If one class dominates more than 99%
                    stats["mask_quality"]["imbalanced_masks"] += 1
                else:
                    stats["mask_quality"]["valid_masks"] += 1
                
            # Update total pixel counts for each class
            for label, count in zip(unique_labels, counts):
                label_key = int(label)  # Ensure it's an integer for JSON
                if label_key not in total_pixel_counts:
                    total_pixel_counts[label_key] = 0
                total_pixel_counts[label_key] += count
                
            # Log progress periodically
            if (i+1) % 20 == 0:
                elapsed = time.time() - start_time
                log_or_print(f"Processed {i+1}/{min(num_samples, len(dataset))} samples ({elapsed:.2f}s elapsed)",
                           logger, logging.INFO, is_notebook)
                
        except Exception as e:
            log_or_print(f"Error analyzing sample {i}: {str(e)}", logger, logging.WARNING, is_notebook)
    
    # Calculate overall class distribution
    total_pixels = sum(total_pixel_counts.values())
    stats["class_distribution"] = {str(cls): count for cls, count in total_pixel_counts.items()}
    stats["class_percentages"] = {str(cls): (count / total_pixels) * 100 
                                for cls, count in total_pixel_counts.items()}
    
    # Calculate summary statistics
    stats["summary"] = {
        "total_samples_analyzed": min(num_samples, len(dataset)),
        "valid_sample_percentage": (stats["mask_quality"]["valid_masks"] / min(num_samples, len(dataset))) * 100,
        "avg_mask_size": np.mean(stats["mask_sizes"]) if stats["mask_sizes"] else 0,
        "class_count": len(total_pixel_counts)
    }
    
    # Save statistics to JSON file
    try:
        with open(os.path.join(save_dir, "dataset_statistics.json"), "w") as f:
            json.dump(stats, f, indent=2)
        log_or_print(f"Statistics saved to {os.path.join(save_dir, 'dataset_statistics.json')}",
                   logger, logging.INFO, is_notebook)
    except Exception as e:
        log_or_print(f"Error saving statistics: {str(e)}", logger, logging.ERROR, is_notebook)
    
    # Generate class distribution visualization
    try:
        if stats["class_percentages"]:
            # Create a mapping from class indices to names (use defaults for TCD)
            id2label = {0: "background", 1: "tree_crown"}
            
            # Create the visualization
            fig = visualize_class_distribution(
                class_distribution=stats["class_percentages"],
                id2label=id2label,
                title='Class Distribution in Dataset',
                save_path=os.path.join(save_dir, "class_distribution.png")
            )
            plt.close(fig)
            log_or_print(f"Class distribution visualization saved to {os.path.join(save_dir, 'class_distribution.png')}",
                       logger, logging.INFO, is_notebook)
    except Exception as e:
        log_or_print(f"Error creating class distribution visualization: {str(e)}", 
                   logger, logging.WARNING, is_notebook)
    
    elapsed = time.time() - start_time
    log_or_print(f"Dataset statistics analysis completed in {elapsed:.2f}s", 
               logger, logging.INFO, is_notebook)
    
    return stats

def inspect_dataset_samples(dataset_or_dict=None, num_samples=3, save_dir="./dataset_inspection", max_attempts=15, seed=None, enhanced_vis=True, image_processor=None, split="train", plot_return=False, dataset_name=None, logger=None, is_notebook=False):
    """
    Inspect dataset samples and visualize images and segmentation masks.
    
    This function takes samples from the dataset, extracts the images and their
    corresponding segmentation masks, and visualizes them side by side. It also
    reports statistics about the segmentation masks, such as the distribution of
    label values, which is useful for identifying class imbalance issues.
    
    For the TCD dataset, the segmentation masks are binary (0: background, 1: tree_crown).
    
    Args:
        dataset_or_dict: Either a TCDDataset instance or a dataset dictionary.
                        No additional shuffling will be applied if a dataset dictionary is provided.
        num_samples: Number of samples to inspect
        save_dir: Directory to save visualizations
        max_attempts: Maximum number of attempts to find valid samples
        seed: Random seed (used for visualization variations only)
        enhanced_vis: Use enhanced visualization techniques
        image_processor: SegformerImageProcessor instance (required if dataset_or_dict is a dataset dictionary)
        split: Dataset split to use when dataset_or_dict is a dictionary (default: "train")
        plot_return: If True, return plot figures instead of saving them (default: False)
        logger: Optional logger for logging messages
        is_notebook: Whether running in a notebook environment
    """
    # Create seed-specific save directory
    random_seed = 42 if seed is None else seed
    seed_dir = os.path.join(save_dir, f"seed_{random_seed}")
    os.makedirs(seed_dir, exist_ok=True)
    
    log_or_print(f"\n{'='*20} INSPECTING PROCESSED SAMPLES WITH SEED {random_seed} {'='*20}", 
               logger, logging.INFO, is_notebook)

    # --- Augmentation Setup ---
    # Create a dummy config or load one if needed for augmentation settings
    # For inspection, we might just create a default config
    config = Config() # Use default config for inspection
    train_transform = create_augmentation_transform(config)
    visualize_augmentations = config.get("augmentation", {}).get("visualize_augmented_samples", False)
    num_aug_to_show = config.get("augmentation", {}).get("num_augmented_samples_to_visualize", 4)
    augmentation_plotted = False # Flag to plot only once
    
    # To maintain compatibility with existing code that might pass dataset_name as the first parameter:
    # If dataset_or_dict is not None and dataset_name is None, check if it's a string
    if isinstance(dataset_or_dict, str) and dataset_name is None:
        dataset_name = dataset_or_dict
        dataset_or_dict = None

    # Handle the case where dataset_name is provided
    if dataset_or_dict is None and dataset_name is not None:
        # Load the dataset using load_and_shuffle_dataset
        from dataset import load_and_shuffle_dataset
        dataset_dict = load_and_shuffle_dataset(dataset_name, seed=seed)
        dataset_or_dict = dataset_dict
        log_or_print(f"Loaded dataset '{dataset_name}' with seed {seed}", logger, logging.INFO, is_notebook)

    # Check if input is a dataset dictionary rather than a TCDDataset
    if not isinstance(dataset_or_dict, TCDDataset):
        if image_processor is None:
            # Create a default image processor if not provided
            image_processor = SegformerImageProcessor(
                do_resize=False,
                do_rescale=True,
                do_normalize=True
            )
            log_or_print("Created default image processor with do_resize=False, do_rescale=True, do_normalize=True", logger, logging.INFO, is_notebook)
            
        # It's a dataset dictionary, create dataset without shuffling
        dataset_dict = dataset_or_dict
        log_or_print(f"Using provided dataset for processing (no additional shuffling)", logger, logging.INFO, is_notebook)
            
        # Create dataset - Pass the transform if it's the training split
        # Create dataset - Pass the transform if it's the training split
        dataset_transform = train_transform if split == "train" else None
        # Pass the config object to the constructor
        dataset = TCDDataset(dataset_dict, image_processor, config, split=split, transform=dataset_transform)
    else:
        # It's already a TCDDataset - it might already have a transform
        dataset = dataset_or_dict
        # If we want to visualize augmentations, ensure the dataset's transform is used
        if visualize_augmentations and dataset.transform:
             train_transform = dataset.transform # Use the dataset's existing transform
        elif visualize_augmentations and not dataset.transform:
             log_or_print("Augmentation visualization requested, but dataset has no transform.", logger, logging.WARNING, is_notebook)
             visualize_augmentations = False


    # If all samples have class imbalance, we still want to inspect some
    # So we'll try a limited number of indices and accept what we get
    samples_inspected = 0
    attempts = 0
    sample_fig = None  # Initialize to handle case when no samples are processed
    
    # Try different indices until we've inspected enough samples or reached max attempts
    while samples_inspected < num_samples and attempts < max_attempts:
        try:
            # Get a sample
            idx = attempts % len(dataset)
            sample = dataset[idx]
            
            # Extract image and mask tensors
            pixel_values = sample['pixel_values']
            labels = sample['labels'] # This is the mask *after* image_processor

            # --- Visualize Augmentation (only for the first valid sample) ---
            if visualize_augmentations and train_transform and not augmentation_plotted:
                # Create the id2label mapping (default for TCD binary segmentation)
                id2label = {0: "background", 1: "tree_crown"}
                
                # Define the plot path
                aug_plot_path = os.path.join(seed_dir, f"seed_{random_seed}_augmented_sample_{idx}.png")
                
                # Use the centralized helper function 
                success = prepare_and_visualize_augmentations(
                    dataset=dataset,
                    sample_idx=idx,
                    transform=train_transform,
                    save_path=aug_plot_path,
                    num_augmented=num_aug_to_show,
                    logger=logger,
                    id2label=id2label
                )
                
                # Only mark as plotted if successful
                augmentation_plotted = success

            # --- Continue with inspecting the processed sample ---
            # Log shapes and value ranges
            log_or_print(f"Successfully loaded sample {idx} (attempt {attempts+1})", logger, logging.INFO, is_notebook)
            log_or_print(f"  Pixel values shape: {pixel_values.shape}", logger, logging.INFO, is_notebook)
            log_or_print(f"  Labels shape: {labels.shape}", logger, logging.INFO, is_notebook)
            log_or_print(f"  Pixel values range: [{pixel_values.min()}, {pixel_values.max()}]", logger, logging.INFO, is_notebook)
            log_or_print(f"  Unique label values: {torch.unique(labels).tolist()}", logger, logging.INFO, is_notebook)
            
            # Count label distribution
            unique_labels = torch.unique(labels)
            label_counts = [(label.item(), (labels == label).sum().item()) for label in unique_labels]
            total_pixels = labels.numel()
            
            log_or_print(f"  Label counts:", logger, logging.INFO, is_notebook)
            for label, count in label_counts:
                percentage = (count / total_pixels) * 100
                log_or_print(f"    Label {label}: {count} pixels ({percentage:.2f}%)", logger, logging.INFO, is_notebook)
            
            # Convert tensor to numpy for visualization
            img = pixel_values.permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
            
            # Create labels numpy version
            labels_np = labels.numpy()
            
            # Visualize standard version
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            axes[0].imshow(img)
            axes[0].set_title(f"Image (Seed {random_seed})")
            axes[0].axis("off")
            
            # Use binary colormap for segmentation mask since it's binary (0: background, 1: tree_crown)
            axes[1].imshow(labels_np, cmap='binary')
            axes[1].set_title(f"Segmentation Mask (Seed {random_seed})")
            axes[1].axis("off")
            
            plt.tight_layout()
            # Create a figure to return if requested
            sample_fig = fig if plot_return else None
            
            # Save unless plot_return is True
            if not plot_return:
                plt.savefig(os.path.join(seed_dir, f"seed_{random_seed}_sample_{idx}.png"), dpi=300, bbox_inches="tight")
                plt.close()
            
            # Create enhanced visualizations if enabled
            if enhanced_vis:
                # Ensure numpy is available in this scope
                import numpy as np
                # Create pseudocolor version of binary mask based on seed
                # This makes the output visually different for different seeds
                import matplotlib.colors as mcolors
                
                # Create HSV representation where hue is determined by seed
                hsv = np.zeros((labels_np.shape[0], labels_np.shape[1], 3), dtype=np.float32)
                
                # Vary the hue (color) based on the seed
                hue = (random_seed % 360) / 360.0
                
                # Set HSV channels: hue from seed, saturation at max, value from labels
                hsv[..., 0] = hue  # Hue from seed
                hsv[..., 1] = 1.0  # Full saturation
                hsv[..., 2] = labels_np * 0.8 + 0.2  # Value
                
                # Convert HSV to RGB
                rgb = mcolors.hsv_to_rgb(hsv)
                
                # Create a figure with the pseudocolor mask
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                
                axes[0].imshow(img)
                axes[0].set_title(f"Image (Seed {random_seed})")
                axes[0].axis("off")
                
                axes[1].imshow(rgb)
                axes[1].set_title(f"Pseudocolor Mask (Seed {random_seed})")
                axes[1].axis("off")
                
                plt.tight_layout()
                plt.savefig(os.path.join(seed_dir, f"seed_{random_seed}_pseudocolor_sample_{idx}.png"), dpi=300, bbox_inches="tight")
                plt.close()
            
            samples_inspected += 1
        except Exception as e:
            log_or_print(f"Error processing sample {idx}: {str(e)}", logger, logging.ERROR, is_notebook)
        
        attempts += 1
    
    if samples_inspected == 0:
        log_or_print(f"Could not find any valid samples after {max_attempts} attempts.", logger, logging.WARNING, is_notebook)
    else:
        log_or_print(f"Successfully inspected {samples_inspected} samples after {attempts} attempts.", logger, logging.INFO, is_notebook)
    
    # If plot_return is True, return the sample figure and generate a class distribution figure
    if plot_return:
        return_data = {
            "sample_fig": sample_fig
        }
        
        # Create class distribution figure
        try:
            # Calculate class distribution
            class_counts = {}
            for i in range(min(num_samples * 2, len(dataset))):
                try:
                    sample = dataset[i]
                    labels = sample['labels']
                    
                    # Get unique labels and counts
                    if isinstance(labels, torch.Tensor):
                        unique_labels = torch.unique(labels, return_counts=True)
                        for label, count in zip(unique_labels[0].tolist(), unique_labels[1].tolist()):
                            if label not in class_counts:
                                class_counts[label] = 0
                            class_counts[label] += count
                    else:
                        # Ensure numpy is available in this scope
                        import numpy as np
                        # If it's a PIL Image, convert to numpy first
                        if hasattr(labels, 'mode'):
                            labels = np.array(labels)
                        unique_labels, counts = np.unique(labels, return_counts=True)
                        for label, count in zip(unique_labels, counts):
                            if label not in class_counts:
                                class_counts[label] = 0
                            class_counts[label] += count
                except Exception as e:
                    log_or_print(f"Error getting class distribution for sample {i}: {e}", logger, logging.ERROR, is_notebook)

            if class_counts:
                # Create id2label mapping
                id2label = {0: "background", 1: "tree_crown"}  # Default for TCD

                # Calculate percentages
                total_pixels = sum(class_counts.values())
                class_dist_perc = {cls: count / total_pixels * 100 for cls, count in class_counts.items()}

                # Call the centralized visualization function
                class_dist_fig = visualize_class_distribution(
                    class_distribution=class_dist_perc,
                    id2label=id2label,
                    title='Class Distribution in Dataset'
                    # save_path can be added if needed
                )
                return_data["class_dist_fig"] = class_dist_fig
            else:
                log_or_print("No class counts calculated, skipping distribution plot.", logger, logging.WARNING, is_notebook)
                return_data["class_dist_fig"] = None

        except Exception as e:
            log_or_print(f"Error creating class distribution figure: {e}", logger, logging.ERROR, is_notebook)
            return_data["class_dist_fig"] = None

        return return_data


def verify_training_tiling(
    config: Union[Config, str],
    num_samples: int = 5,
    visualize: bool = False,
    logger: Optional[logging.Logger] = None,
    is_notebook: bool = False
) -> bool:
    """
    Verifies that the training dataset loader is producing tiles of the correct size.

    Args:
        config: A Config object or path to a config JSON file.
        num_samples: Number of samples to check.
        visualize: Whether to visualize the fetched tiles.
        logger: Optional logger instance.
        is_notebook: Whether running in a notebook environment.

    Returns:
        True if all checked samples have the correct tile size, False otherwise.
    """
    if logger is None:
        logger = get_logger() # Use default logger if none provided

    log_or_print(f"\n{'='*20} VERIFYING TRAINING TILING {'='*20}", logger, logging.INFO, is_notebook)

    # --- Load Configuration ---
    if isinstance(config, str):
        try:
            config_obj = Config.load(config)
            log_or_print(f"Loaded configuration from: {config}", logger, logging.INFO, is_notebook)
        except Exception as e:
            log_or_print(f"Error loading configuration from {config}: {e}", logger, logging.ERROR, is_notebook)
            return False
    elif isinstance(config, Config):
        config_obj = config
        log_or_print("Using provided Config object.", logger, logging.INFO, is_notebook)
    else:
        log_or_print("Invalid config provided. Must be a Config object or path string.", logger, logging.ERROR, is_notebook)
        return False

    # --- Get Expected Tile Size ---
    train_tile_size = config_obj.get("train_tile_size")
    if train_tile_size is None:
        log_or_print("Configuration does not specify 'train_tile_size'. Tiling verification skipped.", logger, logging.WARNING, is_notebook)
        # Technically not a failure, but tiling isn't configured. Return True? Or False? Let's return True as verification isn't applicable.
        return True

    try:
        if isinstance(train_tile_size, int):
            expected_h, expected_w = train_tile_size, train_tile_size
        elif isinstance(train_tile_size, (list, tuple)) and len(train_tile_size) == 2:
            expected_h, expected_w = train_tile_size
        else:
            raise ValueError(f"Invalid train_tile_size format: {train_tile_size}")
        log_or_print(f"Expected tile size (H x W): {expected_h} x {expected_w}", logger, logging.INFO, is_notebook)
    except ValueError as e:
        log_or_print(f"Error parsing train_tile_size: {e}", logger, logging.ERROR, is_notebook)
        return False

    # --- Load Dataset ---
    try:
        dataset_name = config_obj.get("dataset_name", "restor/tcd")
        seed = config_obj.get("seed", 42)
        log_or_print(f"Loading dataset '{dataset_name}' with seed {seed}...", logger, logging.INFO, is_notebook)
        # Use load_and_shuffle_dataset to ensure consistency
        dataset_dict = load_and_shuffle_dataset(dataset_name, seed=seed)
    except Exception as e:
        log_or_print(f"Error loading dataset: {e}", logger, logging.ERROR, is_notebook)
        return False

    # --- Create Image Processor ---
    # Use default processor settings consistent with training (no resize)
    try:
        image_processor = SegformerImageProcessor(
            do_resize=False,
            do_rescale=True,
            do_normalize=True
        )
        log_or_print("Created default SegformerImageProcessor.", logger, logging.INFO, is_notebook)
    except Exception as e:
        log_or_print(f"Error creating Image Processor: {e}", logger, logging.ERROR, is_notebook)
        return False

    # --- Instantiate Training Dataset ---
    try:
        # Create augmentation transform based on config (though not strictly needed for shape check)
        train_transform = create_augmentation_transform(config_obj)
        # Instantiate the dataset for the 'train' split
        train_dataset = TCDDataset(
            dataset_dict=dataset_dict,
            image_processor=image_processor,
            config=config_obj, # Pass the config object
            split="train",
            transform=train_transform # Apply transform if configured
        )
        log_or_print(f"Instantiated TCDDataset for 'train' split with {len(train_dataset)} samples.", logger, logging.INFO, is_notebook)
    except Exception as e:
        log_or_print(f"Error instantiating TCDDataset: {e}", logger, logging.ERROR, is_notebook)
        return False

    # --- Verify Samples ---
    all_passed = True
    num_to_check = min(num_samples, len(train_dataset))
    if num_to_check == 0:
        log_or_print("Training dataset is empty. Cannot verify tiling.", logger, logging.WARNING, is_notebook)
        return False # Consider empty dataset a failure for verification?

    log_or_print(f"Checking shapes for {num_to_check} training samples...", logger, logging.INFO, is_notebook)

    for i in range(num_to_check):
        try:
            sample = train_dataset[i]
            pixel_values = sample.get('pixel_values')
            labels = sample.get('labels')

            if pixel_values is None or labels is None:
                log_or_print(f"Sample {i}: Missing 'pixel_values' or 'labels'. Skipping.", logger, logging.WARNING, is_notebook)
                all_passed = False
                continue

            # Check shapes (pixel_values: C x H x W, labels: H x W)
            actual_h_pixels, actual_w_pixels = pixel_values.shape[1], pixel_values.shape[2]
            actual_h_labels, actual_w_labels = labels.shape[0], labels.shape[1]

            # Verify consistency between pixel_values and labels shapes
            if actual_h_pixels != actual_h_labels or actual_w_pixels != actual_w_labels:
                 log_or_print(f"Sample {i}: FAIL - Mismatch between pixel_values shape ({actual_h_pixels}x{actual_w_pixels}) and labels shape ({actual_h_labels}x{actual_w_labels})", logger, logging.ERROR, is_notebook)
                 all_passed = False
                 # Optionally visualize the mismatch
                 if visualize:
                     try:
                         fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                         # Need to handle tensor display
                         img_display = pixel_values.permute(1, 2, 0).numpy()
                         img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min()) # Normalize
                         axes[0].imshow(img_display)
                         axes[0].set_title(f"Sample {i} Pixels ({actual_h_pixels}x{actual_w_pixels})")
                         axes[0].axis("off")
                         axes[1].imshow(labels.numpy(), cmap='gray')
                         axes[1].set_title(f"Sample {i} Labels ({actual_h_labels}x{actual_w_labels})")
                         axes[1].axis("off")
                         plt.suptitle("Shape Mismatch Visualization")
                         plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
                         if is_notebook:
                             plt.show()
                         else:
                             save_path = os.path.join(config_obj.get("output_dir", "."), f"tiling_verification_mismatch_{i}.png")
                             os.makedirs(os.path.dirname(save_path), exist_ok=True)
                             plt.savefig(save_path)
                             log_or_print(f"Saved mismatch visualization to {save_path}", logger, logging.INFO, is_notebook)
                         plt.close(fig)
                     except Exception as vis_e:
                         log_or_print(f"Sample {i}: Error visualizing mismatch: {vis_e}", logger, logging.WARNING, is_notebook)
                 continue # Move to next sample after mismatch

            # Verify against expected tile size
            if actual_h_pixels == expected_h and actual_w_pixels == expected_w:
                log_or_print(f"Sample {i}: PASS - Shape ({actual_h_pixels}x{actual_w_pixels}) matches expected ({expected_h}x{expected_w})", logger, logging.INFO, is_notebook)
                # Optional visualization for passed samples
                if visualize:
                     try:
                         fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                         img_display = pixel_values.permute(1, 2, 0).numpy()
                         img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min()) # Normalize
                         axes[0].imshow(img_display)
                         axes[0].set_title(f"Sample {i} Tile ({actual_h_pixels}x{actual_w_pixels})")
                         axes[0].axis("off")
                         axes[1].imshow(labels.numpy(), cmap='gray')
                         axes[1].set_title(f"Sample {i} Mask ({actual_h_labels}x{actual_w_labels})")
                         axes[1].axis("off")
                         plt.suptitle("Tile Verification PASS")
                         plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                         if is_notebook:
                             plt.show()
                         else:
                             save_path = os.path.join(config_obj.get("output_dir", "."), f"tiling_verification_pass_{i}.png")
                             os.makedirs(os.path.dirname(save_path), exist_ok=True)
                             plt.savefig(save_path)
                             log_or_print(f"Saved PASS visualization to {save_path}", logger, logging.INFO, is_notebook)
                         plt.close(fig)
                     except Exception as vis_e:
                         log_or_print(f"Sample {i}: Error visualizing passed sample: {vis_e}", logger, logging.WARNING, is_notebook)

            else:
                log_or_print(f"Sample {i}: FAIL - Shape ({actual_h_pixels}x{actual_w_pixels}) does NOT match expected ({expected_h}x{expected_w})", logger, logging.ERROR, is_notebook)
                all_passed = False
                # Optional visualization for failed samples
                if visualize:
                     try:
                         fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                         img_display = pixel_values.permute(1, 2, 0).numpy()
                         img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min()) # Normalize
                         axes[0].imshow(img_display)
                         axes[0].set_title(f"Sample {i} Tile ({actual_h_pixels}x{actual_w_pixels}) - FAIL")
                         axes[0].axis("off")
                         axes[1].imshow(labels.numpy(), cmap='gray')
                         axes[1].set_title(f"Sample {i} Mask ({actual_h_labels}x{actual_w_labels}) - FAIL")
                         axes[1].axis("off")
                         plt.suptitle("Tile Verification FAIL")
                         plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                         if is_notebook:
                             plt.show()
                         else:
                             save_path = os.path.join(config_obj.get("output_dir", "."), f"tiling_verification_fail_{i}.png")
                             os.makedirs(os.path.dirname(save_path), exist_ok=True)
                             plt.savefig(save_path)
                             log_or_print(f"Saved FAIL visualization to {save_path}", logger, logging.INFO, is_notebook)
                         plt.close(fig)
                     except Exception as vis_e:
                         log_or_print(f"Sample {i}: Error visualizing failed sample: {vis_e}", logger, logging.WARNING, is_notebook)


        except Exception as e:
            log_or_print(f"Error processing sample {i}: {e}", logger, logging.ERROR, is_notebook)
            all_passed = False
            # Optionally add traceback here for debugging
            # import traceback
            # log_or_print(traceback.format_exc(), logger, logging.ERROR, is_notebook)

    # --- Final Result ---
    if all_passed:
        log_or_print(f"\nVERIFICATION RESULT: PASS - All {num_to_check} checked samples have the expected tile size.", logger, logging.INFO, is_notebook)
    else:
        log_or_print(f"\nVERIFICATION RESULT: FAIL - One or more checked samples did not have the expected tile size.", logger, logging.ERROR, is_notebook)

    return all_passed
