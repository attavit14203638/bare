#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Improved dataset handling for TCD-SegFormer model.

This module provides a more robust and consistent approach to dataset
loading, processing, and validation, centralizing functionality that was
previously scattered across multiple files.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset, Dataset as HFDataset, DatasetDict
from transformers import SegformerImageProcessor
import numpy as np
from typing import Dict, Tuple, List, Optional, Union, Any, Callable, Generator
import logging
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import logging # Added logging import

from config import Config # Import Config
from exceptions import (
    DatasetError, InvalidSampleError, EmptyMaskError,
    ShapeMismatchError, ClassImbalanceError, handle_dataset_error
)
from image_utils import ensure_rgb, ensure_mask_shape
from utils import get_logger # Use get_logger

# Setup module logger
logger = get_logger()


# --- Custom Transform for Combined Augmentations ---

class CombinedTransform:
    """
    A transform that applies geometric transformations to both image and mask,
    while applying color transformations only to the image.
    Ensures consistent random transformations between image and mask.
    """
    def __init__(self, geometric_transforms, color_transforms):
        # geometric_transforms is expected to be a list, so Compose it here.
        self.geometric_transforms = T.Compose(geometric_transforms)
        # color_transforms is expected to be already Composed in create_augmentation_transform.
        self.color_transforms = color_transforms

    def __call__(self, sample: Dict[str, Image.Image]) -> Dict[str, Image.Image]:
        """Applies geometric transforms consistently and color transforms only to image."""
        image = sample['image']
        mask = sample['mask']

        # 1. Apply geometric transforms (flips, rotation) consistently
        # Get random parameters for geometric transforms first
        params = {}
        if isinstance(self.geometric_transforms, T.Compose):
            for t in self.geometric_transforms.transforms:
                if isinstance(t, T.RandomHorizontalFlip):
                    params['hflip'] = torch.rand(1) < t.p
                elif isinstance(t, T.RandomVerticalFlip):
                    params['vflip'] = torch.rand(1) < t.p
                elif isinstance(t, T.RandomRotation):
                    params['angle'] = t.get_params(t.degrees)
                # Add other geometric transforms here if needed
        else:
             logger.warning("Geometric transforms not wrapped in T.Compose, consistency not guaranteed.")

        # Apply transforms using the generated parameters
        if params.get('hflip', False):
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if params.get('vflip', False):
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        if 'angle' in params:
            # Ensure NEAREST interpolation for mask rotation
            image = TF.rotate(image, params['angle'], interpolation=T.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, params['angle'], interpolation=T.InterpolationMode.NEAREST)

        # 2. Apply color transforms only to the image
        image = self.color_transforms(image)

        return {'image': image, 'mask': mask}

# --- Augmentation Factory ---

def create_augmentation_transform(config: Config) -> Optional[Callable]:
    """
    Creates a composition of torchvision transforms for data augmentation based on config.
    Separates geometric (image+mask) and color (image only) transforms.

    Args:
        config: Configuration object containing augmentation settings.

    Returns:
        A callable CombinedTransform object, or None if augmentations are disabled.
    """
    aug_config = config.get("augmentation", {})
    if not aug_config.get("apply", False):
        logger.info("Augmentations disabled via config.")
        return None

    geometric_transforms = []
    color_transforms = []

    # --- Geometric Augmentations (applied to both image and mask) ---
    # Note: RandomCrop is handled separately in TCDDataset.__getitem__
    # Reference: H/V Flip, Rotation (any degree)
    h_flip_prob = aug_config.get("h_flip_prob", 0.5) # Default 0.5 if not specified
    v_flip_prob = aug_config.get("v_flip_prob", 0.5) # Default 0.5 if not specified
    rotation_degrees = aug_config.get("rotation_degrees", 180) # Use 180 as default based on plan

    if h_flip_prob > 0:
        geometric_transforms.append(T.RandomHorizontalFlip(p=h_flip_prob))
        logger.debug(f"Added RandomHorizontalFlip (p={h_flip_prob})")
    if v_flip_prob > 0:
        geometric_transforms.append(T.RandomVerticalFlip(p=v_flip_prob))
        logger.debug(f"Added RandomVerticalFlip (p={v_flip_prob})")
    if rotation_degrees != 0: # Allow disabling rotation by setting degrees to 0
        # Use T.RandomRotation. Interpolation is handled in CombinedTransform.__call__
        geometric_transforms.append(T.RandomRotation(degrees=rotation_degrees))
        logger.debug(f"Added RandomRotation (degrees={rotation_degrees})")

    # --- Color Augmentations (applied only to image) ---
    # Reference: Color Adjust, Blur
    color_jitter_prob = aug_config.get("color_jitter_prob", 0.5) # Apply jitter with 50% probability
    brightness = aug_config.get("brightness", 0.25)
    contrast = aug_config.get("contrast", 0.25)
    saturation = aug_config.get("saturation", 0.25)
    hue = aug_config.get("hue", 0.1)

    if color_jitter_prob > 0:
        jitter = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        # Apply jitter randomly based on probability
        color_transforms.append(T.RandomApply([jitter], p=color_jitter_prob))
        logger.debug(f"Added ColorJitter (p={color_jitter_prob}) with b={brightness}, c={contrast}, s={saturation}, h={hue}")

    # Gaussian Blur (applied only to image)
    gaussian_blur_prob = aug_config.get("gaussian_blur_prob", 0.5) # Apply blur with 50% probability
    kernel_size = aug_config.get("gaussian_blur_kernel_size", (3, 7)) # Range for kernel size
    sigma = aug_config.get("gaussian_blur_sigma", (0.1, 2.0)) # Range for sigma

    if gaussian_blur_prob > 0:
        # Ensure kernel size is a tuple/list of 2 ints for range, or single odd int
        if isinstance(kernel_size, int):
            if kernel_size % 2 == 0:
                logger.warning(f"Gaussian blur kernel size {kernel_size} is even, adjusting to {kernel_size + 1}")
                kernel_size += 1
        elif isinstance(kernel_size, (list, tuple)) and len(kernel_size) == 2:
            # Ensure min/max are odd if specifying range? torchvision handles this.
            pass
        else:
            logger.warning(f"Invalid gaussian_blur_kernel_size: {kernel_size}. Using default (3, 7).")
            kernel_size = (3, 7)

        blurrer = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        color_transforms.append(T.RandomApply([blurrer], p=gaussian_blur_prob))
        logger.debug(f"Added GaussianBlur (p={gaussian_blur_prob}) with kernel={kernel_size}, sigma={sigma}")


    if not geometric_transforms and not color_transforms:
         logger.warning("Augmentation enabled but no specific transforms were configured.")
         return None

    logger.info(f"Created CombinedTransform with {len(geometric_transforms)} geometric and {len(color_transforms)} color transforms.")
    # Use the CombinedTransform class to manage applying transforms correctly
    # Pass the list of geometric transforms directly; it will be composed in CombinedTransform.__init__
    return CombinedTransform(geometric_transforms, T.Compose(color_transforms))


# --- Removed Custom Transform Classes (RandomHorizontalFlip, etc.) ---


# --- Dataset Classes ---



class BaseSegmentationDataset(Dataset):
    """
    Base class for segmentation datasets with common functionality.
    """

    def __init__(
        self,
        image_processor: SegformerImageProcessor,
        mask_feature: str = "annotation",
        ignore_index: int = 255,
        binary: bool = True,
        transform: Optional[Callable] = None # Add transform argument
    ):
        """
        Initialize base segmentation dataset.

        Args:
            image_processor: Image processor for preprocessing
            mask_feature: Feature name for the mask/annotation
            ignore_index: Index to ignore in mask
            binary: Whether to convert masks to binary format
            transform: Optional augmentation transform to apply
        """
        self.image_processor = image_processor
        self.mask_feature = mask_feature
        self.ignore_index = ignore_index
        self.binary = binary
        self.transform = transform # Store transform

        # Set default values
        self.max_recursion_depth = 5
        self.class_imbalance_threshold = 99.0  # Percentage threshold for class imbalance

    def validate_sample(self, image: np.ndarray, mask: np.ndarray, idx: int) -> bool:
        """
        Validate a sample, raising appropriate exceptions for invalid samples.

        Args:
            image: Image array
            mask: Mask array
            idx: Sample index

        Returns:
            True if sample is valid, otherwise raises an exception

        Raises:
            ShapeMismatchError: If image and mask shapes don't match
            EmptyMaskError: If mask has only one unique value
            ClassImbalanceError: If mask has severe class imbalance
        """
        # Check if image and mask have compatible shapes
        if image.shape[:2] != mask.shape[:2]:
            raise ShapeMismatchError(
                image_shape=image.shape,
                mask_shape=mask.shape,
                message=f"Sample {idx}: Image shape {image.shape} doesn't match mask shape {mask.shape}"
            )

        # Check if mask has enough unique values
        unique_values = np.unique(mask)
        if len(unique_values) < 2:
            raise EmptyMaskError(
                unique_values=unique_values,
                sample_idx=idx
            )

        # Check for class imbalance
        total_pixels = mask.size
        for label in unique_values:
            label_pixels = np.sum(mask == label)
            percentage = (label_pixels / total_pixels) * 100

            # If a class covers more than threshold% of the mask, warn about class imbalance
            if percentage > self.class_imbalance_threshold:
                percentages = {value: (np.sum(mask == value) / total_pixels) * 100 for value in unique_values}
                logging.warning(f"Sample {idx} has class imbalance. Label {label} covers {percentage:.2f}% of the mask.")

        return True

    def preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Preprocess image for input to the model.

        Args:
            image: Input image

        Returns:
            Processed image
        """
        # Ensure image is in RGB format
        image_rgb = ensure_rgb(image)
        return image_rgb

    def preprocess_mask(
        self,
        mask: np.ndarray,
        target_shape: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Preprocess mask for input to the model.

        Args:
            mask: Input mask
            target_shape: Optional target shape for the mask

        Returns:
            Processed mask
        """
        # Convert to binary format if requested
        if self.binary:
            mask = (mask > 0).astype(np.int32)

        # Resize mask if target shape is provided and ensure correct shape/format
        # The ensure_mask_shape function handles potential multi-channel masks internally now.
        if target_shape is not None:
            mask = ensure_mask_shape(mask, target_shape, binary=self.binary)
        else:
            # Even if not resizing, ensure the mask is 2D and binary if needed
            mask = ensure_mask_shape(mask, mask.shape[:2], binary=self.binary)


        return mask

    def create_fallback_sample(self) -> Dict[str, torch.Tensor]:
        """
        Create a fallback sample for error recovery.

        Returns:
            Dictionary with fallback data
        """
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


class TCDDataset(BaseSegmentationDataset):
    """
    Improved dataset class for the TCD (Tree Crown Delineation) dataset.
    Handles loading from Hugging Face datasets, preprocessing, optional tiling,
    and augmentations based on the provided configuration.
    """
    def __init__(
        self,
        dataset: Union[HFDataset, DatasetDict], # Expecting a HF Dataset or DatasetDict split
        image_processor: SegformerImageProcessor,
        config: Config, # Add config object
        split: str = "train", # Specify train/validation/test
        target_size: Optional[Tuple[int, int]] = None # Add target_size for resizing
    ):
        """
        Initialize TCD Dataset.

        Args:
            dataset: Hugging Face Dataset or a specific split (e.g., dataset['train'])
            image_processor: Pre-initialized SegFormer image processor.
            config: Configuration object with dataset and augmentation parameters.
            split: Which split of the data this dataset represents ('train', 'validation', 'test').
        """
        # Determine mask feature name (handle potential variations)
        mask_feature = "annotation"
        if mask_feature not in dataset.features:
            # Try common alternatives or raise error
            potential_features = ["label", "mask", "segmentation_mask"]
            found = False
            for feat in potential_features:
                if feat in dataset.features:
                    mask_feature = feat
                    found = True
                    logger.warning(f"Using mask feature '{mask_feature}' instead of default 'annotation'.")
                    break
            if not found:
                raise DatasetError(f"Could not find mask feature in dataset features: {dataset.features}")

        super().__init__(
            image_processor=image_processor,
            mask_feature=mask_feature,
            ignore_index=config.get("ignore_index", 255),
            binary=True # Assuming TCD is binary
        )
        self.dataset = dataset
        self.config = config
        self.split = split
        self.num_labels = len(config["id2label"])
        self.target_size = target_size

        # --- Random Crop Configuration (Replaces Tiling) ---
        self.random_crop_size = None
        if self.split == 'train':
            crop_size_config = self.config.get("augmentation", {}).get("random_crop_size")
            if crop_size_config:
                if isinstance(crop_size_config, int):
                    self.random_crop_size = (crop_size_config, crop_size_config)
                elif isinstance(crop_size_config, (list, tuple)) and len(crop_size_config) == 2:
                    self.random_crop_size = tuple(crop_size_config)
                else:
                    logger.warning(f"Invalid random_crop_size format: {crop_size_config}. Disabling random crop.")
                if self.random_crop_size:
                    logger.info(f"Random crop enabled for training split with size: {self.random_crop_size}")
            else:
                logger.info("Random crop not configured for training split.")
        # --- End Random Crop Config ---

        # Create augmentation transform only if it's the training split
        self.transform = None
        if self.split == 'train' and self.config.get("augmentation", {}).get("apply", False):
            self.transform = create_augmentation_transform(config) # Uses updated function
            if self.transform:
                logger.info(f"Augmentation enabled for '{self.split}' split.")
            else:
                logger.info(f"Augmentation configured but no transforms created for '{self.split}' split.")
        else:
            logger.info(f"Augmentation disabled for '{self.split}' split.")

        # --- Removed Tiling Logic ---
        # self.train_tile_size = ...
        # self.train_tile_stride = ...
        # self.use_training_tiling = ...
        # self._tile_map = None
        self._num_samples = len(self.dataset) # Number of samples is just the dataset length now

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return self._num_samples

    # --- Removed _create_tile_map function ---


    @handle_dataset_error
    def __getitem__(self, idx):
        # --- Removed Tiling Index Logic ---
        # if self.use_training_tiling: ...
        orig_idx = idx # Index directly maps to dataset index

        # Load image (ensure PIL format for transforms)
        img_data = self.dataset[orig_idx]["image"]
        if isinstance(img_data, str):
            image = Image.open(img_data).convert("RGB")
        elif isinstance(img_data, np.ndarray):
            image = Image.fromarray(ensure_rgb(img_data)) # Ensure RGB if numpy
        elif isinstance(img_data, Image.Image):
            image = img_data.convert("RGB") # Ensure RGB if PIL
        else:
            raise TypeError(f"Unsupported image type: {type(img_data)}")

        # Load mask (ensure PIL format for transforms)
        mask_data = self.dataset[orig_idx][self.mask_feature]
        if isinstance(mask_data, str):
            mask = Image.open(mask_data) # Keep as is initially (e.g., 'L' mode)
        elif isinstance(mask_data, np.ndarray):
            # Ensure 2D before converting to PIL
            mask_np = ensure_mask_shape(mask_data, target_shape=None, binary=False) # Don't force binary yet
            mask = Image.fromarray(mask_np)
        elif isinstance(mask_data, Image.Image):
            mask = mask_data
        else:
            raise TypeError(f"Unsupported mask type: {type(mask_data)}")

        # --- Preprocessing and Augmentation ---
        try:
            # Ensure mask is single channel ('L' mode) before transforms
            if mask.mode != 'L':
                 mask = mask.convert('L')

            # 1. Apply Random Crop *before* other augmentations if configured for training
            if self.split == 'train' and self.random_crop_size:
                # Ensure image/mask are large enough for crop, resize if necessary
                img_w, img_h = image.size
                crop_h, crop_w = self.random_crop_size
                if img_h < crop_h or img_w < crop_w:
                    # Resize slightly larger than crop size to allow cropping
                    target_h = max(img_h, crop_h)
                    target_w = max(img_w, crop_w)
                    logger.debug(f"Image {orig_idx} size ({img_h}x{img_w}) < crop size {self.random_crop_size}. Resizing to ({target_h}x{target_w}).")
                    image = TF.resize(image, (target_h, target_w), interpolation=T.InterpolationMode.BILINEAR)
                    mask = TF.resize(mask, (target_h, target_w), interpolation=T.InterpolationMode.NEAREST)

                # Get parameters for random crop
                i, j, h, w = T.RandomCrop.get_params(image, output_size=self.random_crop_size)
                # Apply crop to both image and mask
                image = TF.crop(image, i, j, h, w)
                mask = TF.crop(mask, i, j, h, w)
                logger.debug(f"Applied RandomCrop {self.random_crop_size} to sample {idx}")

            # 2. Apply other augmentations (flips, rotation, color jitter, blur) if transform exists
            if self.transform:
                # CombinedTransform expects PIL images
                sample = self.transform({'image': image, 'mask': mask})
                image = sample['image']
                mask = sample['mask'] # Mask remains PIL Image

            # 3. Apply final resize if target_size is specified (for SETR)
            if self.target_size:
                image = TF.resize(image, self.target_size, interpolation=T.InterpolationMode.BILINEAR)
                mask = TF.resize(mask, self.target_size, interpolation=T.InterpolationMode.NEAREST)
                logger.debug(f"Resized sample {idx} to {self.target_size}")

            # 4. Apply image processor (handles normalization, tokenization)
            # The processor expects PIL Images or numpy arrays.
            # Convert mask back to numpy *before* processor
            mask_np = np.array(mask)
            # Ensure mask is binary after all transforms (as expected by TCD)
            mask_np = (mask_np > 0).astype(np.uint8) # Convert to 0/1

            # --- Final Preprocessing with Image Processor ---
            # Use the stored image_processor. `do_resize` should be False.
            encoding = self.image_processor(image, mask_np, return_tensors="pt")

            # Remove batch dimension added by processor
            pixel_values = encoding['pixel_values'].squeeze(0)
            labels = encoding['labels'].squeeze(0).long() # Ensure labels are LongTensor

            # Validate final shapes after all processing
            if pixel_values.ndim != 3:
                raise ShapeMismatchError(f"Invalid pixel_values dimensions: {pixel_values.ndim}, expected 3 (C, H, W)")
            if labels.ndim != 2:
                raise ShapeMismatchError(f"Invalid labels dimensions: {labels.ndim}, expected 2 (H, W)")

            # Check if processed image size matches label size
            img_h, img_w = pixel_values.shape[-2:]
            lbl_h, lbl_w = labels.shape[-2:]
            if img_h != lbl_h or img_w != lbl_w:
                 # This should NOT happen if do_resize=False and cropping is handled correctly
                 logger.error(f"CRITICAL: Post-processor shape mismatch! Image ({img_h}x{img_w}), Label ({lbl_h}x{lbl_w}) for index {idx}. Check processor/augmentation logic.")
                 raise ShapeMismatchError(f"Post-processor shape mismatch: Image ({img_h}x{img_w}), Label ({lbl_h}x{lbl_w})")

            return {
                'pixel_values': pixel_values,
                'labels': labels
            }

        except (InvalidSampleError, EmptyMaskError, ShapeMismatchError) as e:
            # Log specific error and return fallback
            logger.error(f"Data loading error for index {idx} (orig: {orig_idx}): {e}")
            # Fallback might need adjustment if crop size is large
            # For now, keep the small fallback
            return self.create_fallback_sample()
        except Exception as e:
            # Catch unexpected errors
            logger.exception(f"Unexpected error loading data for index {idx} (orig: {orig_idx}): {e}")
            return self.create_fallback_sample()


def load_and_shuffle_dataset(
    dataset_name: str,
    seed: Optional[int] = None,
    cache_dir: Optional[str] = None
) -> DatasetDict:
    """
    Load and shuffle a dataset once to ensure consistent ordering across the pipeline.

    Args:
        dataset_name: Name of the dataset on HuggingFace Hub
        seed: Random seed for reproducible shuffling
        cache_dir: Optional cache directory for datasets

    Returns:
        Shuffled dataset dictionary
    """
    # Set seed for deterministic operations
    random_seed = 42 if seed is None else seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Load dataset from HuggingFace Hub
    try:
        dataset_dict = load_dataset(dataset_name, cache_dir=cache_dir)
        logger.info(f"Successfully loaded dataset '{dataset_name}'")
    except Exception as e:
        raise DatasetError(f"Failed to load dataset '{dataset_name}': {str(e)}")

    # Apply seed-based shuffling to ensure consistent dataset ordering
    if "train" in dataset_dict:
        # Create a deterministically shuffled index array based on the seed
        indices = np.arange(len(dataset_dict["train"]))
        rng = np.random.RandomState(random_seed)
        rng.shuffle(indices)

        # Shuffle the train dataset using the shuffled indices
        dataset_dict["train"] = dataset_dict["train"].select(indices.tolist())
        logger.info(f"Shuffled training dataset using seed {random_seed}")

    # Log the number of samples in each split
    for split in ["train", "validation", "test"]:
        if split in dataset_dict:
            logger.info(f"Loaded {len(dataset_dict[split])} samples in '{split}' split.")
        else:
            logger.info(f"No '{split}' split found in dataset.")

    return dataset_dict


def load_evaluation_dataset(
    dataset_name: str,
    validation_split: float = 0.1,
    seed: int = 42,
    cache_dir: Optional[str] = None
) -> HFDataset:
    """
    Load only the evaluation split of a dataset for memory efficiency.
    If validation split doesn't exist, creates one from training data.

    Args:
        dataset_name: Name of the dataset on HuggingFace Hub
        validation_split: Fraction of training data to use for validation if needed
        seed: Random seed for reproducible dataset splitting
        cache_dir: Optional cache directory

    Returns:
        Evaluation dataset

    Raises:
        DatasetError: If dataset cannot be loaded
    """
    try:
        # First try to load only the validation split
        eval_dataset = load_dataset(dataset_name, split="validation", cache_dir=cache_dir)
        logger.info(f"Loaded existing validation split from {dataset_name}")
        return eval_dataset
    except Exception as e:
        logger.info(f"No separate validation split found in {dataset_name}. Creating from training data.")
        try:
            # Load training data and create a validation split
            train_dataset = load_dataset(dataset_name, split="train", cache_dir=cache_dir)

            # Set seed for reproducibility
            random_seed = seed
            rng = np.random.RandomState(random_seed)

            # Create indices for splitting
            total_size = len(train_dataset)
            val_size = int(total_size * validation_split)
            indices = np.arange(total_size)
            rng.shuffle(indices)

            # Select validation indices
            val_indices = indices[:val_size]
            eval_dataset = train_dataset.select(val_indices.tolist())

            logger.info(f"Created validation split with {len(eval_dataset)} samples from training data")
            return eval_dataset
        except Exception as split_error:
            raise DatasetError(f"Failed to create validation split: {str(split_error)}")


def get_dataset_info(
    dataset_dict: DatasetDict,
    mask_feature: str = "annotation"
) -> Dict[str, Any]:
    """
    Get information about the dataset.

    Args:
        dataset_dict: Dataset dictionary
        mask_feature: Feature name for the mask/annotation

    Returns:
        Dictionary with dataset information
    """
    # Check if train split exists
    if "train" not in dataset_dict:
        raise DatasetError("Dataset does not have a 'train' split")

    # Get dataset features
    features = dataset_dict["train"].features

    # Check if annotation feature exists
    if mask_feature in features:
        # For TCD dataset, annotation is binary (0: background, 1: tree_crown)
        id2label = {0: "background", 1: "tree_crown"}
        label2id = {"background": 0, "tree_crown": 1}
        num_labels = 2
    else:
        # If annotation feature doesn't exist, use default values
        # For binary segmentation (background and tree crown)
        id2label = {0: "background", 1: "tree_crown"}
        label2id = {"background": 0, "tree_crown": 1}
        num_labels = 2
        logger.warning(f"Warning: '{mask_feature}' feature not found in dataset. Using default binary segmentation labels.")

    # Get dataset sizes
    train_size = len(dataset_dict["train"])

    # Check if validation split exists
    has_validation = "validation" in dataset_dict
    val_size = len(dataset_dict["validation"]) if has_validation else 0

    # Check if test split exists
    has_test = "test" in dataset_dict
    test_size = len(dataset_dict["test"]) if has_test else 0

    # Print warnings for missing splits
    if not has_validation:
        logger.warning("Dataset does not have a 'validation' split")

    if not has_test:
        logger.warning("Dataset does not have a 'test' split")

    return {
        "id2label": id2label,
        "label2id": label2id,
        "num_labels": num_labels,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "has_validation": has_validation,
        "has_test": has_test
    }


def create_eval_dataloader(
    dataset_dict_or_name: Union[DatasetDict, str],
    image_processor: Optional[SegformerImageProcessor] = None,
    config: Optional[Config] = None,
    eval_batch_size: int = 16,
    num_workers: int = 4,
    validation_split: float = 0.1,
    seed: int = 42,
    mask_feature: str = "annotation"
) -> Tuple[DataLoader, Dict[int, str], Dict[str, int]]:
    """
    Create evaluation dataloader optimized for memory usage.

    Args:
        dataset_dict_or_name: Dataset dictionary or name of dataset on HuggingFace Hub
        image_processor: Image processor for preprocessing (created if None)
        config: Configuration object with dataloader settings (optional)
        eval_batch_size: Batch size for evaluation
        num_workers: Number of workers for data loading
        validation_split: Fraction of training data to use for validation if needed
        seed: Random seed for reproducible dataset splitting
        mask_feature: Feature name for the mask/annotation

    Returns:
        Tuple of (eval_dataloader, id2label, label2id)
    """
    # Handle case where dataset_dict_or_name is a string (dataset name)
    if isinstance(dataset_dict_or_name, str):
        # Load only evaluation data for efficiency
        eval_dataset_hf = load_evaluation_dataset(
            dataset_name=dataset_dict_or_name,
            validation_split=validation_split,
            seed=seed
        )
        # Wrap in a temporary DatasetDict to match existing code flow
        dataset_dict = DatasetDict({"validation": eval_dataset_hf})
    else:
        # Use provided dataset dictionary
        dataset_dict = dataset_dict_or_name

    # Create image processor if not provided (using default settings)
    if image_processor is None:
        image_processor = SegformerImageProcessor()
        logger.info("Created default SegformerImageProcessor.")

    # For binary segmentation (TCD dataset)
    id2label = {0: "background", 1: "tree_crown"}
    label2id = {"background": 0, "tree_crown": 1}

    # Create evaluation dataset
    eval_dataset = TCDDataset(
        dataset_dict=dataset_dict,
        image_processor=image_processor,
        split="validation",
        mask_feature=mask_feature
        # No augmentation for evaluation (handled internally by split)
    )

    # Determine dataloader settings
    pin_memory = True
    persistent_workers = True
    prefetch_factor = 2

    if config:
        pin_memory = config.get("dataloader_pin_memory", True)
        persistent_workers = config.get("dataloader_persistent_workers", True)
        prefetch_factor = config.get("dataloader_prefetch_factor", 2)

    # Create dataloader
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=False
    )

    return eval_dataloader, id2label, label2id


def create_dataloaders(
        dataset_dict: DatasetDict,
        image_processor: Optional[SegformerImageProcessor], # Make optional
        config: Config, # Pass config object for augmentation and tiling settings
        train_batch_size: int = 8,
        eval_batch_size: int = 16,
    num_workers: int = 4,
    validation_split: float = 0.1,
    seed: Optional[int] = None,
        mask_feature: str = "annotation",
        target_size: Optional[Tuple[int, int]] = None # Add target_size for resizing
    ) -> Tuple[DataLoader, DataLoader, Dict[int, str], Dict[str, int]]:
    """
    Create train and evaluation dataloaders for the dataset, applying augmentations and tiling to train set.

    Args:
        dataset_dict: Dataset dictionary
        image_processor: Image processor for preprocessing (optional, will be created if None)
        config: Configuration object containing augmentation, tiling, and other settings.
        train_batch_size: Batch size for training
        eval_batch_size: Batch size for evaluation
        num_workers: Number of workers for data loading
        validation_split: Fraction of training data to use for validation if no validation split exists
        seed: Random seed for reproducible dataset splitting
        mask_feature: Feature name for the mask/annotation

    Returns:
        Tuple of (train_dataloader, eval_dataloader, id2label, label2id)
    """
    # Get dataset info
    dataset_info = get_dataset_info(dataset_dict, mask_feature)
    id2label = dataset_info["id2label"]
    label2id = dataset_info["label2id"]
    has_validation = dataset_info["has_validation"]

    # Set seed for random operations (only used for random_split if needed)
    random_seed = 42 if seed is None else seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Create image processor if not provided (using settings consistent with prediction)
    if image_processor is None:
        # Ensure consistency with prediction pipeline: NO resizing during training either
        image_processor = SegformerImageProcessor(
            do_resize=False,  # Prevent automatic resizing
            do_rescale=True,  # Still normalize pixel values
            do_normalize=True  # Still perform ImageNet normalization
        )
        logger.info("Created SegformerImageProcessor with do_resize=False for training.")

    # Create augmentation transform for training set based on config
    train_transform = create_augmentation_transform(config)
    if train_transform:
        logger.info("Applying data augmentation to the training set.")
    else:
        logger.info("No data augmentation applied to the training set.")

    # Create datasets
    train_dataset = TCDDataset(
        dataset=dataset_dict['train'], # Pass specific train split
        image_processor=image_processor,
        config=config, # Pass config
        split="train",
        target_size=target_size # Pass target_size
        # mask_feature=mask_feature, # Removed: TCDDataset determines this internally
        # transform=train_transform # Removed: TCDDataset handles transform internally
    )

    # Create eval dataset (no transform, no tiling needed from config here)
    eval_split_name = "validation" if has_validation else "train" # Base split for eval
    eval_dataset_base = TCDDataset(
        dataset=dataset_dict[eval_split_name], # Pass specific eval split
        image_processor=image_processor,
        config=config, # Pass config (though tiling params won't be used for eval split)
        split=eval_split_name,
        target_size=target_size # Pass target_size
        # mask_feature=mask_feature, # Removed: TCDDataset determines this internally
        # No augmentation for eval (handled internally by split)
    )

    if not has_validation:
        # Create validation split from the base eval dataset (which uses 'train' split)
        logger.warning(
            f"Dataset does not have a 'validation' split. Creating one from the training data "
            f"with {validation_split:.1%} of samples for evaluation."
        )
        total_size = len(eval_dataset_base)
        val_size = int(total_size * validation_split)
        train_size = total_size - val_size # This train_size is just for the split logic

        # Create indices for splitting
        indices = list(range(total_size))
        generator = torch.Generator().manual_seed(random_seed)
        shuffled_indices = torch.randperm(total_size, generator=generator).tolist()

        # Split indices into train and validation
        # Note: These indices refer to the *original* training set order
        train_indices_for_split = shuffled_indices[val_size:]
        val_indices_for_split = shuffled_indices[:val_size]

        # Important: The actual train_dataset uses ALL original train indices and applies transform/tiling internally
        # We only need the val_indices_for_split to create the eval_dataset subset
        eval_dataset = torch.utils.data.Subset(eval_dataset_base, val_indices_for_split)

        logger.info(f"Created validation split: {len(train_dataset)} train (effective), {len(eval_dataset)} validation samples.")
    else:
        # Use the existing validation split directly
        eval_dataset = eval_dataset_base # Already loaded with split="validation"


    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config.get("dataloader_pin_memory", True) and torch.cuda.is_available(),
        persistent_workers=config.get("dataloader_persistent_workers", True) and num_workers > 0,
        prefetch_factor=config.get("dataloader_prefetch_factor", 2) if num_workers > 0 else None,
        drop_last=False
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.get("dataloader_pin_memory", True) and torch.cuda.is_available(),
        persistent_workers=config.get("dataloader_persistent_workers", True) and num_workers > 0,
        prefetch_factor=config.get("dataloader_prefetch_factor", 2) if num_workers > 0 else None,
        drop_last=False
    )
    # Log DataLoader settings
    log_msg = f"DataLoader settings: pin_memory={train_dataloader.pin_memory}, persistent_workers={train_dataloader.persistent_workers}, prefetch_factor={train_dataloader.prefetch_factor}"
    # Use the logger defined at the start of the function
    logger.info(log_msg)
    # Removed the redundant print statement as logging should handle output

    return train_dataloader, eval_dataloader, id2label, label2id

def create_dataset_from_masks(
    images: List[Union[str, np.ndarray, Image.Image]],
    masks: List[Union[str, np.ndarray]],
    image_processor: SegformerImageProcessor,
    transform: Optional[Callable] = None
) -> Dataset:
    """
    Create a custom dataset from a list of images and masks.

    Args:
        images: List of image paths or arrays
        masks: List of mask paths or arrays
        image_processor: Image processor for preprocessing
        transform: Optional transform to apply to the samples

    Returns:
        PyTorch Dataset
    """
    class CustomSegmentationDataset(BaseSegmentationDataset):
        def __init__(
            self,
            images,
            masks,
            image_processor,
            transform=None
        ):
            super().__init__(image_processor)
            self.images = images
            self.masks = masks
            self.transform = transform

            # Validate inputs
            if len(images) != len(masks):
                raise DatasetError(
                    f"Number of images ({len(images)}) doesn't match number of masks ({len(masks)})"
                )

        def __len__(self):
            return len(self.images)

        @handle_dataset_error
        def __getitem__(self, idx):
            # Load image
            if isinstance(self.images[idx], str):
                image = Image.open(self.images[idx]).convert("RGB")
            elif isinstance(self.images[idx], np.ndarray):
                image = self.images[idx]
            elif isinstance(self.images[idx], Image.Image):
                image = self.images[idx]
            else:
                raise TypeError(f"Unsupported image type: {type(self.images[idx])}")

            # Load mask
            if isinstance(self.masks[idx], str):
                mask = np.array(Image.open(self.masks[idx]))
            elif isinstance(self.masks[idx], np.ndarray):
                mask = self.masks[idx]
            else:
                raise TypeError(f"Unsupported mask type: {type(self.masks[idx])}")

            # Preprocess image and mask
            image = self.preprocess_image(image)
            mask = self.preprocess_mask(mask)

            # Apply transform if provided
            if self.transform:
                sample = self.transform({"image": image, "mask": mask})
                image, mask = sample["image"], sample["mask"]

            # Apply image processor
            encoding = self.image_processor(image, mask, return_tensors="pt")

            # Remove batch dimension
            for k, v in encoding.items():
                encoding[k] = v.squeeze()

            return encoding

    return CustomSegmentationDataset(images, masks, image_processor, transform)
