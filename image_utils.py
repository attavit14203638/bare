#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image processing utilities for TCD-SegFormer model.

This module consolidates all image processing functionality to avoid
code duplication and ensure consistent handling of images and masks.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict, Tuple, Optional, Union, List, Any, Callable, Generator
import logging
from utils import LOGGER_NAME, get_logger

# Setup module logger
logger = get_logger()

def ensure_rgb(image: Union[np.ndarray, Image.Image]) -> Union[np.ndarray, Image.Image]:
    """
    Ensure image is in RGB format.
    
    Args:
        image: Input image (PIL Image or numpy array)
        
    Returns:
        RGB image (same type as input)
    """
    if isinstance(image, Image.Image):
        if image.mode != "RGB":
            return image.convert("RGB")
        return image
    elif isinstance(image, np.ndarray):
        # If image is grayscale (2D)
        if len(image.shape) == 2:
            # Convert to 3D by duplicating channels
            return np.stack([image] * 3, axis=2)
        # If image is already 3D with 3 channels, assume it's RGB
        elif len(image.shape) == 3 and image.shape[2] == 3:
            return image
        # If image is 3D with 1 channel, duplicate to 3 channels
        elif len(image.shape) == 3 and image.shape[2] == 1:
            return np.concatenate([image] * 3, axis=2)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

def normalize_image(
    image: np.ndarray, 
    mean: Optional[Union[float, List[float]]] = None, 
    std: Optional[Union[float, List[float]]] = None
) -> np.ndarray:
    """
    Normalize image by subtracting mean and dividing by standard deviation.
    
    Args:
        image: Input image (H, W, C)
        mean: Mean value(s) for normalization (per channel)
        std: Standard deviation value(s) for normalization (per channel)
        
    Returns:
        Normalized image
    """
    # If image is uint8, convert to float [0, 1]
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    # If mean and std are not provided, use ImageNet values
    if mean is None:
        mean = [0.485, 0.456, 0.406]  # ImageNet mean
    if std is None:
        std = [0.229, 0.224, 0.225]  # ImageNet std
    
    # Ensure mean and std are lists
    if isinstance(mean, (int, float)):
        mean = [mean] * 3
    if isinstance(std, (int, float)):
        std = [std] * 3
    
    # Normalize image
    for i in range(3):
        image[..., i] = (image[..., i] - mean[i]) / std[i]
    
    return image


# --- Tiling Utilities ---

def get_tiles(
    image: np.ndarray, 
    tile_size: int, 
    overlap: int
) -> Generator[Tuple[int, int, np.ndarray], None, None]:
    """
    Generates overlapping tiles from an image.

    Args:
        image: Input image (H, W, C) or (H, W).
        tile_size: The size of the square tiles.
        overlap: The overlap between adjacent tiles.

    Yields:
        Tuple of (y_coord, x_coord, tile_array).
    """
    if image.ndim not in [2, 3]:
        raise ValueError(f"Unsupported image dimensions: {image.ndim}. Expected 2 or 3.")
        
    h, w = image.shape[:2]
    stride = tile_size - overlap

    if stride <= 0:
        raise ValueError("Overlap must be less than tile_size.")

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Calculate tile boundaries
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            
            # Adjust start coordinates if tile goes out of bounds
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)

            # Extract tile
            if image.ndim == 3:
                tile = image[y_start:y_end, x_start:x_end, :]
            else: # ndim == 2
                tile = image[y_start:y_end, x_start:x_end]

            # Pad tile if it's smaller than tile_size (edge tiles)
            pad_h = tile_size - tile.shape[0]
            pad_w = tile_size - tile.shape[1]
            
            if pad_h > 0 or pad_w > 0:
                if image.ndim == 3:
                    padding = [(0, pad_h), (0, pad_w), (0, 0)]
                else:
                    padding = [(0, pad_h), (0, pad_w)]
                tile = np.pad(tile, padding, mode='constant', constant_values=0)

            yield y_start, x_start, tile


def stitch_tiles(
    tiles: Dict[Tuple[int, int], np.ndarray],
    target_shape: Tuple[int, int],
    tile_size: int,
    overlap: int
) -> np.ndarray:
    """
    Stitches overlapping predicted tiles back into a full mask or image.
    Uses averaging in overlapping regions and ensures exact matching to target dimensions.

    Args:
        tiles: Dictionary mapping (y_coord, x_coord) to predicted tile arrays.
               Assumes tiles are (tile_size, tile_size, num_classes) for logits/probs
               or (tile_size, tile_size) for class indices.
        target_shape: The desired shape (H, W) of the final stitched output.
        tile_size: The size of the square tiles used.
        overlap: The overlap between adjacent tiles used.

    Returns:
        Stitched array of shape (target_shape[0], target_shape[1], num_classes) or (target_shape[0], target_shape[1]).
    """
    h, w = target_shape
    stride = tile_size - overlap
    
    # Log target shape
    from utils import log_or_print
    import logging
    
    log_or_print(
        f"Stitching tiles to target shape: {target_shape}, tile_size: {tile_size}, overlap: {overlap}",
        logger if 'logger' in locals() else None,
        logging.INFO,
        False
    )
    
    # Determine output shape based on tile shape
    first_tile_coords = next(iter(tiles))
    first_tile = tiles[first_tile_coords]
    if first_tile.ndim == 3: # Logits/probabilities
        num_classes = first_tile.shape[2]
        stitched_output = np.zeros((h, w, num_classes), dtype=np.float32)
        counts = np.zeros((h, w, num_classes), dtype=np.int16)
        log_or_print(
            f"Creating multi-class output of shape ({h}, {w}, {num_classes})",
            logger if 'logger' in locals() else None,
            logging.DEBUG,
            False
        )
    elif first_tile.ndim == 2: # Class indices
        # For class indices, we cannot simply average.
        # A common approach is to take the prediction from the center of the tile,
        # or use a more complex voting/majority scheme.
        # For simplicity, we'll implement an overwrite strategy where later tiles
        # overwrite earlier ones in overlap regions. This is less ideal but simpler.
        # A better approach for masks would be to stitch logits/probabilities first, then argmax.
        logger.warning("Stitching class index tiles directly using overwrite strategy. Stitching probabilities/logits before argmax is recommended for better results.")
        stitched_output = np.zeros((h, w), dtype=first_tile.dtype)
        # No counts needed for overwrite strategy
    else:
        raise ValueError(f"Unsupported tile dimension: {first_tile.ndim}. Expected 2 or 3.")

    # Stitch tiles
    for (y_start, x_start), tile in tiles.items():
        # Calculate the region in the full output array
        y_end = min(y_start + tile_size, h)
        x_end = min(x_start + tile_size, w)
        
        # Calculate the corresponding region in the tile (accounting for padding removal)
        tile_y_end = tile_size - ((y_start + tile_size) - y_end)
        tile_x_end = tile_size - ((x_start + tile_size) - x_end)
        
        # Ensure we don't have negative indices
        tile_y_end = max(0, tile_y_end)
        tile_x_end = max(0, tile_x_end)
        
        # Crop tile to fit the region
        try:
            tile_crop = tile[:tile_y_end, :tile_x_end]
            
            # Get shapes for comparison
            region_shape = stitched_output[y_start:y_end, x_start:x_end].shape
            tile_crop_shape = tile_crop.shape
            
            log_or_print(
                f"Stitching tile at (y={y_start}, x={x_start}): region={region_shape}, tile={tile_crop_shape}",
                logger if 'logger' in locals() else None,
                logging.DEBUG,
                False
            )
            
            # Check if shapes match
            if region_shape[:2] != tile_crop_shape[:2]:
                log_or_print(
                    f"Shape mismatch at (y={y_start}, x={x_start}): region={region_shape}, tile={tile_crop_shape}. Resizing tile...",
                    logger if 'logger' in locals() else None,
                    logging.WARNING,
                    False
                )
                
                # Handle different tile dimensions explicitly
                if len(region_shape) == 3 and len(tile_crop_shape) == 3:
                    # For 3D arrays (with channels), resize each channel separately
                    resized_tile = np.zeros((region_shape[0], region_shape[1], tile_crop_shape[2]), dtype=tile_crop.dtype)
                    for c in range(tile_crop_shape[2]):
                        import cv2
                        resized_tile[:, :, c] = cv2.resize(
                            tile_crop[:, :, c], 
                            (region_shape[1], region_shape[0]), 
                            interpolation=cv2.INTER_LINEAR if tile_crop.dtype == np.float32 else cv2.INTER_NEAREST
                        )
                    tile_crop = resized_tile
                elif len(region_shape) == 2 and len(tile_crop_shape) == 2:
                    # For 2D arrays (masks), use nearest neighbor
                    import cv2
                    tile_crop = cv2.resize(
                        tile_crop, 
                        (region_shape[1], region_shape[0]), 
                        interpolation=cv2.INTER_NEAREST
                    )
                else:
                    # If dimensions don't match, raise an error
                    raise ValueError(
                        f"Dimension mismatch: region shape {region_shape}, tile shape {tile_crop_shape}"
                    )
            
            # Add tile to the stitched output
            if stitched_output.ndim == 3: # Logits/probabilities
                stitched_output[y_start:y_end, x_start:x_end] += tile_crop
                counts[y_start:y_end, x_start:x_end] += 1
            else: # Class indices (overwrite)
                stitched_output[y_start:y_end, x_start:x_end] = tile_crop
                
        except Exception as e:
            log_or_print(
                f"Error stitching tile at (y={y_start}, x={x_start}): {e}",
                logger if 'logger' in locals() else None,
                logging.ERROR,
                False
            )
            # Continue with other tiles instead of failing

    # Average overlapping regions for logits/probabilities
    if stitched_output.ndim == 3:
        # Avoid division by zero where counts are zero
        counts[counts == 0] = 1
        stitched_output /= counts

    # Ensure output matches target_shape exactly
    if stitched_output.shape[:2] != target_shape:
        log_or_print(
            f"Stitched output shape {stitched_output.shape[:2]} doesn't match target shape {target_shape}. Applying final resize.",
            logger if 'logger' in locals() else None,
            logging.WARNING,
            False
        )
        
        # Final resize to ensure correct dimensions
        if stitched_output.ndim == 3:
            # For 3D arrays (with channels)
            import cv2
            final_output = np.zeros((target_shape[0], target_shape[1], stitched_output.shape[2]), dtype=stitched_output.dtype)
            for c in range(stitched_output.shape[2]):
                final_output[:, :, c] = cv2.resize(
                    stitched_output[:, :, c], 
                    (target_shape[1], target_shape[0]), 
                    interpolation=cv2.INTER_LINEAR if stitched_output.dtype == np.float32 else cv2.INTER_NEAREST
                )
            stitched_output = final_output
        else:
            # For 2D arrays (masks)
            import cv2
            stitched_output = cv2.resize(
                stitched_output, 
                (target_shape[1], target_shape[0]), 
                interpolation=cv2.INTER_NEAREST
            )
    
    # Final consistency check
    assert stitched_output.shape[:2] == target_shape, \
        f"Final stitched output shape {stitched_output.shape[:2]} still doesn't match target shape {target_shape}"
    
    return stitched_output

def resize_image(
    image: Union[np.ndarray, torch.Tensor], 
    size: Tuple[int, int],
    mode: str = 'bilinear',
    align_corners: bool = False
) -> Union[np.ndarray, torch.Tensor]:
    """
    Resize image to target size.
    
    Args:
        image: Input image (numpy array or torch tensor)
        size: Target size (height, width)
        mode: Interpolation mode ('nearest', 'bilinear', 'bicubic')
        align_corners: Whether to align corners (only for 'bilinear' and 'bicubic')
        
    Returns:
        Resized image (same type as input)
    """
    is_numpy = isinstance(image, np.ndarray)
    
    # Convert numpy array to torch tensor if needed
    if is_numpy:
        # Handle channel dimension
        if len(image.shape) == 3 and image.shape[2] == 3:  # HWC format
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()  # Convert to CHW
        elif len(image.shape) == 2:  # Single channel
            image_tensor = torch.from_numpy(image).unsqueeze(0).float()  # Add channel dim
        else:
            image_tensor = torch.from_numpy(image).float()
    else:
        image_tensor = image
    
    # Ensure batch dimension
    if len(image_tensor.shape) == 3:  # CHW format
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dim -> BCHW
    
    # Resize image
    resized_tensor = F.interpolate(
        image_tensor,
        size=size,
        mode=mode,
        align_corners=align_corners if mode in ['bilinear', 'bicubic'] else None
    )
    
    # Remove batch dimension
    resized_tensor = resized_tensor.squeeze(0)
    
    # Convert back to numpy if input was numpy
    if is_numpy:
        if len(resized_tensor.shape) == 3:  # CHW format
            resized_array = resized_tensor.permute(1, 2, 0).numpy()  # Convert to HWC
        else:
            resized_array = resized_tensor.numpy()
        return resized_array
    
    return resized_tensor

def resize_mask(
    mask: Union[np.ndarray, torch.Tensor], 
    size: Tuple[int, int],
    ignore_index: int = 255
) -> Union[np.ndarray, torch.Tensor]:
    """
    Resize segmentation mask to target size.
    
    Args:
        mask: Input mask (numpy array or torch tensor)
        size: Target size (height, width)
        ignore_index: Index to ignore when resizing
        
    Returns:
        Resized mask (same type as input)
    """
    is_numpy = isinstance(mask, np.ndarray)
    
    # Convert numpy array to torch tensor if needed
    if is_numpy:
        mask_tensor = torch.from_numpy(mask).long()
    else:
        mask_tensor = mask.long()
    
    # Ensure batch dimension
    if len(mask_tensor.shape) == 2:  # HW format
        mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dim -> BHW
    
    # Use nearest neighbor interpolation to preserve class indices
    resized_tensor = F.interpolate(
        mask_tensor.float().unsqueeze(1),  # Add channel dim -> BCHW
        size=size,
        mode='nearest'
    ).squeeze(1).long()  # Remove channel dim -> BHW
    
    # Remove batch dimension
    resized_tensor = resized_tensor.squeeze(0)  # -> HW
    
    # Convert back to numpy if input was numpy
    if is_numpy:
        resized_array = resized_tensor.numpy()
        return resized_array
    
    return resized_tensor

def ensure_mask_shape(
    mask: np.ndarray,
    expected_shape: Tuple[int, ...],
    binary: bool = True
) -> np.ndarray:
    """
    Ensure mask has expected shape and format.
    
    Args:
        mask: Input mask
        expected_shape: Expected mask shape
        binary: Whether to convert to binary format (0/1)
        
    Returns:
        Processed mask with expected shape
    """
    # Make a copy to avoid modifying the original mask
    mask = mask.copy()
    
    # Handle unexpected mask shape
    if len(mask.shape) > 2:
        # If mask has shape (H, W, C) (RGB image), convert to grayscale
        if len(mask.shape) == 3 and mask.shape[2] == 3:
            # Convert RGB to grayscale by taking the mean of the channels
            mask = np.mean(mask, axis=2).astype(np.uint8)
        # If mask has shape (C, H, W), take the first channel
        elif len(mask.shape) == 3:
            mask = mask[0]
        # If mask has more dimensions, try to get a 2D slice
        else:
            mask = mask[0, 0]
    
    # Convert to binary format (0: background, 1: foreground) if requested
    if binary:
        mask = (mask > 0).astype(np.int32)
    
    # Ensure mask shape matches expected shape
    if mask.shape != expected_shape[:2]:
        mask = resize_mask(mask, expected_shape[:2])
    
    return mask

def get_device_for_tensor(tensor: torch.Tensor) -> torch.device:
    """
    Get the device of a tensor safely.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Tensor device
    """
    if hasattr(tensor, 'device'):
        return tensor.device
    else:
        return torch.device('cpu')
