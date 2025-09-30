#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Checkpoint management utilities for TCD-SegFormer model.

This module provides functions for saving, loading, and managing model checkpoints
to ensure consistent handling across the codebase.
"""

import os
import json
import torch
import logging
import shutil
from typing import Dict, Any, Optional, Union, Tuple, List
from datetime import datetime
from types import SimpleNamespace

from exceptions import FileError, FileNotFoundError


def save_checkpoint(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    global_step: int,
    epoch: int,
    output_dir: str,
    max_to_keep: int = 3,
    metrics: Optional[Dict[str, float]] = None,
    metric_to_monitor: Optional[str] = None,
    monitor_mode: str = 'max',
    is_final: bool = False,
    config: Optional[Any] = None  # Add config parameter
):
    """
    Saves a checkpoint of the model, optimizer, and scheduler state.

    Args:
        model: The model to save.
        optimizer: The optimizer state to save.
        scheduler: The scheduler state to save.
        global_step: The current global training step.
        epoch: The current epoch number.
        output_dir: Directory where checkpoints will be saved.
        max_to_keep: Maximum number of recent checkpoints to keep.
        metrics: Optional dictionary of metrics from evaluation.
        metric_to_monitor: Optional metric key from 'metrics' to track for best checkpoint.
        monitor_mode: 'max' or 'min'. How to compare the monitored metric.
        is_final: If True, marks this as the final checkpoint, potentially skipping cleanup.
    """
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    try:
        # Save model state
        model_to_save = model.module if hasattr(model, 'module') else model
        model_path = os.path.join(checkpoint_dir, 'pytorch_model.bin')
        torch.save(model_to_save.state_dict(), model_path)
        logging.info(f"Model state saved to {model_path}")

        # Save optimizer state
        optimizer_path = os.path.join(checkpoint_dir, 'optimizer.pt')
        torch.save(optimizer.state_dict(), optimizer_path)
        logging.info(f"Optimizer state saved to {optimizer_path}")

        # Save scheduler state
        scheduler_path = os.path.join(checkpoint_dir, 'scheduler.pt')
        torch.save(scheduler.state_dict(), scheduler_path)
        logging.info(f"Scheduler state saved to {scheduler_path}")

        # Save the full configuration
        if config:
            config_path = os.path.join(checkpoint_dir, 'config.json')
            try:
                # If it's a class with to_dict, use it, otherwise convert directly
                config_dict = config.to_dict() if hasattr(config, 'to_dict') else config
                with open(config_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
                logging.info(f"Configuration saved to {config_path}")
            except Exception as e:
                logging.warning(f"Could not save configuration to {config_path}: {e}")

        # Save training state (step, epoch, etc.)
        state_path = os.path.join(checkpoint_dir, 'trainer_state.json')
        state = {'global_step': global_step, 'epoch': epoch}
        if metrics:
            state['metrics'] = metrics
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        logging.info(f"Training state saved to {state_path}")

        # --- Start: Add logic for saving best checkpoint ---
        if metric_to_monitor and metrics and metric_to_monitor in metrics:
            current_metric_value = metrics[metric_to_monitor]
            best_checkpoint_dir = os.path.join(output_dir, "best_checkpoint")
            best_metric_file = os.path.join(best_checkpoint_dir, "best_metric.json")

            previous_best_metric = -float('inf') if monitor_mode == 'max' else float('inf')

            # Check if a previous best checkpoint exists and read its metric
            if os.path.exists(best_metric_file):
                try:
                    with open(best_metric_file, 'r') as f:
                        best_metric_data = json.load(f)
                        # Ensure the metric name matches what we are currently monitoring
                        if best_metric_data.get('metric_name') == metric_to_monitor:
                            previous_best_metric = best_metric_data.get('value', previous_best_metric)
                except Exception as e:
                    logging.warning(f"Could not read previous best metric from {best_metric_file}: {e}")

            # Compare current metric with previous best
            is_better = False
            if monitor_mode == 'max' and current_metric_value > previous_best_metric:
                is_better = True
            elif monitor_mode == 'min' and current_metric_value < previous_best_metric:
                is_better = True

            if is_better:
                logging.info(f"New best metric '{metric_to_monitor}': {current_metric_value:.4f} (previous: {previous_best_metric:.4f}). Saving to {best_checkpoint_dir}")

                # Remove old best checkpoint directory if it exists
                if os.path.exists(best_checkpoint_dir):
                    try:
                        shutil.rmtree(best_checkpoint_dir)
                    except Exception as e:
                        logging.error(f"Failed to remove previous best checkpoint {best_checkpoint_dir}: {e}")

                # Copy the current checkpoint to the best checkpoint directory
                try:
                    # Use dirs_exist_ok=True for Python 3.8+
                    shutil.copytree(checkpoint_dir, best_checkpoint_dir, dirs_exist_ok=True)

                    # Save the new best metric info
                    best_metric_data = {
                        'metric_name': metric_to_monitor,
                        'value': current_metric_value,
                        'global_step': global_step,
                        'epoch': epoch,
                        'timestamp': datetime.now().isoformat()
                    }
                    with open(best_metric_file, 'w') as f:
                        json.dump(best_metric_data, f, indent=2)

                except Exception as e:
                    logging.error(f"Failed to copy or save best checkpoint to {best_checkpoint_dir}: {e}")
        # --- End: Add logic for saving best checkpoint ---

        # Clean up old checkpoints
        cleanup_old_checkpoints(output_dir, max_to_keep)

    except Exception as e:
        logging.error(f"Failed to save checkpoint at step {global_step}: {e}")
        # Clean up partially created checkpoint directory if saving failed
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
            logging.warning(f"Removed partially created checkpoint directory: {checkpoint_dir}")


def cleanup_old_checkpoints(output_dir: str, max_to_keep: int):
    """
    Removes older checkpoints, keeping only the `max_to_keep` most recent ones.
    The 'best_checkpoint' directory, if exists, is always preserved.

    Args:
        output_dir: Directory containing the checkpoints.
        max_to_keep: Maximum number of recent checkpoints to retain.
    """
    logging.info("Cleaning up older checkpoints...")
    # Get all checkpoint-{step} directories
    checkpoints = []
    best_checkpoint_name = "best_checkpoint" # Define the name
    try:
        for d in os.listdir(output_dir):
            dir_path = os.path.join(output_dir, d)
            # Exclude the best checkpoint directory from cleanup consideration
            if d == best_checkpoint_name or not os.path.isdir(dir_path):
                continue

            if d.startswith("checkpoint-"):
                try:
                    step = int(d.split("-")[1])
                    checkpoints.append((step, dir_path))
                except (IndexError, ValueError):
                    logging.warning(f"Could not parse step from checkpoint directory name: {d}")
                    continue
    except FileNotFoundError:
        logging.warning(f"Checkpoint directory {output_dir} not found during cleanup.")
        return

    if len(checkpoints) <= max_to_keep:
        logging.info(f"No old checkpoints to remove. Found {len(checkpoints)}, max_to_keep={max_to_keep}")
        return

    # Sort checkpoints by step number (ascending)
    checkpoints.sort(key=lambda x: x[0])

    # Determine checkpoints to remove
    num_to_remove = len(checkpoints) - max_to_keep
    to_remove = checkpoints[:num_to_remove]

    logging.info(f"Found {len(checkpoints)} checkpoints. Keeping {max_to_keep}, removing {num_to_remove}." )

    for step, dir_path in to_remove:
        try:
            shutil.rmtree(dir_path)
            logging.info(f"Removed old checkpoint: {dir_path}")
        except OSError as e:
            logging.error(f"Error removing checkpoint {dir_path}: {e}")

def find_best_checkpoint(output_dir: str) -> Optional[str]:
    """
    Find the best checkpoint directory saved based on the monitored metric.

    Args:
        output_dir: Directory where checkpoints are saved.

    Returns:
        Path to the best checkpoint directory or None if not found.
    """
    best_checkpoint_dir = os.path.join(output_dir, "best_checkpoint")
    if os.path.exists(best_checkpoint_dir) and os.path.isdir(best_checkpoint_dir):
        # Optionally, verify it contains the best_metric.json file
        if os.path.exists(os.path.join(best_checkpoint_dir, "best_metric.json")):
            return best_checkpoint_dir
        else:
            logging.warning(f"Found {best_checkpoint_dir} but it's missing 'best_metric.json'.")
            # Depending on requirements, you might return the path anyway or be strict:
            return None # Strict: require the metric file
            # return best_checkpoint_dir # Less strict: return path if directory exists
    return None


def load_checkpoint(
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Optional[Union[str, torch.device]] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load model checkpoint with optimizer and scheduler state.
    
    Args:
        checkpoint_dir: Directory with saved checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        map_location: Device to map tensors to
        strict: Whether to strictly enforce that the keys in state_dict match
        
    Returns:
        Dictionary with loaded training state
        
    Raises:
        FileNotFoundError: If checkpoint directory or files don't exist
        FileError: If loading fails
    """
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} not found")
    
    # Load model weights
    try:
        model.from_pretrained(checkpoint_dir, map_location=map_location)
    except Exception as e:
        error_msg = f"Failed to load model weights from {checkpoint_dir}: {str(e)}"
        logging.error(error_msg)
        raise FileError(error_msg) from e
    
    # Load optimizer and scheduler
    opt_sch_path = os.path.join(checkpoint_dir, "optimizer_scheduler.pt")
    training_state = {}
    
    if os.path.exists(opt_sch_path):
        try:
            checkpoint = torch.load(opt_sch_path, map_location=map_location)
            
            # Load optimizer state
            if optimizer is not None and 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            
            # Load scheduler state
            if scheduler is not None and 'scheduler' in checkpoint and checkpoint['scheduler']:
                scheduler.load_state_dict(checkpoint['scheduler'])
            
            # Get training state
            training_state = {
                'global_step': checkpoint.get('global_step', 0),
                'epoch': checkpoint.get('epoch', 0),
                'metrics': checkpoint.get('metrics', {}),
                'timestamp': checkpoint.get('timestamp', '')
            }
            
            logging.info(f"Loaded checkpoint from {checkpoint_dir} (step: {training_state['global_step']})")
            
        except Exception as e:
            error_msg = f"Failed to load optimizer/scheduler from {opt_sch_path}: {str(e)}"
            logging.error(error_msg)
            raise FileError(error_msg) from e
    
    # Load training state from JSON if available
    json_path = os.path.join(checkpoint_dir, "training_state.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                json_state = json.load(f)
                for k, v in json_state.items():
                    if k not in training_state:
                        training_state[k] = v
        except Exception as e:
            logging.warning(f"Failed to load training state from {json_path}: {str(e)}")
    
    return training_state


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in output directory.
    
    Args:
        output_dir: Directory with checkpoints
        
    Returns:
        Path to latest checkpoint directory or None if no checkpoints found
    """
    if not os.path.exists(output_dir):
        return None
    
    # Look for "final_checkpoint" first
    final_ckpt = os.path.join(output_dir, "final_checkpoint")
    if os.path.exists(final_ckpt) and os.path.isdir(final_ckpt):
        return final_ckpt
    
    # Look for checkpoint-{step} directories
    checkpoints = []
    for d in os.listdir(output_dir):
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d)):
            try:
                step = int(d.split("-")[1])
                checkpoints.append((step, os.path.join(output_dir, d)))
            except (IndexError, ValueError):
                continue
    
    # Find checkpoint with highest step
    if checkpoints:
        _, latest_checkpoint = max(checkpoints, key=lambda x: x[0])
        return latest_checkpoint
    
    return None


def verify_checkpoint(checkpoint_dir: str) -> bool:
    """
    Verify that a checkpoint directory contains valid files.
    
    Args:
        checkpoint_dir: Directory with checkpoint
        
    Returns:
        True if checkpoint is valid, False otherwise
    """
    if not os.path.exists(checkpoint_dir) or not os.path.isdir(checkpoint_dir):
        logging.warning(f"Checkpoint directory {checkpoint_dir} not found")
        return False
    
    # Support two layouts:
    #   1) HF-style: config.json + (pytorch_model.bin | model.safetensors)
    #   2) Generic:  model.pt + config.json (our custom baseline format)

    # Check for HF-style files
    required_files = ["config.json"]
    model_files = ["pytorch_model.bin", "model.safetensors"]
    has_hf_model_file = any(os.path.exists(os.path.join(checkpoint_dir, f)) for f in model_files)
    has_config = os.path.exists(os.path.join(checkpoint_dir, "config.json"))

    # Check for generic baseline style
    has_generic_model = os.path.exists(os.path.join(checkpoint_dir, "model.pt")) and has_config
    
    if (has_hf_model_file and has_config) or has_generic_model:
        return True
    else:
        logging.warning(f"No valid checkpoint layout found in {checkpoint_dir}")
        return False


def extract_checkpoint_step(checkpoint_dir: str) -> Optional[int]:
    """
    Extract step number from checkpoint directory name.
    
    Args:
        checkpoint_dir: Checkpoint directory path
        
    Returns:
        Step number or None if not found
    """
    basename = os.path.basename(checkpoint_dir)
    if basename.startswith("checkpoint-"):
        try:
            return int(basename.split("-")[1])
        except (IndexError, ValueError):
            pass
    
    # Try to get step from training_state.json
    json_path = os.path.join(checkpoint_dir, "training_state.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                state = json.load(f)
                if 'global_step' in state:
                    return state['global_step']
        except Exception:
            pass
    
    return None


def create_checkpoint_metadata(
    model_name: str,
    global_step: int,
    epoch: int,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create metadata dictionary for checkpoint.
    
    Args:
        model_name: Name of the model
        global_step: Current global step
        epoch: Current epoch
        metrics: Evaluation metrics
        config: Model configuration
        
    Returns:
        Metadata dictionary
    """
    return {
        "model_name": model_name,
        "global_step": global_step,
        "epoch": epoch,
        "metrics": metrics or {},
        "config": config or {},
        "timestamp": datetime.now().isoformat(),
        "pytorch_version": torch.__version__
    }


def load_model_for_evaluation(
    model_path: str,
    config: Optional[Any] = None,
    device: Optional[Union[str, torch.device]] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[Any, Any]:
    """
    Load a model for evaluation from a checkpoint or Hugging Face model name.
    
    Automatically detects whether the model is a standard Segformer or TrueResSegformer
    and loads it accordingly. Also creates an appropriate image processor.
    
    Args:
        model_path: Path to checkpoint directory or Hugging Face model name
        config: Optional configuration object for TrueResSegformer
        device: Device to load the model onto (defaults to CPU if None)
        logger: Optional logger for logging messages
        
    Returns:
        Tuple of (model, image_processor)
    """
    # Set up logger
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Loading model from: {model_path}")
    
    # Set device for model loading
    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    
    try:
        # Check if it's a local path or HF model name
        is_local_path = os.path.exists(model_path)
        
        if is_local_path:
            # Try to determine if this is a TrueResSegformer checkpoint
            is_true_res = False
            model_type_indicators = [
                "TrueResSegformer" in model_path,
                "Segformer_TrueRes" in model_path,
                (config is not None and config.get("use_true_res_segformer", False))
            ]
            
            # Also check config.json if it exists
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        model_config_dict = json.load(f)
                        # Check for indicators in the config
                        is_true_res = is_true_res or model_config_dict.get("model_type", "") == "trueressegformer"
                except Exception as e:
                    logger.warning(f"Error reading config.json: {e}")
            
            is_true_res = any(model_type_indicators) or is_true_res
            
            # Load model configuration
            from transformers import AutoConfig
            try:
                model_config = AutoConfig.from_pretrained(model_path)
            except Exception as e:
                logger.warning(f"Failed to load model config with AutoConfig: {e}")
                model_config = None
            
            if is_true_res:
                # Load TrueResSegformer
                try:
                    from model import TrueResSegformer
                    
                    # Use provided config or create default
                    if config is None:
                        from config import Config
                        eval_config = Config()
                    else:
                        eval_config = config
                    
                    # Create the model instance
                    if model_config is not None:
                        model = TrueResSegformer(
                            config=model_config,
                            project_config=eval_config,
                            class_weights=None,  # Not needed for inference
                            apply_class_weights=False  # Not needed for inference
                        )
                    else:
                        # Fallback if no config
                        from transformers import SegformerConfig
                        default_config = SegformerConfig.from_pretrained("nvidia/mit-b0")
                        model = TrueResSegformer(
                            config=default_config,
                            project_config=eval_config,
                            class_weights=None,
                            apply_class_weights=False
                        )
                    
                    # Load the weights
                    checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
                    if os.path.exists(checkpoint_path):
                        state_dict = torch.load(checkpoint_path, map_location="cpu")
                        # Remap keys: strip "segformer." prefix if present
                        remapped_state_dict = {}
                        for k, v in state_dict.items():
                            if k.startswith("segformer."):
                                remapped_state_dict[k[len("segformer."):]] = v
                            else:
                                remapped_state_dict[k] = v
                        logger.info(f"Remapped state_dict keys for TrueResSegformer. Original count: {len(state_dict)}, Remapped count: {len(remapped_state_dict)}.")
                        if len(state_dict) > 10 and len(remapped_state_dict) > 10:
                             logger.debug(f"First 10 original keys: {list(state_dict.keys())[:10]}")
                             logger.debug(f"First 10 remapped keys: {list(remapped_state_dict.keys())[:10]}")
                        elif len(state_dict) > 0 :
                             logger.debug(f"Original keys: {list(state_dict.keys())}")
                             logger.debug(f"Remapped keys: {list(remapped_state_dict.keys())}")

                        load_result = model.load_state_dict(remapped_state_dict, strict=False)
                        logger.info("Successfully loaded TrueResSegformer weights")
                    else:
                        logger.warning(f"No pytorch_model.bin found at {checkpoint_path}")
                        
                    logger.info("Successfully initialized TrueResSegformer model")
                except Exception as e:
                    logger.error(f"Failed to load TrueResSegformer: {e}")
                    logger.info("Falling back to standard SegformerForSemanticSegmentation")
                    is_true_res = False
            
            # If not TrueResSegformer or fallback needed
            if not is_true_res:
                # Check if this is a PSPNet model
                is_pspnet = False
                
                # Check config.json for PSPNet architecture
                config_path = os.path.join(model_path, "config.json")
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as f:
                            model_config_dict = json.load(f)
                            is_pspnet = model_config_dict.get("architecture", "") == "pspnet"
                    except Exception as e:
                        logger.warning(f"Error reading config.json for PSPNet detection: {e}")
                
                # Also check from passed config
                if config is not None:
                    is_pspnet = is_pspnet or config.get("architecture", "") == "pspnet"
                
                if is_pspnet:
                    logger.info("Detected PSPNet architecture, loading PSPNetWrapper model")
                    try:
                        from model import PSPNetWrapper
                        
                        # Load config from checkpoint or parent directory
                        # First try checkpoint directory, then parent directory
                        pspnet_config_path = config_path
                        if not os.path.exists(pspnet_config_path):
                            # Try parent directory
                            pspnet_config_path = os.path.join(os.path.dirname(model_path), "config.json")
                        
                        if os.path.exists(pspnet_config_path):
                            with open(pspnet_config_path, 'r') as f:
                                model_config_dict = json.load(f)
                        else:
                            raise FileNotFoundError(f"Could not find config.json in {config_path} or {pspnet_config_path}")
                        
                        # Create PSPNet model
                        backbone = model_config_dict.get("backbone", "resnet50")
                        num_classes = len(model_config_dict.get("id2label", {"0": "__background__", "1": "tree"}))
                        ignore_index = model_config_dict.get("semantic_loss_ignore_index", 255)
                        
                        # Use provided config or create default
                        if config is None:
                            from config import Config
                            eval_config = Config()
                        else:
                            eval_config = config
                        
                        model = PSPNetWrapper(
                            backbone=backbone,
                            num_classes=num_classes,
                            ignore_index=ignore_index,
                            project_config=eval_config,
                            class_weights=None  # Not needed for inference
                        )
                        
                        # Load the weights
                        checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
                        if os.path.exists(checkpoint_path):
                            state_dict = torch.load(checkpoint_path, map_location="cpu")
                            model.load_state_dict(state_dict, strict=True)
                            logger.info("Successfully loaded PSPNet weights")
                        else:
                            logger.warning(f"No pytorch_model.bin found at {checkpoint_path}")
                        
                        logger.info("Successfully initialized PSPNet model")
                    except Exception as e:
                        logger.error(f"Failed to load PSPNet: {e}")
                        logger.info("Falling back to standard SegformerForSemanticSegmentation")
                        is_pspnet = False
                
                # If not PSPNet, check for SETR
                if not is_pspnet:
                    # Check if this is a SETR model
                    is_setr = False
                    
                    # Check config.json for SETR architecture
                    if os.path.exists(config_path):
                        try:
                            with open(config_path, 'r') as f:
                                model_config_dict = json.load(f)
                                is_setr = model_config_dict.get("architecture", "") == "setr"
                        except Exception as e:
                            logger.warning(f"Error reading config.json for SETR detection: {e}")
                    
                    # Also check from passed config
                    if config is not None:
                        is_setr = is_setr or config.get("architecture", "") == "setr"
                    
                    if is_setr:
                        logger.info("Detected SETR architecture, loading SETRWrapper model")
                        try:
                            from model import SETRWrapper
                            
                            # Load config from checkpoint or parent directory
                            # First try checkpoint directory, then parent directory
                            setr_config_path = config_path
                            if not os.path.exists(setr_config_path):
                                # Try parent directory
                                setr_config_path = os.path.join(os.path.dirname(model_path), "config.json")
                            
                            if os.path.exists(setr_config_path):
                                with open(setr_config_path, 'r') as f:
                                    model_config_dict = json.load(f)
                            else:
                                raise FileNotFoundError(f"Could not find config.json in {config_path} or {setr_config_path}")
                            
                            # Create SETR model
                            num_classes = len(model_config_dict.get("id2label", {"0": "__background__", "1": "tree"}))
                            ignore_index = model_config_dict.get("semantic_loss_ignore_index", 255)
                            embed_dim = model_config_dict.get("setr_embed_dim", 768)
                            patch_size = model_config_dict.get("setr_patch_size", 16)
                            
                            # Use provided config or create default
                            if config is None:
                                from config import Config
                                eval_config = Config()
                            else:
                                eval_config = config
                            
                            model = SETRWrapper(
                                num_classes=num_classes,
                                ignore_index=ignore_index,
                                project_config=eval_config,
                                class_weights=None,  # Not needed for inference
                                embed_dim=embed_dim,
                                patch_size=patch_size
                            )
                            
                            # Load the weights
                            checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
                            if os.path.exists(checkpoint_path):
                                state_dict = torch.load(checkpoint_path, map_location="cpu")
                                model.load_state_dict(state_dict, strict=True)
                                logger.info("Successfully loaded SETR weights")
                            else:
                                logger.warning(f"No pytorch_model.bin found at {checkpoint_path}")
                            
                            logger.info("Successfully initialized SETR model")
                        except Exception as e:
                            logger.error(f"Failed to load SETR: {e}")
                            logger.info("Falling back to standard SegformerForSemanticSegmentation")
                            is_setr = False
                
                # If not PSPNet or SETR, load SegFormer
                if not is_pspnet and not is_setr:
                    from transformers import SegformerForSemanticSegmentation, SegformerConfig # Added SegformerConfig
                    try:
                        # Use the 'config' dictionary passed to load_model_for_evaluation
                        # to ensure the correct model architecture is loaded.
                        hf_backbone_name = config.get('model_name') if config else None # Get HF backbone identifier, e.g., "nvidia/mit-b5"
                        
                        if not hf_backbone_name:
                            logger.warning(
                                "Key 'model_name' for backbone (e.g., 'nvidia/mit-b5') not found in 'config' dict. "
                                "Attempting to load model without explicit architecture. This may fail if the checkpoint "
                                f"at '{model_path}' does not contain a compatible 'config.json' or if it implies a "
                                "different architecture than the one intended for this run."
                            )
                            # Fallback to original method: load model using only model_path.
                            # This relies on model_path containing a valid config.json or being a HF model ID.
                            model = SegformerForSemanticSegmentation.from_pretrained(model_path)
                            logger.info(f"Successfully loaded standard SegformerForSemanticSegmentation model from '{model_path}' (no explicit backbone override from config dict).")
                        else:
                            logger.info(f"Explicitly configuring Segformer model architecture using backbone: '{hf_backbone_name}' "
                                        f"and loading weights from '{model_path}'.")
                            
                            num_labels = config.get('num_labels')
                            id2label = config.get('id2label')
                            label2id = config.get('label2id')

                            # Infer num_labels if not provided but id2label or label2id is available
                            if num_labels is None:
                                if id2label is not None and isinstance(id2label, dict):
                                    num_labels = len(id2label)
                                    logger.info(f"Inferred num_labels={num_labels} from length of id2label.")
                                elif label2id is not None and isinstance(label2id, dict):
                                    num_labels = len(label2id)
                                    logger.info(f"Inferred num_labels={num_labels} from length of label2id.")
                                else:
                                    logger.warning(
                                        f"Key 'num_labels' not found in 'config' dict for backbone '{hf_backbone_name}', "
                                        "and could not be inferred from id2label/label2id. "
                                        "The number of labels will be determined by the backbone's default configuration. "
                                        "Ensure this is intended, especially for classification heads."
                                    )
                            
                            # Create the SegformerConfig object based on the specified backbone and any overrides from the config dict.
                            seg_model_config = SegformerConfig.from_pretrained(
                                pretrained_model_name_or_path=hf_backbone_name, # e.g., "nvidia/mit-b5"
                                num_labels=num_labels,   # Pass None if not in config; from_pretrained will use backbone's default
                                id2label=id2label,       # Pass None if not in config
                                label2id=label2id        # Pass None if not in config
                                # Add any other relevant config overrides here if needed, e.g., image_size from config dict
                            )
                            
                            # Load the model weights from model_path, but use the explicitly created seg_model_config
                            # to define the model's architecture.
                            model = SegformerForSemanticSegmentation.from_pretrained(
                                pretrained_model_name_or_path=model_path, # Path to the checkpoint (directory or .bin file)
                                config=seg_model_config                   # The explicit SegformerConfig object
                            )
                            logger.info(
                                f"Successfully loaded standard SegformerForSemanticSegmentation model from '{model_path}' "
                                f"with architecture explicitly configured from backbone '{hf_backbone_name}'."
                            )
                    except Exception as e:
                        # Do not raise here; allow generic fallback loader (model.pt) below
                        logger.error(f"Failed to load standard Segformer from checkpoint: {e}")
                        # Intentionally not raising to try generic checkpoint formats next
        else:
            # For HF model names, just use the standard loading
            try:
                from transformers import SegformerForSemanticSegmentation
                model = SegformerForSemanticSegmentation.from_pretrained(model_path)
                logger.info(f"Successfully loaded model from Hugging Face: {model_path}")
            except Exception as e:
                logger.error(f"Failed to load Hugging Face model {model_path}: {e}")
                raise
        
        # If neither HF nor TrueRes was resolved and this is a local path, try generic baseline layout
        if is_local_path and 'model' not in locals():
            generic_model_path = os.path.join(model_path, "model.pt")
            generic_cfg_path = os.path.join(model_path, "config.json")
            if os.path.exists(generic_model_path) and os.path.exists(generic_cfg_path):
                with open(generic_cfg_path, 'r') as f:
                    cfg_json = json.load(f)
                arch = cfg_json.get("architecture", "segformer")
                num_labels = len(cfg_json.get("id2label", {0:"__background__",1:"tree"}))
                ignore_index = cfg_json.get("ignore_index", 255)
                # Build the appropriate wrapper model
                if arch == "unet":
                    try:
                        from model import UNetWrapper
                        backbone = cfg_json.get("backbone", "resnet34")
                        model = UNetWrapper(
                            backbone=backbone,
                            num_classes=num_labels,
                            ignore_index=ignore_index,
                            project_config=config if config is not None else cfg_json,
                            class_weights=None
                        )
                    except Exception as e:
                        logger.error(f"Failed to construct UNetWrapper: {e}")
                        raise
                elif arch == "deeplabv3plus":
                    try:
                        from model import DeepLabV3PlusWrapper
                        backbone = cfg_json.get("backbone", "resnet50")
                        model = DeepLabV3PlusWrapper(
                            backbone=backbone,
                            num_classes=num_labels,
                            ignore_index=ignore_index,
                            project_config=config if config is not None else cfg_json,
                            class_weights=None
                        )
                    except Exception as e:
                        logger.error(f"Failed to construct DeepLabV3PlusWrapper: {e}")
                        raise
                elif arch == "pspnet":
                    try:
                        from model import PSPNetWrapper
                        backbone = cfg_json.get("backbone", "resnet50")
                        model = PSPNetWrapper(
                            backbone=backbone,
                            num_classes=num_labels,
                            ignore_index=ignore_index,
                            project_config=config if config is not None else cfg_json,
                            class_weights=None
                        )
                    except Exception as e:
                        logger.error(f"Failed to construct PSPNetWrapper: {e}")
                        raise
                elif arch == "setr":
                    try:
                        from model import SETRWrapper
                        embed_dim = cfg_json.get("setr_embed_dim", 768)
                        patch_size = cfg_json.get("setr_patch_size", 16)
                        model = SETRWrapper(
                            num_classes=num_labels,
                            ignore_index=ignore_index,
                            project_config=config if config is not None else cfg_json,
                            class_weights=None,
                            embed_dim=embed_dim,
                            patch_size=patch_size
                        )
                    except Exception as e:
                        logger.error(f"Failed to construct SETRWrapper: {e}")
                        raise
                else:
                    logger.error(f"Unsupported architecture in generic checkpoint: {arch}")
                    raise ValueError(f"Unsupported architecture in generic checkpoint: {arch}")

                # Load weights
                state_dict = torch.load(generic_model_path, map_location="cpu")
                model.load_state_dict(state_dict, strict=True)

                # Attach a minimal HF-like config shim so downstream code can access id2label/label2id
                try:
                    # Ensure id2label keys are ints
                    id2label_map = cfg_json.get("id2label", {})
                    id2label_map = {int(k): v for k, v in id2label_map.items()} if isinstance(id2label_map, dict) else {}
                    label2id_map = cfg_json.get("label2id", {})
                    model.config = SimpleNamespace(
                        id2label=id2label_map,
                        label2id=label2id_map
                    )
                except Exception as e:
                    logger.warning(f"Failed to attach config shim to generic model: {e}")
            else:
                logger.error("Local checkpoint path is not HF-style and no model.pt found.")
                raise FileNotFoundError("Unsupported checkpoint layout")

        # Move model to device and build processor
        model = model.to(device)
        from transformers import SegformerImageProcessor
        image_processor = SegformerImageProcessor(
            do_resize=False,
            do_rescale=True,
            do_normalize=True
        )
        return model, image_processor
    
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}", exc_info=True)
        raise
