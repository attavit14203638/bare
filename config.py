#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Centralized configuration management for BARE model.

This module provides a consistent way to manage configuration across the codebase,
reducing duplication and enforcing consistent parameter handling.
"""

import os
import json
import copy
from typing import Dict, Any, Optional, Union, List, Tuple
from utils import LOGGER_NAME, get_logger

# Setup module logger
logger = get_logger()

# Default configuration values
DEFAULT_CONFIG = {
    # Architecture selection
    # segformer: uses Hugging Face backbone via `model_name`
    # pspnet: uses `backbone` (e.g., 'resnet50') via segmentation-models-pytorch
    # setr: uses timm's Vision Transformer (configurable via setr_embed_dim, setr_patch_size)
    
    "architecture": "setr",
    "backbone": "resnet50", # Use with PSPNet
    # Dataset parameters
    "dataset_name": "restor/tcd",
    "image_size": None, 
    "validation_split": 0.1,  

    # Model parameters
    "model_name": "nvidia/mit-b5", #Use with SegFormer 
    "train_time_upsample": False,  
    "interpolation_mode": "bilinear", 
    "interpolation_align_corners": False,
    
    # SETR-specific parameters
    "setr_embed_dim": 768,
    "setr_patch_size": 16,
    "setr_input_size": 1024,  # Native input resolution for SETR (enables 1024x1024 processing)

    # Training parameters 
    "output_dir": "./outputs_setr_baseline",
    "train_batch_size": 8,
    "eval_batch_size": 16,
    "num_epochs": 50,
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "num_workers": 8,
    "seed": 42,
    "gradient_accumulation_steps": 1,
    "class_weights_enabled": False,  
    "gradient_checkpointing": True,
    "mixed_precision": True,

    # Scheduler parameters
    "scheduler_type": "cosine_with_restarts",
    #"scheduler_monitor": "val_loss",
    #"scheduler_mode": "min",  
    #"scheduler_patience": 5,  # RoP
    #"scheduler_factor": 0., # RoP
    "min_lr_scheduler": 1e-8, # CwR
    "warmup_ratio": 0.1, # CwR
    "num_cycles": 1.0, # CwR  
    "power": 1.0, # CwR

    # DataLoader parameters 
    "dataloader_pin_memory": True,
    "dataloader_persistent_workers": True,
    "dataloader_prefetch_factor": 2,

    # Inference parameters
    "inference_tile_size": 1024,
    "inference_tile_overlap": 320,  

    # Augmentation parameters
    "augmentation": {
        "apply": True,
        "random_crop_size": 1024,
        "h_flip_prob": 0.5,
        "v_flip_prob": 0.5,
        "rotation_degrees": 180,
        "color_jitter_prob": 0.7,  
        "brightness": 0.3,  
        "contrast": 0.3, 
        "saturation": 0.3,  
        "hue": 0.1,
        "gaussian_blur_prob": 0.5,
        "gaussian_blur_kernel_size": [3, 7],
        "gaussian_blur_sigma": [0.1, 2.0],
        "atmospheric_prob": 0.2,  
        "shadow_prob": 0.2,  
        "visualize_augmented_samples": True,
        "num_augmented_samples_to_visualize": 5
    },

    # Alt. size parameters 

    "training_tile_size": None, 
    "train_tile_stride": None,
    "train_tile_random_offset": False,
    "train_tile_threshold": 0.0,
    "inference_cpu_offload": False,
    
    # Cross-validation parameters
    "cross_validation": {
        "enabled": False,
        "num_folds": 5,
        "save_best_model_per_fold": True,
        "metrics_to_track": ["f1_score_class_1", "IoU_class_1"]
    },
    
    # Logging parameters
    "logging_steps": 25,
    "eval_steps": 250,  
    "save_steps": 500,
    "save_best_checkpoint": True,
    "best_checkpoint_metric": "val_loss",
    "best_checkpoint_monitor_mode": "min",
    "max_checkpoints_to_keep": 3,

    # Checkpoint resumption parameters
    "resume_from_checkpoint": False, # Master switch for resuming model weights and OPTIONALLY optimizer state
    # "resume_from_checkpoint_path": None, # Path to model weights (e.g., "./outputs/best_checkpoint/pytorch_model.bin")
    # "resume_optimizer_path": None, # Path to optimizer state (e.g., "./outputs/best_checkpoint/optimizer.pt"), can be None if not resuming optimizer
    
    # Worst visualization parameters
    "visualize_worst": False,
    "num_worst_samples": 5,
    "visualize_confidence_comparison": False,
    "analyze_errors": False,
    
    #Class parameters
    "id2label": {0: "__background__", 1: "tree"},
    "label2id": {"__background__": 0, "tree": 1}
}


class Config:
    """
    Configuration class for BARE.

    This class manages configuration parameters with validation,
    saving/loading, and provides a consistent interface for all
    configuration needs in the project.
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize configuration with default values, overridden by provided config.

        Args:
            config_dict: Optional dictionary with configuration values to override defaults
        """
        # Start with default configuration
        self._config = copy.deepcopy(DEFAULT_CONFIG)

        # Override with provided configuration if any
        if config_dict:
            self._update_config(config_dict)

    def _update_config(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration with values from dictionary.

        Args:
            config_dict: Dictionary with configuration values
        """
        for key, value in config_dict.items():
            if key not in self._config:
                # Allow nested updates for augmentation dictionary
                if key == "augmentation" and isinstance(value, dict):
                    if "augmentation" not in self._config:
                        self._config["augmentation"] = {}
                    for aug_key, aug_value in value.items():
                        if aug_key not in self._config["augmentation"]:
                             logger.warning(f"Unknown augmentation parameter: {aug_key}")
                        self._config["augmentation"][aug_key] = aug_value
                else:
                    logger.warning(f"Unknown configuration parameter: {key}")
            self._config[key] = value


    def __getitem__(self, key: str) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration parameter name

        Returns:
            Configuration value

        Raises:
            KeyError: If key doesn't exist in configuration
        """
        if key not in self._config:
            raise KeyError(f"Configuration parameter '{key}' not found")
        return self._config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set configuration value.

        Args:
            key: Configuration parameter name
            value: Configuration value
        """
        self._config[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with default fallback.

        Args:
            key: Configuration parameter name
            default: Default value if key doesn't exist

        Returns:
            Configuration value or default
        """
        return self._config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary with configuration values
        """
        return copy.deepcopy(self._config)

    def save(self, path: str) -> None:
        """
        Save configuration to JSON file with error handling.

        Args:
            path: Path to save configuration to

        Raises:
            FileError: If saving the configuration fails
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(self._config, f, indent=2)
        except Exception as e:
            from exceptions import FileError
            raise FileError(f"Failed to save configuration to {path}: {e}")

    @classmethod
    def load(cls, path: str) -> 'Config':
        """
        Load configuration from JSON file.

        Args:
            path: Path to load configuration from

        Returns:
            Loaded configuration

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            json.JSONDecodeError: If configuration file is invalid JSON
        """
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls(config_dict)

    def validate(self) -> List[str]:
        """
        Validate configuration values.

        Returns:
            List of validation errors, empty if configuration is valid
        """
        errors = []

        # Validate image_size (allow None initially)
        img_size = self._config.get("image_size")
        if img_size is not None and (not isinstance(img_size, list) or len(img_size) != 2 or not all(isinstance(x, int) and x > 0 for x in img_size)):
            errors.append("image_size must be None or a list of two positive integers")

        # Validate batch sizes
        if self._config.get("train_batch_size", 0) <= 0:
            errors.append("train_batch_size must be positive")
        if self._config.get("eval_batch_size", 0) <= 0:
            errors.append("eval_batch_size must be positive")

        # Validate learning rate
        if self._config.get("learning_rate", 0) <= 0:
            errors.append("learning_rate must be positive")

        # Validate output_dir
        if not self._config.get("output_dir"):
            errors.append("output_dir must be specified")

        # Validate id2label and label2id are consistent (if both exist)
        id2label = self._config.get("id2label")
        label2id = self._config.get("label2id")
        if id2label and label2id:
            if not isinstance(id2label, dict) or not isinstance(label2id, dict):
                 errors.append("id2label and label2id must be dictionaries")
            else:
                for id_val, label in id2label.items():
                    # Ensure id_val is int or can be converted
                    try:
                        id_int = int(id_val)
                    except ValueError:
                        errors.append(f"Key '{id_val}' in id2label is not a valid integer.")
                        continue

                    if label not in label2id or label2id[label] != id_int:
                        errors.append(f"Inconsistency between id2label and label2id for id {id_int} and label '{label}'")
                for label, id_val in label2id.items():
                    if id_val not in id2label or id2label[id_val] != label:
                         # Check if key exists as int or string in id2label
                         if str(id_val) not in id2label or id2label[str(id_val)] != label:
                             errors.append(f"Inconsistency between label2id and id2label for label '{label}' and id {id_val}")


        # Validate architecture/backbone
        arch = self._config.get("architecture", "segformer")
        supported_archs = ["segformer", "pspnet", "setr"]
        if arch not in supported_archs:
            errors.append(f"architecture must be one of {supported_archs}, got {arch}")
        
        # Validate backbone for architectures that require it
        backbone_required_archs = ["pspnet"]
        if arch in backbone_required_archs:
            bb = self._config.get("backbone")
            if not isinstance(bb, str) or len(bb.strip()) == 0:
                errors.append(f"backbone must be a non-empty string for '{arch}'")
        
        # Validate SETR-specific parameters
        if arch == "setr":
            embed_dim = self._config.get("setr_embed_dim", 768)
            if not isinstance(embed_dim, int) or embed_dim <= 0:
                errors.append("setr_embed_dim must be a positive integer")
            patch_size = self._config.get("setr_patch_size", 16)
            if not isinstance(patch_size, int) or patch_size <= 0:
                errors.append("setr_patch_size must be a positive integer")
            input_size = self._config.get("setr_input_size", 1024)
            if not isinstance(input_size, int) or input_size <= 0:
                errors.append("setr_input_size must be a positive integer")
            # Validate that input_size is divisible by patch_size for proper patch tokenization
            if input_size % patch_size != 0:
                errors.append(f"setr_input_size ({input_size}) must be divisible by setr_patch_size ({patch_size})")

        # Validate scheduler parameters
        allowed_schedulers = ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "reduce_on_plateau"]
        scheduler_type = self._config.get("scheduler_type", "linear")
        if scheduler_type not in allowed_schedulers:
            errors.append(f"scheduler_type must be one of {allowed_schedulers}, got {scheduler_type}")

        warmup_ratio = self._config.get("warmup_ratio", 0.0)
        if not (0 <= warmup_ratio <= 1):
            errors.append(f"warmup_ratio must be between 0 and 1, got {warmup_ratio}")

        num_cycles = self._config.get("num_cycles", 0.5)
        if scheduler_type == "cosine_with_restarts" and num_cycles <= 0:
            errors.append(f"num_cycles must be positive for cosine_with_restarts scheduler, got {num_cycles}")

        power = self._config.get("power", 1.0)
        if scheduler_type == "polynomial" and power <= 0:
             errors.append(f"power must be positive for polynomial scheduler, got {power}")

        min_lr_sched = self._config.get("min_lr_scheduler", 0)
        if not isinstance(min_lr_sched, (int, float)) or min_lr_sched < 0:
            errors.append(f"min_lr_scheduler must be a non-negative number, got {min_lr_sched}")

        # Validate augmentation parameters
        aug_config = self._config.get("augmentation", {})
        if not isinstance(aug_config, dict):
             errors.append("augmentation config must be a dictionary")
        else:
            if not isinstance(aug_config.get("apply", True), bool):
                errors.append("augmentation.apply must be a boolean")
            if not (0 <= aug_config.get("h_flip_prob", 0.5) <= 1):
                errors.append("augmentation.h_flip_prob must be between 0 and 1")
            if not (0 <= aug_config.get("v_flip_prob", 0.5) <= 1):
                errors.append("augmentation.v_flip_prob must be between 0 and 1")
            if not (aug_config.get("rotation_degrees", 180) >= 0):
                 errors.append("augmentation.rotation_degrees must be non-negative")
            if not (0 <= aug_config.get("color_jitter_prob", 0.5) <= 1):
                errors.append("augmentation.color_jitter_prob must be between 0 and 1")
            if not (aug_config.get("brightness", 0.25) >= 0):
                 errors.append("augmentation.brightness must be non-negative")
            if not (aug_config.get("contrast", 0.25) >= 0):
                 errors.append("augmentation.contrast must be non-negative")
            if not (aug_config.get("saturation", 0.25) >= 0):
                 errors.append("augmentation.saturation must be non-negative")
            if not (0 <= aug_config.get("hue", 0.1) <= 0.5):
                 errors.append("augmentation.hue must be between 0 and 0.5")

            if not (0 <= aug_config.get("gaussian_blur_prob", 0.5) <= 1):
                errors.append("augmentation.gaussian_blur_prob must be between 0 and 1")
            kernel_size = aug_config.get("gaussian_blur_kernel_size", [3, 7])
            if isinstance(kernel_size, int):
                if kernel_size <= 0 or kernel_size % 2 == 0:
                    errors.append("augmentation.gaussian_blur_kernel_size (int) must be a positive odd integer")
            elif isinstance(kernel_size, (list, tuple)) and len(kernel_size) == 2:
                if not all(isinstance(x, int) and x > 0 for x in kernel_size) or kernel_size[0] > kernel_size[1]:
                    errors.append("augmentation.gaussian_blur_kernel_size (range) must be two positive integers (min, max)")
            else:
                errors.append("augmentation.gaussian_blur_kernel_size must be a positive odd int or a list/tuple of two positive ints [min, max]")

            sigma = aug_config.get("gaussian_blur_sigma", [0.1, 2.0])
            if isinstance(sigma, (float, int)):
                if sigma <= 0:
                    errors.append("augmentation.gaussian_blur_sigma (float) must be positive")
            elif isinstance(sigma, (list, tuple)) and len(sigma) == 2:
                 if not all(isinstance(x, (float, int)) and x > 0 for x in sigma) or sigma[0] > sigma[1]:
                     errors.append("augmentation.gaussian_blur_sigma (range) must be two positive floats/ints (min, max)")
            else:
                 errors.append("augmentation.gaussian_blur_sigma must be a positive float/int or a list/tuple of two positive floats/ints [min, max]")

            if not (0 <= aug_config.get("atmospheric_prob", 0.0) <= 1):
                errors.append("augmentation.atmospheric_prob must be between 0 and 1")
            if not (0 <= aug_config.get("shadow_prob", 0.0) <= 1):
                errors.append("augmentation.shadow_prob must be between 0 and 1")
            if not isinstance(aug_config.get("visualize_augmented_samples", False), bool):
                errors.append("augmentation.visualize_augmented_samples must be a boolean")
            if not (aug_config.get("num_augmented_samples_to_visualize", 5) >= 0):
                 errors.append("augmentation.num_augmented_samples_to_visualize must be non-negative")

        # Validate mixed precision backend
        allowed_backends = ["cuda_amp", "native_amp"]
        mp_backend = self._config.get("mixed_precision_backend", "cuda_amp")
        if mp_backend not in allowed_backends:
            errors.append(f"mixed_precision_backend must be one of {allowed_backends}, got {mp_backend}")

        # Validate gradient checkpointing
        if not isinstance(self._config.get("gradient_checkpointing", False), bool):
            errors.append("gradient_checkpointing must be a boolean")

        # Validate train_time_upsample
        if not isinstance(self._config.get("train_time_upsample", False), bool):
            errors.append("train_time_upsample must be a boolean")

        # Validate DataLoader parameters
        if not isinstance(self._config.get("dataloader_pin_memory", True), bool):
            errors.append("dataloader_pin_memory must be a boolean")
        if not isinstance(self._config.get("dataloader_persistent_workers", True), bool):
            errors.append("dataloader_persistent_workers must be a boolean")
        if not isinstance(self._config.get("dataloader_prefetch_factor", 2), int) or self._config.get("dataloader_prefetch_factor", 2) < 0:
             errors.append("dataloader_prefetch_factor must be a non-negative integer")

        # Validate Inference parameters
        if not isinstance(self._config.get("inference_cpu_offload", False), bool):
            errors.append("inference_cpu_offload must be a boolean")
        tile_size = self._config.get("inference_tile_size", 1024)
        if tile_size is not None and (not isinstance(tile_size, int) or tile_size <= 0):
            errors.append("inference_tile_size must be a positive integer or None")
        tile_overlap = self._config.get("inference_tile_overlap", 256)
        if not isinstance(tile_overlap, int) or tile_overlap < 0:
            errors.append("inference_tile_overlap must be a non-negative integer")
        if tile_size is not None and tile_overlap >= tile_size:
            errors.append("inference_tile_overlap must be smaller than inference_tile_size")

        # Validate error analysis flag
        if not isinstance(self._config.get("analyze_errors", False), bool):
            errors.append("analyze_errors must be a boolean")

        # Validate interpolation parameters
        allowed_modes = ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area']
        interp_mode = self._config.get("interpolation_mode", "bilinear")
        if not isinstance(interp_mode, str) or interp_mode not in allowed_modes:
            errors.append(f"interpolation_mode must be one of {allowed_modes}, got {interp_mode}")
        align_corners = self._config.get("interpolation_align_corners", False)
        if not isinstance(align_corners, bool):
             if align_corners is not None:
                 errors.append("interpolation_align_corners must be a boolean or None")

        # Validate Training Tiling parameters
        train_tile_size = self._config.get("train_tile_size", None)
        train_tile_stride = self._config.get("train_tile_stride", None)
        train_tile_random_offset = self._config.get("train_tile_random_offset", False)
        train_tile_threshold = self._config.get("train_tile_threshold", 0.0)

        if train_tile_size is not None:
            if isinstance(train_tile_size, int):
                if train_tile_size <= 0:
                    errors.append("train_tile_size must be a positive integer if specified as int")
            elif isinstance(train_tile_size, list):
                if len(train_tile_size) != 2 or not all(isinstance(x, int) and x > 0 for x in train_tile_size):
                    errors.append("train_tile_size must be a list of two positive integers if specified as list")
            else:
                errors.append("train_tile_size must be an integer, a list of two integers, or None")

            if train_tile_stride is None:
                pass
            elif isinstance(train_tile_stride, int):
                if train_tile_stride <= 0:
                    errors.append("train_tile_stride must be a positive integer if specified as int")
                if isinstance(train_tile_size, int) and train_tile_stride > train_tile_size:
                     errors.append("train_tile_stride cannot be larger than train_tile_size")
                elif isinstance(train_tile_size, list) and (train_tile_stride > train_tile_size[0] or train_tile_stride > train_tile_size[1]):
                     errors.append("train_tile_stride cannot be larger than train_tile_size dimensions")
            elif isinstance(train_tile_stride, list):
                 if len(train_tile_stride) != 2 or not all(isinstance(x, int) and x > 0 for x in train_tile_stride):
                     errors.append("train_tile_stride must be a list of two positive integers if specified as list")
                 if isinstance(train_tile_size, list) and (train_tile_stride[0] > train_tile_size[0] or train_tile_stride[1] > train_tile_size[1]):
                     errors.append("train_tile_stride dimensions cannot be larger than train_tile_size dimensions")
            else:
                 errors.append("train_tile_stride must be an integer, a list of two integers, or None")

            if not isinstance(train_tile_random_offset, bool):
                errors.append("train_tile_random_offset must be a boolean")

            if not (0.0 <= train_tile_threshold <= 1.0):
                errors.append("train_tile_threshold must be between 0.0 and 1.0")

        # --- NEW: Validate Checkpoint parameters ---
        if not isinstance(self._config.get("save_best_checkpoint", True), bool):
            errors.append("save_best_checkpoint must be a boolean")
        if not isinstance(self._config.get("best_checkpoint_metric", "iou_class_1"), str):
            errors.append("best_checkpoint_metric must be a string")
        monitor_mode = self._config.get("best_checkpoint_monitor_mode", "max")
        if monitor_mode not in ['max', 'min']:
            errors.append("best_checkpoint_monitor_mode must be either 'max' or 'min'")
        max_to_keep = self._config.get("max_checkpoints_to_keep", 3)
        if not isinstance(max_to_keep, int) or max_to_keep < 0:
             errors.append("max_checkpoints_to_keep must be a non-negative integer")
        # --- End NEW Validation ---

        return errors


def load_config_from_args(args) -> Config:
    """
    Create configuration solely from command-line arguments namespace.

    Note: This ignores DEFAULT_CONFIG and only uses provided args.
    Consider using load_config_from_file_and_args for merging defaults, file, and args.

    Args:
        args: Command-line arguments namespace (from argparse.parse_args())

    Returns:
        Configuration object based ONLY on args provided.
    """
    config_dict = {k: v for k, v in vars(args).items() if v is not None}
    conf = Config() 
    conf._update_config(config_dict)
    return conf

def load_config_from_file_and_args(config_path: Optional[str], args) -> Config:
    """
    Load configuration by merging defaults, a JSON file (optional), and CLI arguments.

    Priority: CLI args > JSON file > Defaults

    Args:
        config_path: Optional path to a base JSON configuration file.
        args: Command-line arguments namespace (from argparse.parse_args()).
              Arguments set to None in args are ignored.

    Returns:
        Merged Configuration object.

    Raises:
        FileNotFoundError: If the specified config file doesn't exist.
        ConfigurationError: If the configuration file is invalid or validation fails.
    """
    from exceptions import ConfigurationError, FileError

    conf = Config()

    if config_path:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path, "r") as f:
                file_config = json.load(f)
            conf._update_config(file_config)
            logger.info(f"Loaded base configuration from {config_path}")
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in config file {config_path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load config file {config_path}: {e}")

    cli_config = {k: v for k, v in vars(args).items() if v is not None and k != 'config_path' and k != 'func'}
    if cli_config:
        conf._update_config(cli_config)
        logger.info(f"Overriding configuration with CLI arguments: {cli_config}")

    errors = conf.validate()
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"- {e}" for e in errors)
        logger.error(error_msg)
        raise ConfigurationError(error_msg)

    return conf


def get_optimizer_config(config: Config) -> Dict[str, Any]:
    """
    Get optimizer configuration from global configuration.

    Args:
        config: Global configuration

    Returns:
        Optimizer configuration dictionary
    """
    return {
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "betas": (0.9, 0.999),
        "eps": 1e-8
    }


def get_training_config(config: Config) -> Dict[str, Any]:
    """
    Get training configuration from global configuration.

    Args:
        config: Global configuration

    Returns:
        Training configuration dictionary
    """
    return {
        "num_epochs": config["num_epochs"],
        "mixed_precision": config["mixed_precision"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "logging_steps": config["logging_steps"],
        "eval_steps": config["eval_steps"],
        "save_steps": config["save_steps"]
    }


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.

    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary

    Returns:
        Merged configuration dictionary
    """
    result = copy.deepcopy(base_config)
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def adjust_config_for_architecture(config: Config) -> Config:
    """
    Adjusts configuration based on the selected architecture.
    Args:
        config: The configuration object.
    Returns:
        The adjusted configuration object.
    """
    arch = config.get("architecture", "segformer")
    logger.info(f"Adjusting configuration for architecture: {arch}")

    if arch == "setr":
        if 'setr_input_size' in config._config and 'augmentation' in config._config:
            input_size = config['setr_input_size']
            config['augmentation']['random_crop_size'] = input_size
            logger.info(f"SETR architecture selected. Set random_crop_size to {input_size}")

    elif arch == "segformer":
        # Example of a segformer-specific adjustment
        # config['model_name'] = 'nvidia/segformer-b5-finetuned-ade-640-640'
        logger.info("SegFormer architecture selected. No specific adjustments needed at this time.")

    # Add other architecture-specific adjustments here
    # elif arch == "pspnet":
    #     ...

    return config
