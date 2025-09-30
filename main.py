#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified Command-Line Interface for BARE.

This script serves as the main entry point for interacting with the BARE
project. It provides subcommands for training, evaluation, prediction, and dataset inspection.

Subcommands:
  train     Train a new segmentation model.
  predict   Generate segmentation predictions for input images using a trained model.
  evaluate  Evaluate a trained model on a dataset.
  inspect   Inspect dataset samples and analyze dataset statistics.

Use 'python main.py <subcommand> --help' for specific options.
"""

import os
import argparse
import torch
import time
import logging
from typing import Dict, Optional, Any, Tuple
import sys # Added sys for exit

# Import the centralized configuration and pipeline modules
from config import Config, load_config_from_args, load_config_from_file_and_args
from pipeline import run_training_pipeline, run_prediction_pipeline, evaluate_model
from utils import setup_logging, log_or_print, get_logger # Added get_logger
from dataset import load_and_shuffle_dataset, create_dataloaders, create_eval_dataloader, TCDDataset # Added TCDDataset
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from checkpoint import verify_checkpoint
from exceptions import ConfigurationError, FileNotFoundError, DatasetError
from inspect_dataset import examine_raw_annotations, examine_dataset_statistics, inspect_dataset_samples, verify_training_tiling # Import verify_training_tiling

# --- Argument Parsing Setup ---

def create_parent_parser() -> argparse.ArgumentParser:
    """Creates a parent parser for common arguments."""
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--output_dir", type=str, default=None,
                               help="Base directory for outputs (logs, models, visualizations). Overrides config.")
    parent_parser.add_argument("--config_path", type=str, default=None,
                               help="Path to a base JSON config file to load (required for predict/evaluate).")
    parent_parser.add_argument("--seed", type=int, default=None,
                               help="Random seed for reproducibility (overrides config).")
    parent_parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False,
                               help="Enable verbose logging.")
    return parent_parser

def setup_train_parser(subparsers, parent_parser):
    """Adds arguments for the 'train' subcommand."""
    parser = subparsers.add_parser("train", help="Train a new model", parents=[parent_parser])
    # Dataset parameters (specific overrides)
    parser.add_argument("--dataset_name", type=str,
                        help="HuggingFace dataset name (overrides config)")
    parser.add_argument("--image_size", type=int, nargs=2,
                        help="Size to resize images to (overrides config)")
    parser.add_argument("--validation_split", type=float,
                        help="Fraction for validation split (overrides config)")

    # Model parameters
    parser.add_argument("--model_name", type=str,
                        help="Base model name (overrides config)")
    parser.add_argument("--architecture", type=str, choices=["segformer","pspnet","setr"],
                        help="Model architecture: segformer | pspnet | setr (overrides config)")
    parser.add_argument("--backbone", type=str,
                        help="Backbone for pspnet (e.g., resnet50). Overrides config")
    parser.add_argument("--output_dir", type=str,
                        help="Directory for outputs (overrides config)")

    # Training parameters
    parser.add_argument("--train_batch_size", type=int,
                        help="Training batch size (overrides config)")
    parser.add_argument("--eval_batch_size", type=int,
                        help="Evaluation batch size (overrides config)")
    parser.add_argument("--num_epochs", type=int,
                        help="Number of training epochs (overrides config)")
    parser.add_argument("--learning_rate", type=float,
                        help="Learning rate (overrides config)")
    parser.add_argument("--weight_decay", type=float,
                        help="Weight decay (overrides config)")
    parser.add_argument("--seed", type=int,
                        help="Random seed (overrides config)")
    parser.add_argument("--mixed_precision", action=argparse.BooleanOptionalAction,
                        help="Use mixed precision (overrides config)")
    parser.add_argument("--gradient_accumulation_steps", type=int,
                        help="Gradient accumulation steps (overrides config)")
    parser.add_argument("--num_workers", type=int,
                        help="Dataloader workers (overrides config)")
    parser.add_argument("--compute_class_weights", action=argparse.BooleanOptionalAction,
                        help="Compute class weights (overrides config)")
    parser.add_argument("--train_time_upsample", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable train-time upsampling for all models during training (overrides config).")

    # Scheduler parameters (specific overrides)
    # Note: scheduler_type, patience, factor are already in config.py's DEFAULT_CONFIG
    # and can be overridden by a JSON config file.
    # Adding min_lr_scheduler here for direct CLI control.
    parser.add_argument("--min_lr_scheduler", type=float, default=None,
                        help="Minimum learning rate for schedulers like ReduceLROnPlateau (overrides config).")
    # warmup_ratio, num_cycles, power are also in DEFAULT_CONFIG.

    # Logging parameters
    parser.add_argument("--logging_steps", type=int,
                        help="Log every X steps (overrides config)")
    parser.add_argument("--eval_steps", type=int,
                        help="Evaluate every X steps (overrides config)")
    parser.add_argument("--save_steps", type=int,
                        help="Save checkpoint every X steps (overrides config)")

    # Visualization parameters
    parser.add_argument("--visualize_worst", action=argparse.BooleanOptionalAction, default=True,
                        help="Visualize worst predictions during eval (overrides config)")
    parser.add_argument("--num_worst_samples", type=int, default=None,
                        help="Number of worst samples to show (overrides config)")
    # Add flags for evaluation options during training
    parser.add_argument("--analyze_errors", action=argparse.BooleanOptionalAction, default=None,
                        help="Perform error analysis during evaluation steps (overrides config).")
    parser.add_argument("--visualize_confidence_comparison", action=argparse.BooleanOptionalAction, default=None,
                        help="Visualize confidence comparison during evaluation steps (overrides config).")

    parser.set_defaults(func=handle_train)

def setup_predict_parser(subparsers, parent_parser):
    """Adds arguments for the 'predict' subcommand."""
    parser = subparsers.add_parser("predict", help="Make predictions with a trained model", parents=[parent_parser])
    # Make config_path required for predict
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to the JSON config file from training output.")
    parser.add_argument("--image_paths", type=str, required=True, nargs='+',
                        help="Path(s) to input image(s).")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the trained model directory (defaults to output_dir/final_model).")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for prediction.")
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=True,
                        help="Generate and save prediction visualizations.")
    parser.add_argument("--show_confidence", action=argparse.BooleanOptionalAction, default=False,
                        help="Generate and save confidence map visualizations.")
    parser.add_argument("--show_class_activation_maps", action=argparse.BooleanOptionalAction, default=False,
                        help="Generate and save class activation map visualizations.")
    parser.set_defaults(func=handle_predict)

def setup_evaluate_parser(subparsers, parent_parser):
    """Adds arguments for the 'evaluate' subcommand."""
    parser = subparsers.add_parser("evaluate", help="Evaluate a trained model", parents=[parent_parser])
    # Make config_path required for evaluate
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to the JSON config file from training output.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model directory to evaluate.")
    # Dataset/Eval specific overrides
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Dataset name (overrides config).")
    parser.add_argument("--eval_batch_size", type=int, default=None,
                        help="Evaluation batch size (overrides config).")
    parser.add_argument("--visualize_worst", action=argparse.BooleanOptionalAction, default=None, # Default None to use config
                        help="Visualize worst predictions (overrides config).")
    parser.add_argument("--num_worst_samples", type=int, default=None,
                        help="Number of worst samples to show (overrides config).")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Dataloader workers (overrides config).")
    parser.add_argument("--validation_split", type=float, default=None,
                        help="Validation split fraction if needed (overrides config).")
    parser.add_argument("--analyze_errors", action=argparse.BooleanOptionalAction, default=None,
                        help="Perform error analysis during evaluation (overrides config).")
    parser.add_argument("--visualize_confidence_comparison", action=argparse.BooleanOptionalAction, default=None,
                        help="Visualize confidence comparison during evaluation (overrides config).")

    parser.set_defaults(func=handle_evaluate)

def setup_inspect_parser(subparsers, parent_parser):
    """Adds arguments for the 'inspect' subcommand."""
    # Note: Inspect doesn't use config_path, but uses output_dir (save_dir) and seed from parent
    parser = subparsers.add_parser("inspect", help="Inspect dataset samples", parents=[parent_parser])
    parser.add_argument("--dataset_name", type=str, required=True,
                      help="HuggingFace dataset name")
    parser.add_argument("--num_samples", type=int, default=5,
                      help="Number of samples to inspect")
    # Rename save_dir to output_dir to use parent parser's arg
    parser.add_argument("--output_dir", type=str, default="./dataset_inspection",
                      help="Directory to save inspection results")
    parser.add_argument("--enhanced_vis", action=argparse.BooleanOptionalAction, default=True,
                      help="Use enhanced visualization techniques")
    parser.add_argument("--analyze_statistics", action=argparse.BooleanOptionalAction, default=True,
                      help="Generate and save dataset statistics")
    parser.add_argument("--max_attempts", type=int, default=15,
                      help="Maximum number of attempts to find valid samples")

    parser.set_defaults(func=handle_inspect)

def setup_verify_tiling_parser(subparsers, parent_parser):
    """Adds arguments for the 'verify-tiling' subcommand."""
    parser = subparsers.add_parser("verify-tiling", help="Verify training tiling configuration", parents=[parent_parser])
    # Make config_path required for verify-tiling
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to the JSON config file to verify.")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to check.")
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=False,
                        help="Visualize the fetched tiles during verification.")
    parser.set_defaults(func=handle_verify_tiling)


# --- Helper Functions ---

def _load_model_for_eval_predict(model_path: str, device: torch.device, logger: logging.Logger) -> Optional[torch.nn.Module]:
    """Unified loader for both HF and non-HF checkpoints via checkpoint.load_model_for_evaluation."""
    if not verify_checkpoint(model_path):
        logger.error(f"Model not found or invalid at path: {model_path}")
        return None
    try:
        from checkpoint import load_model_for_evaluation
        model, _ = load_model_for_evaluation(model_path=model_path, config=None, device=device, logger=logger)
        model.eval()
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        return None

def _determine_output_dir(args: argparse.Namespace, config: Optional[Config]) -> str:
    """Determines the final output directory based on args and config."""
    # Command-specific defaults if output_dir is not provided via CLI or config
    default_dirs = {
        'train': './outputs',
        'predict': './outputs/predictions',
        'evaluate': './outputs/evaluation',
        'inspect': './dataset_inspection'
    }

    # 1. Use CLI --output_dir if provided
    if args.output_dir:
        return args.output_dir

    # 2. Use config's output_dir if config exists
    if config and config.get("output_dir"):
        # Adjust for predict/evaluate if no specific CLI output_dir was given
        if args.command == 'predict':
            return os.path.join(config["output_dir"], "predictions")
        elif args.command == 'evaluate':
            return os.path.join(config["output_dir"], "evaluation")
        else: # train or other commands using config
            return config["output_dir"]

    # 3. Use command-specific default
    return default_dirs.get(args.command, ".")


# --- Command Handlers ---

def handle_train(args: argparse.Namespace, logger: logging.Logger, config: Config, device: torch.device):
    """Handles the 'train' subcommand."""
    logger.info("Mode: Training")
    # Config is already loaded and merged

    # Ensure output directory exists
    os.makedirs(config["output_dir"], exist_ok=True)

    # Save the final effective config
    config.save(os.path.join(config["output_dir"], "effective_train_config.json"))
    logger.info(f"Effective training config saved to {config['output_dir']}/effective_train_config.json")

    results = run_training_pipeline(
        config=config,
        logger=logger,
        is_notebook=False # Assuming CLI is not a notebook
    )
    logger.info(f"Training completed. Model saved to {results['model_dir']}")
    logger.info(f"Final metrics: {results['metrics']}")

def handle_predict(args: argparse.Namespace, logger: logging.Logger, config: Config, device: torch.device):
    """Handles the 'predict' subcommand."""
    logger.info("Mode: Prediction")
    # Config is loaded

    # Determine model path
    model_path = args.model_path if args.model_path else os.path.join(config.get("output_dir", "."), "final_model") # Use config's output_dir as base if model_path not given

    # Load model using helper
    model = _load_model_for_eval_predict(model_path, device, logger)
    if model is None:
        return # Error logged in helper

    # Output directory is already determined and created in main
    output_dir = args.output_dir

    logger.info(f"Predicting using model: {model_path}")
    logger.info(f"Input images: {args.image_paths}")
    logger.info(f"Output directory: {output_dir}")

    # Pass the loaded model to the pipeline function
    results = run_prediction_pipeline(
        config=config,
        image_paths=args.image_paths,
        model_path=model_path, # Pass model_path for reference, though model obj is used
        output_dir=output_dir,
        batch_size=args.batch_size,
        visualize=args.visualize,
        show_confidence=args.show_confidence,
        show_class_activation_maps=args.show_class_activation_maps,
        logger=logger,
        is_notebook=False
        # Note: run_prediction_pipeline currently re-loads the model,
        # ideally it should accept the loaded model object. Refactoring pipeline.py is needed for that.
    )
    logger.info(f"Prediction completed. Outputs saved to {output_dir}")

def handle_inspect(args: argparse.Namespace, logger: logging.Logger, config: Optional[Config], device: torch.device): # Added device (unused but consistent)
    """Handles the 'inspect' subcommand."""
    logger.info("Mode: Dataset Inspection")
    # Output directory is args.output_dir (renamed from save_dir)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Set log level based on verbosity flag
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger.setLevel(log_level)

    try:
        start_time = time.time()
        logger.info(f"Starting dataset inspection for {args.dataset_name}...")

        # Load dataset
        try:
            # Use seed from args (which defaults to parent parser's default or CLI override)
            logger.info(f"Loading dataset {args.dataset_name} with seed {args.seed}...")
            dataset_dict = load_and_shuffle_dataset(args.dataset_name, seed=args.seed)
            logger.info(f"Dataset loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}", exc_info=True)
            return

        # Examine raw annotations
        logger.info("Examining raw annotations...")
        raw_dir = os.path.join(output_dir, "raw_annotations")
        os.makedirs(raw_dir, exist_ok=True)
        examine_raw_annotations(
            dataset_name=args.dataset_name,
            num_samples=args.num_samples,
            save_dir=raw_dir,
            seed=args.seed,
            enhanced_vis=args.enhanced_vis,
            dataset_dict=dataset_dict,
            logger=logger,
            is_notebook=False
        )
        logger.info(f"Raw annotations examination complete. Results saved to {raw_dir}")

        # Process dataset with image processor
        logger.info("Creating image processor...")
        # Use default processor settings for inspection
        image_processor = SegformerImageProcessor()

        # Create dataset using TCDDataset
        logger.info("Creating dataset for inspection...")
        train_dataset = TCDDataset(dataset_dict, image_processor, split="train")

        # Inspect processed samples
        logger.info("Inspecting processed samples...")
        processed_dir = os.path.join(output_dir, "processed_samples")
        os.makedirs(processed_dir, exist_ok=True)
        inspect_dataset_samples(
            train_dataset,
            num_samples=args.num_samples,
            save_dir=processed_dir,
            max_attempts=args.max_attempts,
            seed=args.seed,
            enhanced_vis=args.enhanced_vis,
            logger=logger,
            is_notebook=False
        )
        logger.info(f"Processed samples inspection complete. Results saved to {processed_dir}")

        # Analyze dataset statistics if requested
        if args.analyze_statistics:
            logger.info("Analyzing dataset statistics...")
            stats_dir = os.path.join(output_dir, "statistics")
            os.makedirs(stats_dir, exist_ok=True)
            stats = examine_dataset_statistics(
                train_dataset,
                num_samples=min(100, len(train_dataset)), # Use up to 100 samples for statistics
                save_dir=stats_dir,
                logger=logger,
                is_notebook=False
            )
            logger.info(f"Dataset statistics analysis complete. Results saved to {stats_dir}")

            # Log summary statistics
            if "summary" in stats:
                summary = stats["summary"]
                logger.info("\nDataset Summary Statistics:")
                for key, value in summary.items():
                    if isinstance(value, float):
                        logger.info(f"  {key}: {value:.2f}")
                    else:
                        logger.info(f"  {key}: {value}")

        # Inspect validation samples if available
        if "validation" in dataset_dict:
            logger.info("Inspecting validation samples...")
            val_dir = os.path.join(output_dir, "validation_samples")
            os.makedirs(val_dir, exist_ok=True)
            val_dataset = TCDDataset(dataset_dict, image_processor, split="validation")
            inspect_dataset_samples(
                val_dataset,
                num_samples=min(3, len(val_dataset)),
                save_dir=val_dir,
                max_attempts=args.max_attempts,
                seed=args.seed,
                enhanced_vis=args.enhanced_vis,
                logger=logger,
                is_notebook=False
            )
            logger.info(f"Validation samples inspection complete. Results saved to {val_dir}")

        elapsed_time = time.time() - start_time
        logger.info(f"Dataset inspection completed in {elapsed_time:.2f} seconds.")

    except Exception as e:
        logger.error(f"Error during dataset inspection: {e}", exc_info=True)

def handle_evaluate(args: argparse.Namespace, logger: logging.Logger, config: Config, device: torch.device):
    """Handles the 'evaluate' subcommand."""
    logger.info("Mode: Evaluation")
    # Config is loaded

    # Override config with CLI args where provided
    # These checks ensure CLI args take precedence if they are not None
    if args.dataset_name: config["dataset_name"] = args.dataset_name
    if args.eval_batch_size: config["eval_batch_size"] = args.eval_batch_size
    if args.visualize_worst is not None: config["visualize_worst"] = args.visualize_worst
    if args.num_worst_samples is not None: config["num_worst_samples"] = args.num_worst_samples
    if args.num_workers is not None: config["num_workers"] = args.num_workers
    if args.validation_split is not None: config["validation_split"] = args.validation_split
    if args.analyze_errors is not None: config["analyze_errors"] = args.analyze_errors
    if args.visualize_confidence_comparison is not None: config["visualize_confidence_comparison"] = args.visualize_confidence_comparison
    # Note: image_size is not directly used by evaluate_model or create_eval_dataloader

    # Output directory is already determined and created in main
    eval_output_dir = args.output_dir

    # Save the effective config used for evaluation
    config.save(os.path.join(eval_output_dir, "effective_eval_config.json"))
    logger.info(f"Effective evaluation config saved to {eval_output_dir}/effective_eval_config.json")

    # Load model using helper
    model = _load_model_for_eval_predict(args.model_path, device, logger)
    if model is None:
        return # Error logged in helper

    # Get id2label from loaded model
    id2label = model.config.id2label

    # Load dataset and create dataloader
    try:
        logger.info(f"Creating evaluation dataloader for dataset {config['dataset_name']}...")
        eval_dataloader, _, _ = create_eval_dataloader(
            dataset_dict_or_name=config["dataset_name"],
            image_processor=None, # Processor is implicitly handled by model
            config=config, # Pass config for dataloader settings like num_workers, batch_size
            eval_batch_size=config["eval_batch_size"],
            num_workers=config.get("num_workers", 4), # Use get with default
            validation_split=config["validation_split"],
            seed=config.get("seed", 42)
        )
        logger.info("Evaluation dataloader created.")
    except Exception as e:
        logger.error(f"Failed to create evaluation dataloader: {e}", exc_info=True)
        return

    # Run evaluation
    logger.info(f"Evaluating model {args.model_path}...")
    metrics = evaluate_model(
        model=model,
        eval_dataloader=eval_dataloader,
        device=device,
        output_dir=eval_output_dir,
        id2label=id2label,
        visualize_worst=config.get("visualize_worst", True),
        num_worst_samples=config.get("num_worst_samples", 5),
        analyze_errors=config.get("analyze_errors", False),
        visualize_confidence_comparison=config.get("visualize_confidence_comparison", False),
        logger=logger,
        is_notebook=False
    )

    # Log and save metrics
    logger.info("Evaluation metrics:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name} = {metric_value:.4f}")

    metrics_path = os.path.join(eval_output_dir, "evaluation_metrics.txt")
    with open(metrics_path, "w") as f:
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name} = {metric_value:.4f}\n")
    logger.info(f"Evaluation metrics saved to {metrics_path}")

def handle_verify_tiling(args: argparse.Namespace, logger: logging.Logger, config: Config, device: torch.device): # config and device loaded in main
    """Handles the 'verify-tiling' subcommand."""
    logger.info("Mode: Verify Training Tiling")
    # Config is already loaded in main based on args.config_path

    # Call the verification function
    verification_passed = verify_training_tiling(
        config=config, # Pass the loaded config
        num_samples=args.num_samples,
        visualize=args.visualize,
        logger=logger,
        is_notebook=False
    )

    if verification_passed:
        logger.info("Tiling verification successful.")
        sys.exit(0) # Exit with success code
    else:
        logger.error("Tiling verification failed.")
        sys.exit(1) # Exit with failure code


# --- Main Execution ---

from inspect_dataset import verify_training_tiling # Import the function - Already imported above, ensure it's there

def main():
    """Main entry point for the script."""
    parent_parser = create_parent_parser()
    parser = argparse.ArgumentParser(description="BARE: Train, Evaluate, Predict, Inspect, Verify Tiling")
    subparsers = parser.add_subparsers(title="Available Commands", dest="command", required=True)

    # Setup parsers for each command, passing the parent
    setup_train_parser(subparsers, parent_parser)
    setup_predict_parser(subparsers, parent_parser)
    setup_evaluate_parser(subparsers, parent_parser)
    setup_inspect_parser(subparsers, parent_parser)
    setup_verify_tiling_parser(subparsers, parent_parser) # Add the new command parser

    # Parse arguments
    args = parser.parse_args()

    # --- Centralized Setup ---
    config = None
    logger = None
    output_dir = "." # Default

    try:
        # 1. Load Configuration (if applicable)
        # For verify-tiling, predict, evaluate, config_path is required by their parsers
        if args.command in ['predict', 'evaluate', 'verify-tiling']:
             # config_path is already checked by argparse 'required=True'
             config = Config.load(args.config_path)
             # Apply common CLI overrides if they exist in args
             if hasattr(args, 'seed') and args.seed is not None: config["seed"] = args.seed
             # output_dir handled below
        elif args.command == 'train':
             # Train loads config and merges args internally
             config = load_config_from_file_and_args(args.config_path, args)
        # 'inspect' doesn't load a main config file by default

        # 2. Determine Output Directory
        output_dir = _determine_output_dir(args, config)
        args.output_dir = output_dir # Store final output_dir back into args namespace for handlers

        # 3. Setup Logging
        os.makedirs(output_dir, exist_ok=True)
        log_level = logging.DEBUG if hasattr(args, 'verbose') and args.verbose else logging.INFO
        # Use a command-specific log file name
        log_file_name = f"{args.command}.log"
        logger = setup_logging(output_dir, log_level=log_level, log_file_name=log_file_name)

        logger.info(f"Executing command: {args.command}")
        logger.info(f"Arguments: {vars(args)}")
        if config:
            logger.debug(f"Loaded/Merged Config: {config.to_dict()}") # Log full config only in debug

        # 4. Set Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # 5. Execute Command Handler
        args.func(args, logger, config, device) # Pass common objects

    except (ConfigurationError, FileNotFoundError, DatasetError) as e:
        # Log errors using the logger if available, otherwise print
        if logger:
            logger.error(f"Pipeline error: {e}", exc_info=True)
        else:
            print(f"ERROR: Pipeline error: {e}")
        sys.exit(1)
    except Exception as e:
        if logger:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        else:
            print(f"ERROR: An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
