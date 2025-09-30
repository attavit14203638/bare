#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cross-validation utilities for TCD-SegFormer model.

This module provides functions for performing k-fold cross-validation
to evaluate model performance with robust metrics.
"""

import os
import torch
import numpy as np
import json
import time
import copy
from typing import Dict, Any, List, Optional, Union
import logging
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt

from config import Config
from pipeline import run_training_pipeline, evaluate_model
from checkpoint import load_model_for_evaluation
from utils import get_logger, set_seed, log_or_print
from dataset import load_and_shuffle_dataset, create_dataloaders

logger = get_logger()

def run_cross_validation(
    config: Config,
    num_folds: int = 5,
    logger_obj: Optional[logging.Logger] = None,
    is_notebook: bool = False
) -> Dict[str, Any]:
    """
    Run k-fold cross-validation training and evaluation.
    
    Args:
        config: Configuration object
        num_folds: Number of folds for cross-validation
        logger_obj: Optional logger
        is_notebook: Whether running in notebook environment
        
    Returns:
        Dictionary with aggregated results and per-fold metrics
    """
    # Use provided logger or default module logger
    logger_to_use = logger_obj or logger
    
    # Initialize results storage
    cv_results = {
        "metrics_per_fold": [],
        "best_model_dirs": [],
        "fold_splits": [],
        "config": config.to_dict()
    }
    
    # Create a base output directory for all CV results
    base_output_dir = os.path.join(config["output_dir"], "cross_validation")
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Save the overall config
    config_path = os.path.join(base_output_dir, "cv_config.json")
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    log_or_print(f"Starting {num_folds}-fold cross-validation", logger_to_use, logging.INFO, is_notebook)
    log_or_print(f"Results will be saved to: {base_output_dir}", logger_to_use, logging.INFO, is_notebook)
    
    # Load the full dataset once
    dataset_dict = load_and_shuffle_dataset(
        dataset_name=config["dataset_name"], 
        seed=config["seed"]
    )
    
    # Create indices for train dataset to split
    indices = np.arange(len(dataset_dict["train"]))
    
    # Create a cross-validation splitter
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=config["seed"])
    
    # Setup for each fold
    fold_metrics_list = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        fold_num = fold + 1
        log_or_print(f"\n{'='*30} Fold {fold_num}/{num_folds} {'='*30}\n", logger_to_use, logging.INFO, is_notebook)
        fold_start_time = time.time()
        
        # Create a deep copy of config to modify for this fold
        fold_config = copy.deepcopy(config)
        
        # Create fold-specific output directory
        fold_dir = os.path.join(base_output_dir, f"fold_{fold_num}")
        os.makedirs(fold_dir, exist_ok=True)
        fold_config["output_dir"] = fold_dir
        
        # Save fold indices for reproducibility
        fold_indices = {
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist()
        }
        cv_results["fold_splits"].append(fold_indices)
        
        indices_path = os.path.join(fold_dir, "fold_indices.json")
        with open(indices_path, 'w') as f:
            json.dump(fold_indices, f, indent=2)
        
        # Create fold-specific datasets from the original dataset
        from datasets import Dataset
        # Create train dataset for this fold
        fold_train_dataset = Dataset.from_dict(
            {k: [dataset_dict["train"][i][k] for i in train_idx] for k in dataset_dict["train"].features}
        )
        # Create validation dataset for this fold
        fold_val_dataset = Dataset.from_dict(
            {k: [dataset_dict["train"][i][k] for i in val_idx] for k in dataset_dict["train"].features}
        )
        
        # Create fold-specific dataset dict
        fold_dataset_dict = {
            "train": fold_train_dataset,
            "validation": fold_val_dataset
        }
        
        # If there's a test set in the original dataset, keep it
        if "test" in dataset_dict:
            fold_dataset_dict["test"] = dataset_dict["test"]
        
        # Log fold dataset sizes
        log_or_print(f"Fold {fold_num} dataset sizes:", logger_to_use, logging.INFO, is_notebook)
        log_or_print(f"  Train: {len(fold_train_dataset)} samples", logger_to_use, logging.INFO, is_notebook)
        log_or_print(f"  Validation: {len(fold_val_dataset)} samples", logger_to_use, logging.INFO, is_notebook)
        
        # Run the training pipeline for this fold
        log_or_print(f"Training model for fold {fold_num}...", logger_to_use, logging.INFO, is_notebook)
        
        try:
            # Set explicit validation_split to None since we've already split the data
            fold_config["_explicit_fold_validation"] = True  # Add marker to indicate pre-split data
            
            # Run training pipeline for this fold
            fold_results = run_training_pipeline(
                config=fold_config, 
                logger=logger_to_use,
                is_notebook=is_notebook,
                fold_dataset_dict=fold_dataset_dict  # Pass the pre-split dataset
            )
            
            # Store best model path
            best_model_dir = fold_results.get("best_model_dir") or fold_results["model_dir"]
            cv_results["best_model_dirs"].append(best_model_dir)
            
            # Store metrics
            fold_metrics = fold_results["metrics"]
            cv_results["metrics_per_fold"].append(fold_metrics)
            fold_metrics_list.append({
                "fold": fold_num,
                **{k: v for k, v in fold_metrics.items() if isinstance(v, (int, float))}
            })
            
            # Log metrics and timing
            fold_duration = (time.time() - fold_start_time) / 60  # Convert to minutes
            log_or_print(f"Fold {fold_num} completed in {fold_duration:.2f} minutes", logger_to_use, logging.INFO, is_notebook)
            log_or_print(f"Fold {fold_num} metrics: {fold_metrics}", logger_to_use, logging.INFO, is_notebook)
        
        except Exception as e:
            log_or_print(f"Error in fold {fold_num}: {e}", logger_to_use, logging.ERROR, is_notebook)
            log_or_print(f"Skipping to next fold", logger_to_use, logging.WARNING, is_notebook)
    
    # Generate aggregate metrics across all folds
    log_or_print("\n" + "="*30 + " Cross-Validation Results " + "="*30, logger_to_use, logging.INFO, is_notebook)
    
    # Calculate aggregate statistics
    aggregate_metrics = {}
    if fold_metrics_list:
        # Convert the list of dicts to a DataFrame for easier analysis
        metrics_df = pd.DataFrame(fold_metrics_list)
        metrics_df.set_index("fold", inplace=True)
        
        # Calculate statistics for each numeric metric
        metric_cols = [col for col in metrics_df.columns if col != "fold"]
        
        for metric in metric_cols:
            values = metrics_df[metric].values
            aggregate_metrics[f"{metric}_mean"] = float(np.mean(values))
            aggregate_metrics[f"{metric}_std"] = float(np.std(values))
            aggregate_metrics[f"{metric}_min"] = float(np.min(values))
            aggregate_metrics[f"{metric}_max"] = float(np.max(values))
            aggregate_metrics[f"{metric}_median"] = float(np.median(values))
        
        # Create visualizations of metrics across folds
        try:
            # Bar chart for key metrics
            key_metrics = ["f1_score", "precision", "recall", "accuracy", "IoU", "boundary_IoU"]
            available_metrics = [m for m in metric_cols if any(km in m for km in key_metrics)]
            
            if available_metrics:
                plt.figure(figsize=(12, 6))
                metrics_df[available_metrics].plot(kind="bar", figsize=(14, 8))
                plt.title("Metrics Across Folds")
                plt.ylabel("Score")
                plt.xlabel("Fold")
                plt.grid(axis="y", alpha=0.3)
                plt.xticks(rotation=0)
                plt.tight_layout()
                
                # Save the figure
                metrics_chart_path = os.path.join(base_output_dir, "metrics_by_fold.png")
                plt.savefig(metrics_chart_path)
                plt.close()
                
                log_or_print(f"Metrics visualization saved to: {metrics_chart_path}", logger_to_use, logging.INFO, is_notebook)
        except Exception as e:
            log_or_print(f"Error creating metrics visualization: {e}", logger_to_use, logging.WARNING, is_notebook)
    
    # Store aggregate metrics in results
    cv_results["aggregate_metrics"] = aggregate_metrics
    
    # Save final results to json
    results_path = os.path.join(base_output_dir, "cv_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            "aggregate_metrics": aggregate_metrics,
            "metrics_per_fold": cv_results["metrics_per_fold"],
            "best_model_dirs": cv_results["best_model_dirs"],
            "num_folds": num_folds,
            "config": config.to_dict()
        }, f, indent=2)
    
    # Log aggregate results
    log_or_print("\nAggregate Metrics:", logger_to_use, logging.INFO, is_notebook)
    for key, value in aggregate_metrics.items():
        log_or_print(f"  {key}: {value:.4f}", logger_to_use, logging.INFO, is_notebook)
    
    log_or_print(f"\nCross-validation complete! Results saved to: {results_path}", logger_to_use, logging.INFO, is_notebook)
    
    return cv_results
