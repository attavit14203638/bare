#!/usr/bin/env python3
"""
TensorBoard Utilities

This module provides utilities for reading TensorBoard event files and creating visualizations
of metrics. It can be used in three ways:
1. As a library imported in Python code
2. As a module imported in Jupyter notebooks
3. As a standalone script run from the command line

All functionality is consolidated from tensorboard_visualizer.py, tensorboard_metrics.py,
and visualize_tensorboard_logs.py into a single, easy-to-use module.
"""

import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from tensorboard.backend.event_processing import event_accumulator
import logging # Added logging
import argparse # Added argparse

# Import logger setup
from utils import get_logger

# Setup module logger
logger = get_logger()

def read_tensorboard_logs(log_dir):
    """
    Read TensorBoard event logs and extract scalar metrics.
    
    Args:
        log_dir: Path to the TensorBoard log directory
        
    Returns:
        Dictionary mapping metric names to (steps, values) tuples
    """
    # Find all event files in the directory
    event_files = []
    for root, _, files in os.walk(log_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                event_files.append(os.path.join(root, file))
    
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {log_dir}")
    
    # Dictionary to store all metrics
    metrics = defaultdict(lambda: ([], []))
    
    # Process each event file
    for event_file in event_files:
        logger.info(f"Processing event file: {event_file}")
        try:
            ea = event_accumulator.EventAccumulator(event_file,
                                                   size_guidance={event_accumulator.SCALARS: 0})
            ea.Reload()
        except Exception as e:
            logger.warning(f"Could not process event file {event_file}: {e}")
            continue # Skip to the next file
        
        # Extract scalar metrics
        for tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            
            # Get the existing steps and values
            steps, values = metrics[tag]
            
            # Append new data
            for event in events:
                steps.append(event.step)
                values.append(event.value)
            
            # Sort by step if needed
            if steps and len(steps) > 1 and steps[-2] > steps[-1]:
                indices = np.argsort(steps)
                steps = [steps[i] for i in indices]
                values = [values[i] for i in indices]
            
            metrics[tag] = (steps, values)
    
    return metrics


def plot_metrics(metrics, output_dir=None, figsize=(14, 10)):
    """
    Create visualizations of metrics from TensorBoard logs.
    
    Args:
        metrics: Dictionary of metrics from read_tensorboard_logs
        output_dir: Directory to save plots (optional)
        figsize: Figure size for the main plot
    
    Returns:
        Dictionary of matplotlib figures
    """
    # Group metrics by type (train, eval, learning_rate)
    metric_groups = defaultdict(list)
    
    for tag in metrics:
        if '/learning_rate' in tag:
            metric_groups['learning_rate'].append(tag)
        elif '/loss' in tag:
            metric_groups['loss'].append(tag)
        elif '/train' in tag:
            metric_groups['train'].append(tag)
        elif '/eval' in tag or '/validation' in tag:
            metric_groups['eval'].append(tag)
        else:
            # Try to categorize by prefix
            prefix = tag.split('/')[0] if '/' in tag else 'other'
            metric_groups[prefix].append(tag)
    
    figures = {}
    
    # Plot all metrics in a single figure
    n_groups = len(metric_groups)
    n_cols = min(3, n_groups)
    n_rows = (n_groups + n_cols - 1) // n_cols
    
    fig_all = plt.figure(figsize=figsize)
    fig_all.suptitle('Training Metrics Overview', fontsize=16)
    
    for i, (group_name, tags) in enumerate(metric_groups.items()):
        ax = fig_all.add_subplot(n_rows, n_cols, i + 1)
        
        # Plot each metric in this group
        for tag in tags:
            steps, values = metrics[tag]
            if not steps:
                continue
                
            # Clean up label name from tag
            label = tag.split('/')[-1] if '/' in tag else tag
            label = label.replace('_', ' ').title()
            
            ax.plot(steps, values, marker='.', markersize=3, label=label)
        
        ax.set_title(group_name.replace('_', ' ').title())
        ax.set_xlabel('Steps')
        ax.set_ylabel('Value')
        ax.grid(alpha=0.3)
        if len(tags) > 1:
            ax.legend(loc='best')
    
    fig_all.tight_layout(rect=[0, 0, 1, 0.97])
    figures['overview'] = fig_all
    
    # Create individual plots for each metric group
    for group_name, tags in metric_groups.items():
        if not tags:
            continue
            
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        for tag in tags:
            steps, values = metrics[tag]
            if not steps:
                continue
                
            label = tag.split('/')[-1] if '/' in tag else tag
            label = label.replace('_', ' ').title()
            
            ax.plot(steps, values, marker='.', markersize=3, label=label)
        
        title = f"{group_name.replace('_', ' ').title()} Metrics"
        ax.set_title(title)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Value')
        ax.grid(alpha=0.3)
        if len(tags) > 1:
            ax.legend(loc='best')
        
        fig.tight_layout()
        figures[group_name] = fig
    
    # Save figures if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        for name, fig in figures.items():
            filename = f"{name}_metrics.png"
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved {name} metrics plot to {filepath}")
            # Close individual figures after saving
            if name != 'overview': # Don't close the overview figure yet
                 plt.close(fig)

    # Close the overview figure if it exists and saving was requested
    if 'overview' in figures and output_dir:
        plt.close(figures['overview'])

    return figures


def visualize_tensorboard_metrics(log_dir, output_dir=None, show_plots=True):
    """
    Main function to read TensorBoard logs and visualize metrics.
    
    Args:
        log_dir: Path to the TensorBoard log directory
        output_dir: Directory to save plots (optional)
        show_plots: Whether to display plots (True) or just save them (False)
        
    Returns:
        Dictionary of matplotlib figures
    """
    try:
        metrics = read_tensorboard_logs(log_dir)
        logger.info(f"Found {len(metrics)} metrics in TensorBoard logs")
        
        if not metrics:
            logger.warning("No metrics found in TensorBoard logs.")
            return {}
        
        figures = plot_metrics(metrics, output_dir)
        
        if show_plots:
            plt.show()
        else:
            # Close all figures to free memory
            for fig in figures.values():
                plt.close(fig)
        
        return figures
    except Exception as e:
        logger.error(f"Error visualizing TensorBoard metrics: {e}", exc_info=True)
        return {}


def visualize_tensorboard_metrics_in_notebook(
    log_dir="./high_performance_model/tensorboard",
    output_dir="./high_performance_model/tensorboard_metrics",
    show_plots=True
):
    """
    Visualize TensorBoard metrics in a Jupyter notebook.
    This is a convenience function for use in notebook environments.
    
    Args:
        log_dir: Path to the TensorBoard log directory
        output_dir: Directory to save plots
        show_plots: Whether to display plots in the notebook
        
    Returns:
        Dictionary of matplotlib figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Reading TensorBoard logs from: {log_dir}")
    logger.info(f"Saving visualizations to: {output_dir}")
    
    # Visualize metrics from TensorBoard logs
    figures = visualize_tensorboard_metrics(
        log_dir=log_dir,
        output_dir=output_dir,
        show_plots=show_plots
    )
    
    # Print summary
    if figures:
        logger.info(f"\nCreated {len(figures)} metric visualizations:")
        for name in figures.keys():
            logger.info(f"- {name}")
    else:
        logger.warning("No metrics were found or visualized.")
    
    return figures


def main():
    """Command-line entry point for visualizing TensorBoard metrics"""
    parser = argparse.ArgumentParser(description="Visualize metrics from TensorBoard logs.")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./outputs/tensorboard", # Default to standard output dir
        help="Path to the TensorBoard log directory."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/tensorboard_metrics", # Default output dir
        help="Directory to save the generated plots."
    )
    args = parser.parse_args()

    # Use determined paths
    tensorboard_log_dir = args.log_dir
    output_dir = args.output_dir

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Reading TensorBoard logs from: {tensorboard_log_dir}")
    logger.info(f"Saving visualizations to: {output_dir}")

    # Visualize metrics from TensorBoard logs
    figures = visualize_tensorboard_metrics(
        log_dir=tensorboard_log_dir,
        output_dir=output_dir,
        show_plots=False  # Don't show plots interactively in script mode
    )
    
    # Print summary
    if figures:
        logger.info(f"\nCreated {len(figures)} metric visualizations:")
        for name, fig in figures.items():
            filename = f"{name}_metrics.png"
            filepath = os.path.join(output_dir, filename)
            logger.info(f"- {filepath}")
    else:
        logger.warning("No metrics were found or visualized.")


if __name__ == "__main__":
    # Basic logging setup if run as script
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
    logger.info("\nTo use these visualizations in your notebook, add a cell with:")
    logger.info("import tensorboard_utils")
    logger.info("figures = tensorboard_utils.visualize_tensorboard_metrics_in_notebook()")
