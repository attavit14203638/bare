# BARE: Technical Documentation

This comprehensive documentation covers the architecture, implementation details, and development history of BARE (Boundary-Aware with Resolution Enhancement) for enhanced boundary delineation in tree crown segmentation tasks.

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
   - [1.1 BARE Strategy Architecture](#11-bare-strategy-architecture)
   - [1.2 Tree Crown Delineation Dataset](#12-tree-crown-delineation-dataset)
2. [Implementation Details](#2-implementation-details)
   - [2.1 Data Processing](#21-data-processing)
   - [2.2 Model Architecture](#22-model-architecture)
   - [2.3 Training Pipeline](#23-training-pipeline)
   - [2.4 Optimization Techniques](#24-optimization-techniques)
   - [2.5 Evaluation Metrics](#25-evaluation-metrics)
3. [Code Structure](#3-code-structure)
   - [3.1 Core Modules](#31-core-modules)
   - [3.2 Configuration Management](#32-configuration-management)
   - [3.3 Error Handling](#33-error-handling)
   - [3.4 Visualization](#34-visualization)
4. [Development History](#4-development-history)
   - [4.1 Refactoring Summary](#41-refactoring-summary)
   - [4.2 Streamlining Efforts](#42-streamlining-efforts)
   - [4.3 Research Background](#43-research-background)
5. [Usage Patterns](#5-usage-patterns)
   - [5.1 CLI Interface](#51-cli-interface)
   - [5.2 Python API](#52-python-api)
   - [5.3 Jupyter Notebook](#53-jupyter-notebook)
6. [Research Applications](#6-research-applications)
   - [6.1 Tree Crown Delineation](#61-tree-crown-delineation)
   - [6.2 Performance Benchmarks](#62-performance-benchmarks)

## 1. Architecture Overview

### 1.1 BARE Strategy Architecture

Our BARE (Boundary-Aware with Resolution Enhancement) strategy enhances multiple segmentation architectures for tree crown delineation tasks. The key innovations include:

1. **Full Resolution Supervision**:
   - All supported architectures produce segmentation maps at the full resolution of input images
   - External upsampling of decoder logits before loss calculation and prediction
   - Significantly improves boundary delineation accuracy without architectural modifications
   - Maintains inference efficiency while improving training supervision

2. **Multi-Architecture Support**:
   - **SegFormer**: Hierarchical Transformer encoder with lightweight MLP decoder
   - **PSPNet**: Pyramid Scene Parsing Network with ResNet backbone
   - **SETR**: Pure Vision Transformer with custom segmentation decoder

3. **Enhanced SETR Implementation**:
   - **Native High-Resolution Processing**: Supports 1024×1024 inputs without downsampling bottlenecks
   - **Flexible ViT Backbone**: Uses timm's Vision Transformer with configurable input dimensions
   - **Position Embedding Interpolation**: Automatically adapts pre-trained 224×224 weights to larger inputs
   - **Memory Efficient**: Maintains computational efficiency through gradient checkpointing

4. **Architecture-Agnostic Benefits**:
   - **Boundary Enhancement**: External full-resolution loss supervision improves boundary accuracy
   - **Resolution Consistency**: All models produce outputs at original input resolution
   - **Fair Comparison**: Identical resolution enhancement applied across different architectures
   - **Computational Efficiency**: Minimal overhead while maintaining model-specific advantages

### 1.2 Tree Crown Delineation Dataset

The Tree Crown Delineation (TCD) dataset contains satellite imagery with pixel-level annotations for tree crown segmentation:

- **Images**: High-resolution satellite/aerial imagery of forest areas
- **Annotations**: Pixel-level segmentation masks (under 'annotation' field)
- **Classes**: Binary (tree crown vs. background)
- **Structure**: Training, validation, and test splits

## 2. Implementation Details

### 2.1 Data Processing

The data processing pipeline includes:

1. **Dataset Loading and Consistent Shuffling**:
   - Centralized `load_and_shuffle_dataset` function
   - Fixed seed for deterministic shuffling
   - Pre-shuffled dataset passed to all pipeline components

2. **Image Preprocessing**:
   - Resizing to fixed dimensions (default: 512×512)
   - Normalization for standardized pixel values
   - Automatic conversion from grayscale to RGB

3. **Segmentation Mask Processing**:
   - Support for 'annotation' field (TCD-specific)
   - Binary format conversion (0: background, 1: tree_crown)
   - Handling of multi-channel annotations
   - Invalid sample detection and skipping

4. **DataLoader Creation**:
   - Efficient batch processing with PyTorch DataLoaders
   - Automatic creation of validation split if needed
   - Consistent error handling

### 2.2 Model Architecture

The BARE strategy enhances multiple architectures (SegFormer, PSPNet, SETR) with the following key modifications:

1. **Resolution Handling**:
   - Modified the decoder to produce logits at the full input resolution
   - Implemented efficient upsampling strategies to minimize computational overhead
   - Added configuration options to control the output resolution behavior

2. **Loss Calculation**:
   - Adapted loss functions to work with full-resolution logits
   - Implemented weighted loss strategies to address class imbalance in tree crown segmentation
   - Added support for boundary-aware loss components to enhance edge preservation

3. **Model Configuration**:
   - Extended the model configuration system to accommodate BARE-specific parameters
   - Created a unified interface supporting multiple architectures (SegFormer, PSPNet, SETR)
   - Maintained backward compatibility with pretrained models

Implementation is in `model.py` with architecture-specific classes enhanced with the BARE strategy.

### 2.3 Training Pipeline

The training pipeline is centralized and includes:

1. **Unified Flow**:
   - Consistent dataset handling throughout
   - Integrated checkpointing and evaluation
   - Standardized configuration

2. **Training Loop**:
   - Forward and backward passes with gradient accumulation
   - Mixed precision support
   - Periodic evaluation and checkpoint saving
   - TensorBoard integration for metrics

3. **Validation**:
   - Regular model evaluation
   - Worst-case sample visualization
   - Comprehensive metrics reporting

### 2.4 Optimization Techniques

Several optimization techniques improve training and performance:

1. **Mixed Precision Training**:
   - Combined 16-bit and 32-bit precision
   - Device-specific implementation (CPU vs. CUDA)
   - Memory usage reduction and faster training

2. **Gradient Accumulation**:
   - Accumulation over multiple batches
   - Support for effective larger batch sizes
   - Training stability improvements

3. **Learning Rate Scheduling**:
   - Linear scheduling with warmup
   - Cosine decay options
   - Checkpoint-based resumption

4. **Class Weight Handling**:
   - Inverse frequency-based weight computation
   - Integration with loss function
   - Support for different weighting strategies

5. **Gradient Checkpointing**:
   - Enabled via the `"gradient_checkpointing": true` setting in the configuration.
   - Reduces memory usage significantly during training by trading computation for memory. Instead of storing intermediate activations for the entire model to compute gradients, it recomputes them during the backward pass.
   - **Trade-off**: Increases training time due to the recomputation overhead. Ideal for training large models on GPUs with limited memory.
   - Implemented by calling `model.gradient_checkpointing_enable()` if available on the Hugging Face model.

6. **DataLoader Performance**:
   - Configurable options in `config.py` to optimize data loading speed:
     - `"dataloader_pin_memory": true`: If using CUDA, pins memory for faster CPU-to-GPU transfers. Can increase CPU memory usage slightly.
     - `"dataloader_persistent_workers": true`: Keeps worker processes alive between epochs, reducing overhead for dataset initialization and startup time, especially for complex datasets or augmentations. Requires `num_workers > 0`.
     - `"dataloader_prefetch_factor": N`: Number of batches loaded in advance by each worker (default: 2). Higher values can improve GPU utilization by reducing I/O wait times but increase memory usage. Requires `num_workers > 0`.
   - These settings are passed to the PyTorch `DataLoader` in `dataset.py`.

### 2.5 Evaluation Metrics

Comprehensive evaluation metrics include:

1. **Primary Metrics**:
   - Mean IoU (Intersection over Union)
   - Dice coefficient
   - Pixel accuracy

2. **Additional Metrics**:
   - Precision and recall
   - F1 score
   - Per-class evaluation

3. **Visualization Metrics**:
   - Worst prediction identification
   - Comparative visualization
   - Confusion matrix and distribution

## 3. Code Structure

### 3.1 Core Modules

The codebase is organized into modular components:

| Module | Description |
|--------|-------------|
| `pipeline.py` | Centralized training and evaluation pipeline |
| `dataset.py` | Dataset loading and processing with error handling |
| `model.py` | SegFormer model with true-resolution output |
| `main.py` | CLI entry point |

### 3.2 Configuration Management

Configuration is managed through:

1. **Config Class** (`config.py`):
   - Comprehensive configuration with validation
   - Parameter grouping (dataset, model, training, etc.)
   - Loading/saving from/to JSON files
   - Default values and type hints

2. **Integration**:
   - CLI argument parsing in `main.py`
   - Configuration conversion from arguments
   - Notebook parameter cell for interactive configuration

### 3.3 Error Handling

Robust error handling is implemented through:

1. **Custom Exception Hierarchy** (`exceptions.py`):
   - `TCDSegformerError` base class
   - Specialized exceptions for different error types
   - Contextual error information

2. **Error Recovery**:
   - Fallback mechanisms for invalid samples
   - Graceful error handling in critical sections
   - Logging with detailed context

3. **Validation**:
   - Configuration parameter validation
   - Dataset sample validation
   - Input/output validation for model operations

### 3.4 Visualization

Comprehensive visualization tools include:

1. **Dataset Inspection**:
   - Raw annotation visualization
   - Processed sample inspection
   - Class distribution visualization

2. **Training Monitoring**:
   - Metric plotting with smoothing
   - Learning rate visualization
   - Loss breakdown by component

3. **Prediction Visualization**:
   - Comparison between prediction and ground truth
   - Worst sample identification
   - Confidence visualization

## 4. Development History

### 4.1 Refactoring Summary

The codebase underwent comprehensive refactoring to improve:

1. **Centralized Configuration**:
   - Creation of `Config` class
   - Removal of hardcoded values
   - Standardized parameter handling

2. **Modular Design**:
   - Clear separation of responsibilities
   - Reduction of interdependencies
   - Enhanced maintainability

3. **Error Handling**:
   - Custom exception hierarchy
   - Consistent recovery mechanisms
   - Specialized error types

4. **Code Organization**:
   - Consolidation of related functionality
   - Standardized interfaces
   - Improved documentation

### 4.2 Streamlining Efforts

Subsequent streamlining efforts focused on:

1. **Centralized Pipeline**:
   - Creation of unified pipeline module
   - Consistent workflow for training operations
   - Standardized parameter handling

2. **Removal of Legacy Components**:
   - Elimination of bridge modules
   - Removal of compatibility layers
   - Direct use of refactored modules

3. **Standardization**:
   - Consistent use of Config class
   - Unified approach to dataset handling
   - Standardized checkpoint management

4. **Modern Notebook Integration**:
   - Creation of refactored notebook
   - Streamlined dataset inspection and training
   - Reduced code duplication

### 4.3 Research Background

The BARE strategy and associated pipeline were developed as part of research addressing challenges in high-resolution semantic segmentation for remote sensing applications. The key motivations were:

1. **Boundary Preservation Challenge**: Conventional semantic segmentation models often struggle with preserving fine-grained details at object boundaries, particularly in complex natural scenes like forests.

2. **Resolution Trade-offs**: Most segmentation architectures reduce spatial resolution internally to manage computational complexity, which can lead to detail loss at object boundaries.

3. **Domain-Specific Requirements**: Tree crown delineation in remote sensing imagery presents unique challenges including varying tree sizes, dense canopy overlap, and complex textures that require specialized approaches.

Our BARE strategy was designed to address these challenges while maintaining computational efficiency, providing consistent improvements across multiple architectures (SegFormer, PSPNet, SETR) for tasks requiring high boundary accuracy.

## 5. Usage Patterns

### 5.1 CLI Interface

The command-line interface provides a simple entry point:

```bash
python main.py --dataset_name restor/tcd --model_name nvidia/segformer-b0-finetuned-ade-512-512 --output_dir ./outputs
```

Additional parameters can customize:
- Training parameters (epochs, batch size, learning rate)
- Model configuration (architecture, weights)
- Optimization settings (mixed precision, gradient accumulation)

### 5.2 Python API

The Python API allows programmatic use:

```python
from config import Config
from pipeline import run_training_pipeline

# Create configuration
config = Config({
    "dataset_name": "restor/tcd",
    "model_name": "nvidia/segformer-b0-finetuned-ade-512-512",
    "output_dir": "./outputs",
    "num_epochs": 10
})

# Run training pipeline
results = run_training_pipeline(config=config)
```

This enables integration with larger ML pipelines or custom workflows.

### 5.3 Jupyter Notebook

The `bare_pipeline.ipynb` notebook provides an interactive interface for:

- Dataset exploration and visualization
- Model training with real-time monitoring
- Result inspection and analysis
- Hyperparameter experimentation

The notebook imports functions from Python modules to create a complete pipeline with interactive components.

## 6. Research Applications

### 6.1 Tree Crown Delineation

The primary application of this codebase is for accurate delineation of individual tree crowns in remote sensing imagery, which serves several important ecological and environmental monitoring purposes:

- **Forest Inventory**: Calculating tree density, crown size distributions, and forest structure metrics
- **Biodiversity Assessment**: Analyzing tree species diversity based on crown characteristics
- **Carbon Stock Estimation**: Improved estimates of above-ground biomass through accurate crown measurements
- **Ecosystem Monitoring**: Tracking changes in forest structure and composition over time

### 6.2 Performance Benchmarks

Our experimental evaluation shows that the BARE strategy consistently improves performance across multiple architectures in tree crown delineation tasks:

- **Improved Boundary IoU**: Significant improvements in boundary-aware metrics across all supported architectures
- **Enhanced Detail Preservation**: Superior performance in identifying small trees and delineating complex crown boundaries
- **Balanced Efficiency**: Achieves these improvements with minimal computational overhead
- **Architecture-Agnostic Benefits**: Consistent gains when applied to SegFormer, PSPNet, and SETR

Full benchmark results are available in our AusDM'25 paper.
