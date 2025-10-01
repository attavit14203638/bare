# BARE: Boundary-Aware with Resolution Enhancement for Tree Crown Delineation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/Paper-AusDM'25-green.svg)](https://github.com/attavit14203638/bare)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

**Official PyTorch implementation of "BARE: Boundary-Aware with Resolution Enhancement for Tree Crown Delineation"**  
*Accepted at Australasian Data Mining Conference (AusDM) 2025*

> **Authors:** Attavit Wilaiwongsakul, Bin Liang, Wenfeng Jia, Bryan Zheng, Fang Chen  
> **Affiliation:** University of Technology Sydney, Charles Sturt University

---

## Abstract

Tree crown delineation from aerial and satellite imagery is critical for forest inventory, biodiversity assessment, and ecosystem monitoring. However, existing segmentation methods struggle with accurate boundary delineation due to resolution loss in deep neural networks. We propose **BARE** (Boundary-Aware with Resolution Enhancement), a novel strategy that enhances segmentation architectures by maintaining full input resolution during training and inference. BARE provides external supervision at the original resolution, significantly improving boundary accuracy across multiple state-of-the-art architectures (SegFormer, PSPNet, SETR) with minimal computational overhead.

## Key Features

- **🎯 BARE Strategy**: Full-resolution supervision and selective use of class-weighting improves boundary delineation without architectural modifications
- **🏗️ Multi-Architecture Support**: Unified framework for SegFormer, PSPNet, and SETR with consistent improvements
- **⚡ Efficient Training**: Mixed precision, gradient accumulation, and optimized data loading
- **📊 Comprehensive Evaluation**: Boundary IoU, standard metrics, and detailed visualization tools
- **🔧 Production-Ready**: Complete pipeline from training to deployment with checkpointing and model export

## How BARE Works

The BARE strategy enhances existing architectures through a simple yet effective approach:

1. **External Resolution Enhancement**: Instead of modifying the architecture, we add an external upsampling layer after the decoder
2. **Full-Resolution Supervision**: Loss is computed on full-resolution outputs, providing stronger gradient signals for boundary regions
3. **Training-Time Enhancement**: The upsampling is applied during training to enforce full-resolution predictions
4. **Inference Flexibility**: Models can generate outputs at any resolution without retraining

**Benefits:**
- ✅ **Plug-and-play**: Works with any segmentation architecture
- ✅ **Boundary accuracy**: Significant improvements in boundary-aware metrics
- ✅ **Minimal overhead**: <5% increase in training time
- ✅ **No architecture changes**: Preserves model properties and pre-trained weights

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (for GPU training)
- 16GB+ RAM (32GB recommended for larger models)
- GPU with 8GB+ VRAM (for training)

## Installation

```bash
# Clone the repository
git clone https://github.com/attavit14203638/bare.git
cd bare

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

## Quick Start

```bash
# Train SegFormer with BARE on TCD dataset
python main.py train \
    --architecture segformer \
    --dataset_name restor/tcd \
    --model_name nvidia/segformer-b0-finetuned-ade-512-512 \
    --output_dir ./outputs \
    --num_epochs 50 \
    --train_batch_size 8 \
    --mixed_precision

# Evaluate the trained model
python main.py evaluate \
    --config_path ./outputs/effective_train_config.json \
    --model_path ./outputs/best_checkpoint \
    --output_dir ./eval_results
```

## Usage Guide

### Training

**Command-Line Interface:**

```bash
# Basic training
python main.py train --output_dir ./training_output

# Training with specific parameters
python main.py train \
    --dataset_name restor/tcd \
    --model_name nvidia/segformer-b0-finetuned-ade-512-512 \
    --output_dir ./my_trained_model \
    --num_epochs 15 \
    --learning_rate 6e-5 \
    --train_batch_size 4 \
    --mixed_precision

# Training using a config file with parameter overrides
python main.py train \
    --config_path ./configs/base_config.json \
    --learning_rate 7e-5 \
    --output_dir ./tuned_model
```

*Use `python main.py train --help` for all available options.*

**Python API:**

```python
from config import Config
from pipeline import run_training_pipeline

# Create configuration for BARE SegFormer
config = Config({
    "architecture": "segformer",
    "dataset_name": "restor/tcd",
    "model_name": "nvidia/segformer-b0-finetuned-ade-512-512",
    "use_bare_strategy": True,
    "output_dir": "./outputs",
    "num_epochs": 10,
    "learning_rate": 1e-4,
    "train_batch_size": 8
})

# Run training pipeline
results = run_training_pipeline(config=config)
```

### Prediction / Inference

**Command-Line Interface:**

```bash
# Predict on a single image
python main.py predict \
    --config_path ./my_trained_model/effective_train_config.json \
    --model_path ./my_trained_model/final_model \
    --image_paths ./path/to/your/image.png \
    --output_dir ./prediction_results

# Predict on multiple images with visualization
python main.py predict \
    --config_path ./my_trained_model/effective_train_config.json \
    --model_path ./my_trained_model/final_model \
    --image_paths ./images/img1.tif ./images/img2.tif \
    --output_dir ./prediction_results_batch \
    --visualize --show_confidence
```

*Use `python main.py predict --help` for all available options.*

**Python API:**

```python
from config import Config
from pipeline import run_prediction_pipeline

# Load config used during training
config = Config.load("./my_trained_model/effective_train_config.json")

# Run prediction
results = run_prediction_pipeline(
    config=config,
    image_paths=["./path/to/your/image.png", "./another/image.tif"],
    model_path="./my_trained_model/final_model",
    output_dir="./api_predictions",
    visualize=True
)

# Access results
# segmentation_map = results["segmentation_maps"][0]
```

### Evaluation

**Command-Line Interface:**

```bash
# Evaluate a trained model
python main.py evaluate \
    --config_path ./my_trained_model/effective_train_config.json \
    --model_path ./my_trained_model/final_model \
    --output_dir ./evaluation_results

# Evaluate with custom parameters
python main.py evaluate \
    --config_path ./my_trained_model/effective_train_config.json \
    --model_path ./my_trained_model/final_model \
    --output_dir ./evaluation_results_custom \
    --eval_batch_size 32 \
    --no-visualize_worst
```

*Use `python main.py evaluate --help` for all available options.*

## Architecture Configuration

BARE supports multiple state-of-the-art architectures. Here are configuration examples:

### SegFormer with BARE

```python
config = Config({
    "architecture": "segformer",
    "model_name": "nvidia/mit-b5",
    "train_time_upsample": True,
    "class_weights_enabled": True
})
```

### PSPNet with BARE

```python
config = Config({
    "architecture": "pspnet",
    "backbone": "resnet50",
    "dataset_name": "restor/tcd",
    "output_dir": "./outputs_pspnet",
    "num_epochs": 10,
    "learning_rate": 1e-4,
    "train_batch_size": 8
})
```

### SETR with BARE

```python
# Enhanced for 1024×1024 resolution
config = Config({
    "architecture": "setr",
    "setr_embed_dim": 768,
    "setr_patch_size": 16,
    "setr_input_size": 1024  # Native 1024×1024 support
})

# Custom input size
config = Config({
    "architecture": "setr",
    "setr_embed_dim": 768,
    "setr_patch_size": 16,
    "setr_input_size": 512   # Configurable input resolution
})
```

## Advanced Usage

### Dataset Inspection

Visualize and analyze dataset samples before training:

```python
from inspect_dataset import inspect_dataset_samples

# Inspect dataset with visualization
inspect_dataset_samples(
    dataset_name="restor/tcd",
    num_samples=5,
    save_dir="./dataset_inspection",
    seed=42
)
```

### Interactive Training

For interactive experimentation and visualization, use the provided Jupyter notebook:

```bash
jupyter notebook bare_pipeline.ipynb
```

**Note:** The notebooks are provided without outputs to keep repository size manageable. Run the cells to generate outputs.

## For Developers

### Code Structure

The codebase is organized into modular components with clear separation of concerns:

| Module | Description |
|--------|-------------|
| `main.py` | CLI entry point with subcommands |
| `pipeline.py` | Centralized training and evaluation pipeline |
| `config.py` | Configuration management with validation |
| `model.py` | Multi-architecture models with BARE strategy support |
| `dataset.py` | Dataset loading and processing with error handling |
| `metrics.py` | Evaluation metrics for segmentation tasks |
| `checkpoint.py` | Checkpoint management with metadata |
| `weights.py` | Class weight computation for handling imbalance |
| `visualization.py` | Visualization tools for images and results |
| `image_utils.py` | Image processing utilities |
| `tensorboard_utils.py` | TensorBoard integration |
| `exceptions.py` | Custom exception hierarchy |
| `utils.py` | General utilities |
| `cross_validation.py` | Cross-validation support |
| `inspect_dataset.py` | Dataset inspection tools |

### Directory Structure

```
bare/
├── main.py                    # CLI entry point
├── pipeline.py                # Training and evaluation pipeline
├── config.py                  # Configuration management
├── model.py                   # Multi-architecture models with BARE
├── dataset.py                 # Dataset handling
├── metrics.py                 # Evaluation metrics
├── checkpoint.py              # Checkpoint management
├── weights.py                 # Class weight computation
├── visualization.py           # Visualization utilities
├── image_utils.py             # Image processing utilities
├── tensorboard_utils.py       # TensorBoard integration
├── exceptions.py              # Custom exceptions
├── utils.py                   # General utilities
├── cross_validation.py        # Cross-validation support
├── inspect_dataset.py         # Dataset inspection tools
├── requirements.txt           # Python dependencies
├── bare_pipeline.ipynb        # Interactive training notebook
├── visualize_validation_predictions_enhanced.ipynb  # Visualization notebook
├── README.md                  # This file
├── DOCUMENTATION.md           # Detailed technical documentation
├── CONTRIBUTING.md            # Contribution guidelines
└── LICENSE                    # MIT License
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@inproceedings{wilaiwongsakul2025bare,
  title={BARE: Boundary-Aware with Resolution Enhancement for Tree Crown Delineation},
  author={Wilaiwongsakul, Attavit and Liang, Bin and Jia, Wenfeng and Zheng, Bryan and Chen, Fang},
  booktitle={The 23rd Australasian Data Science and Machine Learning Conference (AusDM'25)},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**⭐ If you find this work useful, please consider starring the repository and citing our paper!**
