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

- **🎯 BARE Strategy**: Full-resolution supervision improves boundary delineation without architectural modifications
- **🏗️ Multi-Architecture Support**: Unified framework for SegFormer, PSPNet, and SETR with consistent improvements
- **⚡ Efficient Training**: Mixed precision, gradient accumulation, and optimized data loading
- **📊 Comprehensive Evaluation**: Boundary IoU, standard metrics, and detailed visualization tools
- **🔧 Production-Ready**: Complete pipeline from training to deployment with checkpointing and model export

## Highlights

| Feature | Description |
|---------|-------------|
| **BARE Strategy** | External full-resolution loss supervision for improved boundary accuracy |
| **Multi-Architecture** | SegFormer, PSPNet, SETR with unified BARE enhancement |
| **Efficient Processing** | Mixed precision training, gradient checkpointing, optimized dataloaders |
| **Advanced Metrics** | Standard IoU, Boundary IoU, Dice coefficient, per-class analysis |
| **Production Tools** | CLI, Python API, Jupyter notebooks, model export to HuggingFace Hub |
| **Visualization Suite** | Dataset inspection, prediction overlays, confidence maps, error analysis |

## Getting Started

### Quick Start

```bash
# Clone and install
git clone https://github.com/attavit14203638/bare.git
cd bare
pip install -r requirements.txt

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

### Installation

```bash
# Clone the repository
git clone https://github.com/attavit14203638/bare.git
cd bare

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (for GPU training)
- 16GB+ RAM (32GB recommended for larger models)
- GPU with 8GB+ VRAM (for training)

### Command-Line Interface (CLI)

The primary way to interact with the project is through the `main.py` script using subcommands:

**1. Training a Model:**

```bash
# Basic training with default config values (if applicable)
python main.py train --output_dir ./training_output

# Training with specific parameters (overriding defaults or config file)
python main.py train \
    --dataset_name restor/tcd \
    --model_name nvidia/segformer-b0-finetuned-ade-512-512 \
    --output_dir ./my_trained_model \
    --num_epochs 15 \
    --learning_rate 6e-5 \
    --train_batch_size 4 \
    --mixed_precision

# Training using a base config file and overriding some parameters
python main.py train --config_path ./configs/base_config.json --learning_rate 7e-5 --output_dir ./tuned_model
```
*Use `python main.py train --help` for all options.*

**2. Making Predictions:**

```bash
# Predict on a single image using a trained model and its config
python main.py predict \
    --config_path ./my_trained_model/effective_train_config.json \
    --model_path ./my_trained_model/final_model \
    --image_paths ./path/to/your/image.png \
    --output_dir ./prediction_results

# Predict on multiple images with visualization and confidence maps
python main.py predict \
    --config_path ./my_trained_model/effective_train_config.json \
    --model_path ./my_trained_model/final_model \
    --image_paths ./images/img1.tif ./images/img2.tif \
    --output_dir ./prediction_results_batch \
    --visualize --show_confidence
```
*Use `python main.py predict --help` for all options.*

**3. Evaluating a Model:**

```bash
# Evaluate a trained model using its config
python main.py evaluate \
    --config_path ./my_trained_model/effective_train_config.json \
    --model_path ./my_trained_model/final_model \
    --output_dir ./evaluation_results

# Evaluate with specific evaluation parameters
python main.py evaluate \
    --config_path ./my_trained_model/effective_train_config.json \
    --model_path ./my_trained_model/final_model \
    --output_dir ./evaluation_results_custom \
    --eval_batch_size 32 \
    --no-visualize_worst
```
*Use `python main.py evaluate --help` for all options.*

### Training a Model (Python API)

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

# Or use other architectures with BARE strategy
config_pspnet = Config({
    "architecture": "pspnet",
    "backbone": "resnet50",
    "dataset_name": "restor/tcd", 
    "output_dir": "./outputs_pspnet",
    "num_epochs": 10,
    "learning_rate": 1e-4,
    "train_batch_size": 8
})

# Run training pipeline
results = run_training_pipeline(config=config)
```

### Interactive Training

Use the `tcd_segformer_pipeline.ipynb` notebook for interactive training and experimentation. Note: For GitHub, the notebooks are provided without outputs to keep repository size manageable. Run the cells to generate outputs.

## Code Structure

The codebase is organized into modular components with clear separation of concerns:

| Module | Description |
|--------|-------------|
| `pipeline.py` | Centralized training and evaluation pipeline |
| `config.py` | Configuration management with validation |
| `dataset.py` | Dataset loading and processing with error handling |
| `model.py` | Multi-architecture models with BARE strategy support |
| `metrics.py` | Evaluation metrics for segmentation tasks |
| `checkpoint.py` | Checkpoint management with metadata |
| `weights.py` | Class weight computation for handling imbalance |
| `visualization.py` | Visualization tools for images and results |
| `image_utils.py` | Image processing utilities |
| `exceptions.py` | Custom exception hierarchy |
| `main.py` | CLI entry point |

### Enhanced SETR Implementation

**NEW**: SETR now supports native 1024×1024 processing without downsampling bottlenecks:

- **Native High-Resolution Processing**: Processes 1024×1024 inputs directly without 1024→224→1024 pipeline
- **Flexible ViT Backbone**: Uses `vit_base_patch16_224.augreg_in21k_ft_in1k` with input size override
- **Position Embedding Interpolation**: Automatically adapts pre-trained weights from 224×224 to 1024×1024
- **Memory Efficient**: 92M parameters with gradient checkpointing support
- **BARE Strategy Compatible**: Full-resolution loss supervision for improved boundary accuracy

### Architecture Configuration Examples

```python
# SegFormer with BARE
config = Config({
    "architecture": "segformer",
    "model_name": "nvidia/mit-b5",
    "train_time_upsample": True,
    "class_weights_enabled": True
})

# PSPNet with BARE  
config = Config({
    "architecture": "pspnet",
    "backbone": "resnet50"
})

# SETR with BARE (Enhanced for 1024×1024)
config = Config({
    "architecture": "setr",
    "setr_embed_dim": 768,
    "setr_patch_size": 16,
    "setr_input_size": 1024  # NEW: Native 1024×1024 support
})

# SETR with custom input size
config = Config({
    "architecture": "setr",
    "setr_embed_dim": 768,
    "setr_patch_size": 16,
    "setr_input_size": 512   # Configurable input resolution
})
```

### How BARE Works

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

## Example Usage

### Dataset Inspection

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

### Prediction (Python API)

```python
# Using the prediction pipeline function
from config import Config
from pipeline import run_prediction_pipeline

# Load config used during training
config = Config.load("./my_trained_model/effective_train_config.json")

# Run prediction
results = run_prediction_pipeline(
    config=config,
    image_paths=["./path/to/your/image.png", "./another/image.tif"],
    model_path="./my_trained_model/final_model", # Optional, defaults based on config
    output_dir="./api_predictions", # Optional, defaults based on config
    visualize=True
)

# Access results (e.g., segmentation maps)
# segmentation_map = results["segmentation_maps"][0]
```

### Uploading to Hugging Face Hub

```bash
python upload_to_hub.py --model_dir ./outputs/final_model --repo_id attavit14203638/bare
```

## Directory Structure

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
├── upload_to_hub.py           # HuggingFace Hub upload
├── requirements.txt           # Python dependencies
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
  booktitle={Australasian Data Mining Conference (AusDM)},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**⭐ If you find this work useful, please consider starring the repository and citing our paper!**
