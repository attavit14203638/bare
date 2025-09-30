#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to upload the trained TCD-SegFormer model to the Hugging Face Hub.
"""

import os
import argparse
import tempfile
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from huggingface_hub import HfApi
from utils import load_config

def upload_to_hub(
    model,
    repo_name,
    commit_message="Upload TCD-SegFormer model",
    private=False,
    config_path=None
):
    """
    Upload the trained model to the Hugging Face Hub.
    
    Args:
        model: The trained model to upload
        repo_name: Name of the repository on Hugging Face Hub (format: username/repo-name)
        commit_message: Commit message for the upload
        private: Whether to make the repository private
        config_path: Path to the model configuration file
    """
    print(f"Uploading model to {repo_name}...")
    
    # Load image processor
    image_processor = SegformerImageProcessor()
    
    # Create model card content
    model_card_content = """---
language: en
license: mit
datasets:
- restor/tcd
tags:
- segformer
- semantic-segmentation
- tree-crown-delineation
- computer-vision
- remote-sensing
---

# TCD-SegFormer Model

This model is a fine-tuned version of [nvidia/segformer-b0-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512) on the [Tree Crown Delineation (TCD) dataset](https://huggingface.co/datasets/restor/tcd).

## Model description

The model is based on the SegFormer architecture, which is a simple, efficient yet powerful semantic segmentation framework. It consists of a hierarchical Transformer encoder and a lightweight all-MLP decoder.

## Intended uses & limitations

This model is intended for semantic segmentation of tree crowns in satellite or aerial imagery. It can be used for forest monitoring, biodiversity assessment, and other environmental applications.

## Training procedure

The model was trained using the following parameters:

"""
    
    if config_path and os.path.exists(config_path):
        config = load_config(config_path)
        
        # Add training parameters to model card
        model_card_content += "```\n"
        for key, value in config.items():
            if key not in ["id2label", "label2id"]:
                model_card_content += f"{key}: {value}\n"
        model_card_content += "```\n"
    
    model_card_content += """
## Evaluation results

The model was evaluated on the validation set of the TCD dataset. The evaluation metrics include pixel accuracy, mean IoU, mean Dice coefficient, precision, recall, and F1 score.

## Usage

```python
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt

# Load model and processor
model_name = "{repo_name}"
processor = SegformerImageProcessor.from_pretrained(model_name)
model = SegformerForSemanticSegmentation.from_pretrained(model_name)

# Load image
image = Image.open("path/to/image.jpg").convert("RGB")

# Preprocess image
inputs = processor(images=image, return_tensors="pt")

# Make prediction
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get predicted segmentation map
predicted_segmentation = logits.argmax(dim=1).squeeze().cpu().numpy()

# Visualize prediction
plt.imshow(predicted_segmentation)
plt.show()
```

## Citation

If you use this model, please cite the original SegFormer paper:
```
@article{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  journal={arXiv preprint arXiv:2105.15203},
  year={2021}
}
```

And the TCD dataset:
```
@dataset{restor_tcd,
  author = {Restor},
  title = {Tree Crown Delineation Dataset},
  year = {2023},
  publisher = {Hugging Face},
  journal = {Hugging Face Dataset Repository},
  howpublished = {\\url{https://huggingface.co/datasets/restor/tcd}}
}
```
""".format(repo_name=repo_name)
    
    # Create a temporary directory to save the model card
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save model card
        with open(os.path.join(tmp_dir, "README.md"), "w") as f:
            f.write(model_card_content)
        
        # Push model to Hub
        model.push_to_hub(
            repo_id=repo_name,
            use_temp_dir=True,
            commit_message=commit_message,
            private=private,
        )
        
        # Push image processor to Hub
        image_processor.push_to_hub(
            repo_id=repo_name,
            use_temp_dir=True,
            commit_message=commit_message,
        )
        
        # Push model card to Hub
        api = HfApi()
        api.upload_file(
            path_or_fileobj=os.path.join(tmp_dir, "README.md"),
            path_in_repo="README.md",
            repo_id=repo_name,
            commit_message=commit_message,
        )
    
    print(f"Model successfully uploaded to https://huggingface.co/{repo_name}")

def parse_args():
    parser = argparse.ArgumentParser(description="Upload TCD-SegFormer model to Hugging Face Hub")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model checkpoint")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to the model configuration file")
    parser.add_argument("--repo_name", type=str, required=True,
                        help="Name of the repository on Hugging Face Hub (format: username/repo-name)")
    parser.add_argument("--commit_message", type=str, default="Upload TCD-SegFormer model",
                        help="Commit message for the upload")
    parser.add_argument("--private", action="store_true",
                        help="Whether to make the repository private")
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    print(f"Loading model from {args.model_path}...")
    
    # Load model
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_path)
    
    # Upload model to Hub
    upload_to_hub(
        model=model,
        repo_name=args.repo_name,
        commit_message=args.commit_message,
        private=args.private,
        config_path=args.config_path
    )

if __name__ == "__main__":
    main()
