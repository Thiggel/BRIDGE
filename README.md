# BRIDGE: Balancing Representations by Identifying and Generating Underrepresented Data

This project implements the BRIDGE method for improving self-supervised learning on imbalanced datasets by dynamically identifying and generating underrepresented samples in the latent space.

## Overview

The BRIDGE method works by:

1. Training a self-supervised model for a portion of the total epochs
2. Analyzing the latent space to identify underrepresented samples
3. Generating new samples for these underrepresented regions using diffusion models
4. Continuing training with the augmented dataset
5. Repeating this process for multiple cycles

## Features

- Support for various self-supervised learning methods from Lightly (DINO, SimCLR, MoCo, etc.)
- Integration with HuggingFace datasets for easy data loading
- Diffusion-based data augmentation for generating new samples
- Visualization of latent space using UMAP with cluster and class information
- Support for ablation studies (random samples, using removed data)
- Integration with Weights & Biases for experiment tracking

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bridge.git
cd bridge

# Create conda environment
conda create -n bridge python=3.9
conda activate bridge

# Install dependencies
pip install -r requirements.txt
```

## Usage

The project uses Hydra for configuration management. You can run experiments using the following command:

```bash
python main.py
```

To customize configurations:

```bash
# Use different model
python main.py model=simclr

# Use different dataset
python main.py dataset=imagenet100_balanced

# Run ablation experiment
python main.py experiment=ablation/random_samples

# Run with custom parameters
python main.py bridge.num_cycles=3 optimizer.lr=0.001
```

## Project Structure

```
bridge/
├── config/                # Hydra configurations
│   ├── config.yaml        # Base configuration
│   ├── model/             # Model configurations
│   ├── dataset/           # Dataset configurations
│   └── experiment/        # Experiment configurations
├── bridge/                # Main package
│   ├── data/              # Data loading components
│   ├── models/            # Model definitions
│   ├── detectors/         # OOD detection
│   ├── augmentors/        # Data augmentation
│   ├── trainer.py         # Main trainer
│   └── utils/             # Utility functions
├── main.py               # Entry point
└── README.md            # Documentation
```

## Supported Models

The following self-supervised learning models are supported:

- DINO
- SimCLR
- BarlowTwins
- BYOL
- MoCo

## Supported Datasets

The project is designed to work with any HuggingFace dataset, but has been specifically tested with:

- ImageNet-100-LT (long-tailed version)
- ImageNet-100-LT-balanced

## Citation

If you use this code in your research, please cite:

```
@article{bridge2025,
  title={BRIDGE: Balancing Representations by Identifying and Generating Underrepresented Data},
  author={Your Name},
  journal={arXiv preprint arXiv:2025.00000},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
