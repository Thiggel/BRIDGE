#!/usr/bin/env python3
import os
import sys
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from bridge.trainer import BRIDGETrainer
from main import set_seed


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    """Fine-tune a pre-trained SSL model on downstream tasks"""
    # Print current config
    print(OmegaConf.to_yaml(cfg))

    # Set random seed
    set_seed(cfg.seed)

    # Add finetune-specific configs
    if "finetune" not in cfg:
        cfg.finetune = {
            "datasets": ["cifar10", "cifar100", "caltech101"],
            "num_epochs": 5,
            "lr": 0.001,
            "batch_size": 64,
            "mixup": 0.1,
            "freeze_backbone": True,
            "unfreeze_layers": 0,
            "pretrained_path": None,
        }

    # Create trainer
    trainer = BRIDGETrainer(cfg)

    # Load pretrained model if specified
    if cfg.finetune.pretrained_path is not None:
        print(f"Loading pretrained model from {cfg.finetune.pretrained_path}")
        checkpoint = torch.load(cfg.finetune.pretrained_path)
        trainer.model.load_state_dict(checkpoint)

    # Run fine-tuning on all specified datasets
    results = trainer.fine_tune_all(
        datasets=cfg.finetune.datasets,
        num_epochs=cfg.finetune.num_epochs,
        lr=cfg.finetune.lr,
        batch_size=cfg.finetune.batch_size,
        mixup=cfg.finetune.mixup,
        freeze_backbone=cfg.finetune.freeze_backbone,
        unfreeze_layers=cfg.finetune.unfreeze_layers,
    )

    # Print results
    print("\nFine-tuning Results:")
    for dataset, metrics in results.items():
        print(
            f"{dataset}: Error Rate = {metrics['error_rate']:.4f}, Accuracy = {metrics['accuracy']:.4f}"
        )

    return 0


if __name__ == "__main__":
    main()
