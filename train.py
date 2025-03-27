#!/usr/bin/env python3
import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import random
from pathlib import Path
from dotenv import load_dotenv

from bridge.trainer import BRIDGETrainer


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for the application"""
    load_dotenv()

    # Print current config
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # Set random seed
    set_seed(cfg.seed)

    # Create trainer
    trainer = BRIDGETrainer(cfg)

    # Run training
    trainer.train()

    return 0


if __name__ == "__main__":
    main()
