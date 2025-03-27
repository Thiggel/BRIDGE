import os
from typing import Optional, Dict, Any, List
from pathlib import Path
from PIL import Image

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from fastai.vision.all import *
from datasets import load_dataset, Dataset, concatenate_datasets


class HuggingFaceDataModule:
    """Data module for HuggingFace datasets with FastAI integration"""

    def __init__(
        self,
        path: str,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        train_split: str = "train",
        val_split: Optional[str] = "validation",
        test_split: Optional[str] = "test",
        batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
        image_size: int = 224,
        transform: Callable = None,
        keep_in_memory: bool = False,
        val_pct: float = 0.1,  # Percentage of train data to use for validation if no val split exists
    ):
        self.path = path
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.image_size = image_size
        self.keep_in_memory = keep_in_memory
        self.val_pct = val_pct
        self.transform = transform

        self.train_dataset = None

    def _create_transforms(self):
        """Get evaluation transforms (resize and normalize only)"""
        transforms = [
            Resize((self.image_size, self.image_size)),
            Normalize.from_stats(*imagenet_stats),
        ]

        if self.transform is not None:
            transforms.append(self.transform)

    def setup(self):
        """Setup datasets, handling missing splits by creating them"""
        print(f"Loading training dataset from {self.path}")
        try:
            self.train_dataset = load_dataset(
                self.path,
                split=self.train_split,
                keep_in_memory=self.keep_in_memory,
            )
        except Exception as e:
            raise ValueError(f"Error loading training dataset: {e}")

        self.train_dls = self._create_dataloaders(self.train_dataset, is_train=True)

    def _create_dataloaders(self, dataset, is_train=True):
        """Convert HuggingFace dataset to FastAI DataLoaders"""

        class_names = dataset.features["label"].names

        dblock = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_x=lambda row: row["image"],
            get_y=lambda row: class_names[row["label"]],
            splitter=RandomSplitter(valid_pct=self.val_pct),
            item_tfms=self._create_transforms(),
        )

        return dblock.dataloaders(
            dataset,
            bs=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self):
        """Get training DataLoader"""
        return self.train_dls

    def update_train_dataset(self, new_dataset_path):
        """Update training dataset with a new one"""
        print(f"Updating training dataset to {new_dataset_path}")
        self.path = new_dataset_path
        try:
            self.train_dataset = load_dataset(
                self.path,
                split=self.train_split,
                keep_in_memory=self.keep_in_memory,
            )
        except Exception as e:
            # Try loading as a local dataset saved with dataset.save_to_disk()
            print(f"Trying to load as local dataset: {e}")
            self.train_dataset = Dataset.load_from_disk(self.path)

        self.train_dls = self._create_dataloaders(self.train_dataset, is_train=True)
        return self.train_dls
