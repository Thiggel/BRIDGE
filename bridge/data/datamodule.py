import os
from typing import Optional, Dict, Any, List
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
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
        augmentations: Optional[Dict[str, Any]] = None,
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
        self.augmentations = augmentations
        self.keep_in_memory = keep_in_memory
        self.val_pct = val_pct

        self.train_dataset = None

        # Setup transforms
        self.train_tfms = self._get_train_transforms()
        self.eval_tfms = self._get_eval_transforms()

    def _get_train_transforms(self):
        """Get training transforms with data augmentation"""
        return aug_transforms(
            size=self.image_size,
            max_warp=0.1,
            max_rotate=10.0,
            max_zoom=1.1,
            max_lighting=0.2,
            do_flip=True,
        )

    def _get_eval_transforms(self):
        """Get evaluation transforms (resize and normalize only)"""
        return [
            Resize(self.image_size),
            ToTensor(),
            Normalize.from_stats(*imagenet_stats),
        ]

    def setup(self):
        """Setup datasets, handling missing splits by creating them"""
        # Load training dataset
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

        # Define item getter function
        def get_x(row):
            return row["image"]

        def get_y(row):
            return row["label"]

        # Create FastAI DataBlock
        dblock = DataBlock(
            get_x=get_x,
            get_y=get_y,
            splitter=RandomSplitter(valid_pct=self.val_pct),
            item_tfms=[ToTensor()],
            batch_tfms=self.train_tfms if is_train else self.eval_tfms,
        )

        # Create DataLoaders
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
