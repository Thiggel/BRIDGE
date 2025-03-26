import os
from typing import Optional, Dict, Any, List
from pathlib import Path

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

        self.train_transform = self._create_ssl_transforms(self.augmentations)
        self.eval_transform = self._create_eval_transforms()

    def _create_ssl_transforms(self, augmentation_config):
        """Create SSL-specific transforms from configuration"""
        if not augmentation_config:
            # Default SimCLR-style transforms if none provided
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(self.image_size, scale=(0.08, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                    ),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        # Build transforms based on configuration
        transform_list = []

        for aug in augmentation_config:
            # Handle each augmentation type
            if "rrc" in aug:
                # Random Resized Crop
                config = aug["rrc"]
                transform_list.append(
                    transforms.RandomResizedCrop(
                        config.get("crop_size", self.image_size),
                        scale=(
                            config.get("min_scale", 0.08),
                            config.get("max_scale", 1.0),
                        ),
                    )
                )
            elif "color_jitter" in aug:
                # Color Jitter
                config = aug["color_jitter"]
                transform_list.append(
                    transforms.ColorJitter(
                        brightness=config.get("brightness", 0.4),
                        contrast=config.get("contrast", 0.4),
                        saturation=config.get("saturation", 0.4),
                        hue=config.get("hue", 0.1),
                    )
                )
            elif "random_gray_scale" in aug:
                # Random Grayscale
                config = aug["random_gray_scale"]
                transform_list.append(
                    transforms.RandomGrayscale(p=config.get("prob", 0.2))
                )
            elif "random_horizontal_flip" in aug:
                # Random Horizontal Flip
                config = aug["random_horizontal_flip"]
                transform_list.append(
                    transforms.RandomHorizontalFlip(p=config.get("prob", 0.5))
                )
            elif "normalize" in aug:
                # Normalization (should be after ToTensor)
                config = aug["normalize"]
                # Ensure ToTensor is added before normalization
                if not any(isinstance(t, transforms.ToTensor) for t in transform_list):
                    transform_list.append(transforms.ToTensor())
                transform_list.append(
                    transforms.Normalize(
                        mean=config.get("mean", [0.485, 0.456, 0.406]),
                        std=config.get("std", [0.229, 0.224, 0.225]),
                    )
                )

        # Ensure ToTensor is included if not already
        if not any(isinstance(t, transforms.ToTensor) for t in transform_list):
            # Add ToTensor before any normalization
            norm_idx = next(
                (
                    i
                    for i, t in enumerate(transform_list)
                    if isinstance(t, transforms.Normalize)
                ),
                len(transform_list),
            )
            transform_list.insert(norm_idx, transforms.ToTensor())

        return transforms.Compose(transform_list)

    def _create_eval_transforms(self):
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
            return np.array(row["image"])

        def get_y(row):
            return row["label"]

        # Create FastAI DataBlock
        dblock = DataBlock(
            get_x=get_x,
            get_y=get_y,
            splitter=RandomSplitter(valid_pct=self.val_pct),
            item_tfms=None,
            batch_tfms=self.train_transform if is_train else self.eval_transform,
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
