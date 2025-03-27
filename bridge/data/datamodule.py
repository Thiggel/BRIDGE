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


class DebugShape(Transform):
    def __init__(self, label=""):
        super().__init__()
        self.label = label  # Optional label to identify where in the pipeline this is

    def encodes(self, x):
        if isinstance(x, TensorImage) or isinstance(x, torch.Tensor):
            print(f"Shape at {self.label}: {x.shape}")
        elif isinstance(x, PILImage):
            print(f"Shape at {self.label} (PIL): {x.size}")
        elif hasattr(x, "shape"):
            print(f"Shape at {self.label} (other): {x.shape}")
        else:
            print(f"Object at {self.label} has no shape attribute: {type(x)}")
        return x


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
        """Create SSL-specific transforms using FastAI"""
        if not augmentation_config:
            return aug_transforms(
                size=(self.image_size, self.image_size),
                min_scale=0.08,  # Equivalent to RandomResizedCrop(scale=(0.08, 1.0))
                flip_vert=False,  # Horizontal flip only
                max_lighting=0.4,  # ColorJitter (brightness, contrast)
                max_warp=0,  # No affine transform
                max_rotate=0,  # No rotation
                do_flip=True,  # Enables RandomHorizontalFlip
                pad_mode=PadMode.Zeros,
                p_lighting=0.75,  # Probability for brightness & contrast
            ) + [
                Contrast(max_lighting=0, p=0.2),
                Normalize.from_stats(*imagenet_stats),
            ]

        transform_list = []

        transform_list.append(DebugShape("before_resize"))

        for aug in augmentation_config:
            if "rrc" in aug:
                config = aug["rrc"]
                transform_list.append(
                    RandomResizedCrop(
                        size=config.get("crop_size", self.image_size),
                        min_scale=config.get("min_scale", 0.08),
                        max_scale=config.get("max_scale", 1.0),
                    )
                )
                transform_list.append(DebugShape("after_rrc"))
            elif "color_jitter" in aug:
                config = aug["color_jitter"]
                transform_list.extend(
                    aug_transforms(
                        max_lighting=config.get("brightness", 0.4),
                        p_lighting=0.75,
                        size=self.image_size,
                    )
                )
            elif "random_gray_scale" in aug:
                config = aug["random_gray_scale"]
                transform_list.append(
                    Contrast(max_lighting=0, p=config.get("prob", 0.1))
                )
            elif "random_horizontal_flip" in aug:
                config = aug["random_horizontal_flip"]
                transform_list.append(FlipItem(p=config.get("prob", 0.5)))

        transform_list.append(Normalize.from_stats(*imagenet_stats))

        transform_list.append(DebugShape("after_normalize"))

        print("-" * 50)
        for t in transform_list:

            print(t)
            print()
        print("-" * 50)

        return transform_list

    def _create_eval_transforms(self):
        """Get evaluation transforms (resize and normalize only)"""
        normalize_config = (
            self.augmentations[self.augmentations.index("normalize")]
            if "augmentations" in self.augmentations
            else None
        )

        return [
            Resize(self.image_size),
            Normalize.from_stats(
                *(normalize_config if normalize_config else imagenet_stats)
            ),
        ]

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
            batch_tfms=self.train_transform if is_train else self.eval_transform,
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
