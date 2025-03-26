import os
import time
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import wandb
from omegaconf import OmegaConf, DictConfig
from fastai.vision.all import *

from bridge.data.datamodule import HuggingFaceDataModule
from bridge.detectors.ood_detector import OODDetector
from bridge.augmentors.augmentor import DatasetAugmentor
from bridge.models.ssl_models import get_ssl_model


class BRIDGETrainer:
    """Main trainer class for BRIDGE method"""

    def __init__(self, cfg: DictConfig):
        """Initialize trainer with configuration"""
        self.cfg = cfg
        self.name = cfg.name
        self.output_dir = os.path.join(cfg.base_dir, cfg.name)
        os.makedirs(self.output_dir, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Create data module
        self.data_module = self.create_data_module()

        # Create model
        self.model = self.create_model()

        # Create callbacks
        self.callbacks = self.create_callbacks()

    def setup_logging(self):
        """Setup logging with wandb if enabled"""
        if self.cfg.use_wandb:
            wandb.init(
                project="bridge",
                name=self.name,
                config=OmegaConf.to_container(self.cfg, resolve=True),
            )

    def create_data_module(self):
        """Create data module for training"""
        data_module = HuggingFaceDataModule(
            train_path=self.cfg.data.train_path,
            val_path=self.cfg.data.val_path,
            test_path=self.cfg.data.test_path,
            train_split=self.cfg.data.split.train,
            val_split=self.cfg.data.split.val,
            test_split=self.cfg.data.split.test,
            batch_size=self.cfg.optimizer.batch_size,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
            image_size=self.cfg.data.image_size,
            keep_in_memory=False,  # Ensure we don't load entire dataset in memory
        )
        data_module.setup()
        return data_module

    def create_model(self):
        """Create SSL model"""
        return get_ssl_model(
            self.cfg.model.name,
            backbone=self.cfg.model.backbone,
            projection_dim=self.cfg.model.projection_dim,
            pretrained=self.cfg.model.pretrained,
            **{
                k: v
                for k, v in self.cfg.model.items()
                if k not in ["name", "backbone", "projection_dim", "pretrained"]
            },
        )

    def create_callbacks(self):
        """Create training callbacks"""
        callbacks = []

        # Add model checkpoint callback
        callbacks.append(
            SaveModelCallback(
                monitor="valid_loss",
                fname=f"{self.name}_best",
                path=Path(self.output_dir),
                with_opt=True,
            )
        )

        # Add early stopping callback
        callbacks.append(
            EarlyStoppingCallback(
                monitor="valid_loss",
                patience=10,
                min_delta=0.001,
            )
        )

        return callbacks

    def train_cycle(
        self, cycle_idx: int, num_epochs: int, ckpt_path: Optional[str] = None
    ):
        """Train for one cycle"""
        print(f"Starting training cycle {cycle_idx + 1}/{self.cfg.bridge.num_cycles}")

        # Create trainer for this cycle
        learner = Learner(
            self.data_module.train_dls,
            self.model,
            loss_func=None,  # SSL models usually have their own loss
            metrics=[],
            path=Path(self.output_dir),
            model_dir=f"cycle_{cycle_idx}",
            cbs=self.callbacks,
        )

        # Train for specified number of epochs
        learner.fit(num_epochs, lr=self.cfg.optimizer.lr)

        # Save model
        learner.save(f"cycle_{cycle_idx}_final")

        return learner

    def run_ood_detection(self, cycle_idx: int):
        """Run OOD detection on current model and dataset"""
        print(f"Running OOD detection for cycle {cycle_idx + 1}")

        # Create OOD detector
        ood_detector = OODDetector(
            feature_extractor=self.model,
            num_clusters=self.cfg.bridge.num_clusters,
            num_ood_samples_per_cluster=self.cfg.bridge.num_ood_samples_per_cluster,
            batch_size=self.cfg.optimizer.batch_size,
            num_workers=self.cfg.data.num_workers,
            experiment_name=self.name,
            cycle_idx=cycle_idx,
            output_dir=self.output_dir,
        )

        # Get OOD samples
        if self.cfg.bridge.use_ood_augmentation:
            # Get actual OOD samples
            ood_indices, ood_features, ood_labels = ood_detector.get_ood_datapoints(
                self.data_module.train_dataset
            )
        else:
            # For ablation: get random samples instead
            total_samples = (
                self.cfg.bridge.num_clusters
                * self.cfg.bridge.num_ood_samples_per_cluster
            )
            ood_indices, ood_features, ood_labels = ood_detector.get_random_samples(
                self.data_module.train_dataset, total_samples
            )

        return ood_indices, ood_features, ood_labels

    def augment_dataset(self, cycle_idx: int, ood_indices, ood_features, ood_labels):
        """Augment dataset with generated samples or removed data"""
        print(f"Augmenting dataset for cycle {cycle_idx + 1}")

        # Create dataset augmentor
        augmentor = DatasetAugmentor(
            num_generations_per_sample=self.cfg.bridge.num_generations_per_ood_sample,
            diffusion_model=self.cfg.bridge.diffusion_model,
            batch_size=self.cfg.optimizer.batch_size,
            num_workers=self.cfg.data.num_workers,
            experiment_name=self.name,
            cycle_idx=cycle_idx,
            original_dataset_path=self.cfg.data.train_path,
            output_dir=self.output_dir,
        )

        # Augment dataset
        if (
            hasattr(self.cfg.bridge, "use_removed_data")
            and self.cfg.bridge.use_removed_data
        ):
            # Ablation: use removed data instead of generating new samples
            new_dataset_path = augmentor.add_removed_data(
                self.cfg.data.removed_data_path
            )
        else:
            # Normal case: generate new samples with diffusion model
            new_dataset_path = augmentor.augment_dataset(
                ood_indices, ood_features, ood_labels
            )

        # Update data module with new dataset
        self.cfg.data.train_path = new_dataset_path
        self.data_module = self.create_data_module()

        return new_dataset_path

    def train(self):
        """Run the full BRIDGE training process"""
        # Calculate epochs per cycle
        num_epochs_per_cycle = self.cfg.bridge.num_epochs_per_cycle

        # Train for each cycle
        for cycle_idx in range(self.cfg.bridge.num_cycles):
            # Train for this cycle
            learner = self.train_cycle(
                cycle_idx=cycle_idx,
                num_epochs=num_epochs_per_cycle,
            )

            # Skip OOD detection and augmentation for the last cycle
            if cycle_idx < self.cfg.bridge.num_cycles - 1:
                # Run OOD detection
                ood_indices, ood_features, ood_labels = self.run_ood_detection(
                    cycle_idx
                )

                # Augment dataset
                new_dataset_path = self.augment_dataset(
                    cycle_idx, ood_indices, ood_features, ood_labels
                )

                # Log updated dataset path
                print(
                    f"Updated dataset path for cycle {cycle_idx + 1}: {new_dataset_path}"
                )
                if self.cfg.use_wandb:
                    wandb.log({f"dataset_path_cycle_{cycle_idx + 1}": new_dataset_path})

        # Final evaluation
        self.evaluate()

        # Close wandb
        if self.cfg.use_wandb:
            wandb.finish()

    def evaluate(self):
        """Evaluate the model on test set"""
        # TODO: Implement evaluation on test set
        pass

    def finetune(
        self,
        dataset_name: str,
        num_epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 64,
        mixup: float = 0.0,
        freeze_backbone: bool = True,
        unfreeze_layers: int = 0,
        pretrained_path: Optional[str] = None,
    ):
        """Finetune model on a downstream task

        Args:
            dataset_name: Name of dataset for finetuning ('cifar10', 'cifar100', 'caltech101', etc.)
            num_epochs: Number of epochs for finetuning
            lr: Learning rate for finetuning
            batch_size: Batch size for finetuning
            mixup: Mixup alpha parameter (0 to disable)
            freeze_backbone: Whether to freeze the backbone
            unfreeze_layers: Number of layers to unfreeze from the end (if freeze_backbone is True)
            pretrained_path: Path to pretrained model (if None, use the current model)

        Returns:
            Trained learner object
        """
        print(f"Finetuning on {dataset_name} dataset")

        # Load pretrained model if specified
        if pretrained_path:
            print(f"Loading pretrained model from {pretrained_path}")
            self.model.load_state_dict(torch.load(pretrained_path))

        # Create finetuning model by extracting backbone and adding classifier head
        backbone = (
            self.model.backbone if hasattr(self.model, "backbone") else self.model
        )

        # Create data loaders for the specified dataset
        dls = self._create_finetune_dataloaders(dataset_name, batch_size)

        # Create a classifier model using the backbone
        learn = self._create_finetune_learner(
            backbone, dls, freeze_backbone, unfreeze_layers
        )

        # Train with mixup if specified
        if mixup > 0:
            learn.add_cb(MixUp(mixup))

        # Find learning rate if not provided
        if lr is None:
            lr = learn.lr_find().valley
            print(f"Found optimal learning rate: {lr}")

        # Fine-tune the model
        learn.fine_tune(num_epochs, lr)

        # Save the fine-tuned model
        model_path = os.path.join(self.output_dir, f"finetune_{dataset_name}")
        learn.save(model_path)

        # Evaluate on test set
        test_metrics = learn.validate()
        print(f"Test metrics: {test_metrics}")

        # Log metrics to wandb
        if self.cfg.use_wandb:
            wandb.log(
                {
                    f"finetune_{dataset_name}_error_rate": test_metrics[0],
                    f"finetune_{dataset_name}_loss": test_metrics[1],
                }
            )

        return learn

    def _create_finetune_dataloaders(self, dataset_name: str, batch_size: int = 64):
        """Create DataLoaders for finetuning on standard datasets"""
        # Supported datasets
        dataset_config = {
            # ImageNet and variants
            "imagenet100": {"cls": ImageDataLoaders.from_folder, "size": 224},
            # CIFAR
            "cifar10": {
                "cls": vision_learner,
                "size": 32,
                "func": lambda path: untar_data(URLs.CIFAR),
            },
            "cifar100": {
                "cls": vision_learner,
                "size": 32,
                "func": lambda path: untar_data(URLs.CIFAR_100),
            },
            # Standard image classification datasets
            "caltech101": {
                "cls": vision_learner,
                "size": 224,
                "func": lambda path: untar_data(URLs.CALTECH_101),
            },
            "pets": {
                "cls": vision_learner,
                "size": 224,
                "func": lambda path: untar_data(URLs.PETS),
            },
            "flowers": {
                "cls": vision_learner,
                "size": 224,
                "func": lambda path: untar_data(URLs.FLOWERS),
            },
            "food": {
                "cls": vision_learner,
                "size": 224,
                "func": lambda path: untar_data(URLs.FOOD),
            },
            # More specialized datasets
            "cars": {
                "cls": vision_learner,
                "size": 224,
                "func": lambda path: untar_data(URLs.CARS),
            },
            "aircraft": {
                "cls": vision_learner,
                "size": 224,
                "func": None,
            },  # Need custom loading
        }

        if dataset_name not in dataset_config:
            raise ValueError(
                f"Unsupported dataset: {dataset_name}. Supported datasets: {list(dataset_config.keys())}"
            )

        config = dataset_config[dataset_name]

        # Handle custom datasets
        if dataset_name == "aircraft":
            # FGVC Aircraft needs custom loading
            path = Path(self.output_dir) / "fgvc-aircraft-2013b"
            if not path.exists():
                raise ValueError(
                    "FGVC Aircraft dataset not found. Please download manually."
                )
            dls = ImageDataLoaders.from_folder(
                path,
                train="data/images/train",
                valid="data/images/test",
                item_tfms=Resize(config["size"]),
                batch_tfms=aug_transforms(size=config["size"], min_scale=0.75),
                bs=batch_size,
            )
        elif "func" in config and config["func"] is not None:
            # FastAI built-in datasets
            path = config["func"](self.output_dir)
            dls = ImageDataLoaders.from_folder(
                path,
                valid_pct=0.2,
                item_tfms=Resize(config["size"]),
                batch_tfms=aug_transforms(size=config["size"], min_scale=0.75),
                bs=batch_size,
            )
        else:
            # Custom path datasets
            if dataset_name.startswith("imagenet"):
                # Imagenet-style datasets
                path = Path(
                    self.cfg.data.finetune_path
                    if hasattr(self.cfg.data, "finetune_path")
                    else self.output_dir
                )
                dls = ImageDataLoaders.from_folder(
                    path,
                    train="train",
                    valid="val",
                    item_tfms=Resize(config["size"]),
                    batch_tfms=aug_transforms(size=config["size"], min_scale=0.75),
                    bs=batch_size,
                )

        return dls

    def _create_finetune_learner(
        self, backbone, dls, freeze_backbone: bool = True, unfreeze_layers: int = 0
    ):
        """Create a learner for finetuning"""
        # Create a learner with our backbone
        learn = vision_learner(
            dls,
            "resnet50",  # This will be replaced by our backbone
            metrics=[error_rate, accuracy],
        )

        # Replace the model with our backbone
        learn.model.features = backbone

        # Freeze the backbone if needed
        if freeze_backbone:
            learn.freeze()

            # Unfreeze the last few layers if specified
            if unfreeze_layers > 0:
                learn.unfreeze_to(-unfreeze_layers)

        return learn

    def fine_tune_all(self, datasets: List[str], **kwargs):
        """Fine-tune on multiple datasets"""
        results = {}

        for dataset in datasets:
            print(f"Fine-tuning on {dataset}")
            learn = self.finetune(dataset, **kwargs)

            # Store results
            metrics = learn.validate()
            results[dataset] = {
                "error_rate": metrics[0].item(),
                "accuracy": metrics[1].item(),
            }

        # Log summary table to wandb
        if self.cfg.use_wandb:
            table = wandb.Table(columns=["Dataset", "Error Rate", "Accuracy"])
            for dataset, metrics in results.items():
                table.add_data(dataset, metrics["error_rate"], metrics["accuracy"])

            wandb.log({"finetune_results": table})

        return results
