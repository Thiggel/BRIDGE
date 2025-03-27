import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import wandb
from omegaconf import OmegaConf, DictConfig
from fastai.vision.all import *
from fastai.distributed import *
from fastai.callback.wandb import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.transforms.moco_transform import MoCoV2Transform
from lightly.transforms.dino_transform import DINOTransform
from lightly.transforms.byol_transform import (
    BYOLTransform,
    BYOLView1Transform,
    BYOLView2Transform,
)

from bridge.data.datamodule import HuggingFaceDataModule
from bridge.detectors.ood_detector import OODDetector
from bridge.evaluators.knn_evaluator import KNNEvaluator
from bridge.augmentors.augmentor import DatasetAugmentor
from bridge.models.ssl_models import get_ssl_model
from bridge.utils.visualization import (
    plot_embeddings,
    plot_class_distribution,
    plot_cluster_distribution,
)


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
            path=self.cfg.dataset.path,
            train_split=self.cfg.dataset.split.train,
            val_split=self.cfg.dataset.split.val,
            test_split=self.cfg.dataset.split.test,
            batch_size=self.cfg.model.optimizer.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=self.cfg.dataset.pin_memory,
            image_size=self.cfg.dataset.image_size,
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
            input_size=self.cfg.dataset.image_size,
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

        # Add WandbCallback for logging to wandb
        if self.cfg.use_wandb:
            callbacks.append(WandbCallback())

        if self.cfg.model.get("scheduler", None) is not None:
            warmup_pct = warmup_epochs / total_epochs

            pct = [warmup_pct]
            schedulers = [SchedLin(0, lr_max)]

            if self.cfg.model.scheduler.get("policy", None) == "cosine":
                pct.append(1 - warmup_pct)
                schedulers.append(SchedCos(lr_max, 0))

            # Define the learning rate schedule
            scheds = {"lr": combine_scheds(pct, schedulers)}

            # Add learning rate scheduler
            callbacks.append(ParamScheduler(scheds))

        return callbacks

    def train_cycle(
        self, cycle_idx: int, num_epochs: int, ckpt_path: Optional[str] = None
    ):
        """Train for one cycle"""
        print(
            f"Starting training cycle {cycle_idx + 1}/{self.cfg.experiment.bridge.num_cycles}"
        )

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

        learner.path = Path(os.environ.get("BASE_CACHE_DIR", "/tmp"))

        # Train for specified number of epochs
        with learner.distrib_ctx():
            print(num_epochs)
            learner.fit(num_epochs, lr=self.cfg.model.optimizer.lr)

        # Save model
        learner.save(f"cycle_{cycle_idx}_final")

        return learner

    def run_ood_detection(self, cycle_idx: int):
        """Run OOD detection on current model and dataset"""
        print(f"Running OOD detection for cycle {cycle_idx + 1}")

        # Create OOD detector
        ood_detector = OODDetector(
            feature_extractor=self.model,
            num_clusters=self.cfg.experiment.bridge.num_clusters,
            num_ood_samples_per_cluster=self.cfg.experiment.bridge.num_ood_samples_per_cluster,
            batch_size=self.cfg.model.optimizer.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            experiment_name=self.name,
            cycle_idx=cycle_idx,
            output_dir=self.output_dir,
        )

        # Get OOD samples
        if self.cfg.experiment.bridge.use_ood_augmentation:
            # Get actual OOD samples along with cluster assignments and dataset
            ood_indices, ood_features, ood_labels, cluster_assignments, dataset = (
                ood_detector.get_ood_datapoints(self.data_module.train_dataset)
            )
        else:
            # For ablation: get random samples instead
            # Note: cluster_assignments will be None in this case
            total_samples = (
                self.cfg.experiment.bridge.num_clusters
                * self.cfg.experiment.bridge.num_ood_samples_per_cluster
            )
            ood_indices, ood_features, ood_labels = ood_detector.get_random_samples(
                self.data_module.train_dataset, total_samples
            )
            cluster_assignments = None
            dataset = self.data_module.train_dataset

        return ood_indices, ood_features, ood_labels, cluster_assignments, dataset

    def augment_dataset(
        self,
        cycle_idx: int,
        ood_indices,
        ood_features,
        ood_labels,
        cluster_assignments=None,
        original_dataset=None,
    ):
        """Augment dataset with generated samples or removed data"""
        print(f"Augmenting dataset for cycle {cycle_idx + 1}")

        # Create dataset augmentor
        augmentor = DatasetAugmentor(
            num_generations_per_sample=self.cfg.experiment.bridge.num_generations_per_ood_sample,
            diffusion_model=self.cfg.experiment.bridge.diffusion_model,
            batch_size=self.cfg.model.optimizer.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            experiment_name=self.name,
            cycle_idx=cycle_idx,
            original_dataset_path=self.cfg.dataset.path,
            output_dir=self.output_dir,
        )

        # Augment dataset
        if (
            hasattr(self.cfg.experiment.bridge, "use_removed_data")
            and self.cfg.experiment.bridge.use_removed_data
        ):
            # Ablation: use removed data instead of generating new samples
            new_dataset_path = augmentor.add_removed_data(
                self.cfg.dataset.removed_data_path
            )
        else:
            # Normal case: generate new samples with diffusion model
            new_dataset_path = augmentor.augment_dataset(
                ood_indices,
                ood_features,
                ood_labels,
                cluster_assignments=cluster_assignments,
                original_dataset=original_dataset,
            )

        # Update data module with new dataset
        self.cfg.dataset.path = new_dataset_path
        self.data_module = self.create_data_module()

        return new_dataset_path

    def visualize_cycle_results(self, cycle_idx: int):
        """Create and log visualizations at the end of a training cycle"""
        print(f"Generating visualizations for cycle {cycle_idx}")

        # Setup output directory for visualizations
        vis_dir = os.path.join(self.output_dir, "visualizations", f"cycle_{cycle_idx}")
        os.makedirs(vis_dir, exist_ok=True)

        # Get the current dataset
        dataset = self.data_module.train_dataset

        # Extract features, indices, and labels from the current dataset
        features, indices, labels = self._extract_dataset_features(dataset)

        # Obtain metadata for analysis
        metadata = self._get_dataset_metadata(dataset)
        is_generated = metadata.get("is_generated", [False] * len(dataset))
        cycle_idxs = metadata.get("cycle_idx", [-1] * len(dataset))
        is_ood = metadata.get("is_ood", [False] * len(dataset))

        # Cluster the features
        gmm = GaussianMixture(
            n_components=self.cfg.experiment.bridge.num_clusters, random_state=42
        )
        features_norm = F.normalize(features, dim=1)
        cluster_assignments = gmm.fit_predict(features_norm.numpy())

        # 1. Create class distribution visualization
        class_dist_file = os.path.join(
            vis_dir, f"class_distribution_cycle_{cycle_idx}.png"
        )
        class_dist_fig = plot_class_distribution(
            labels=labels.numpy(),
            title=f"Class Distribution - Cycle {cycle_idx}",
            filename=class_dist_file,
        )

        # 2. Create cluster distribution visualization
        cluster_dist_file = os.path.join(
            vis_dir, f"cluster_distribution_cycle_{cycle_idx}.png"
        )
        cluster_dist_fig = plot_cluster_distribution(
            cluster_labels=cluster_assignments,
            class_labels=labels.numpy(),
            title=f"Cluster Distribution - Cycle {cycle_idx}",
            filename=cluster_dist_file,
        )

        # 3. Create UMAP visualization of embeddings colored by class
        # Highlight generated samples from the current cycle
        current_gen_indices = np.where(np.array(cycle_idxs) == cycle_idx - 1)[0]
        umap_class_file = os.path.join(vis_dir, f"umap_by_class_cycle_{cycle_idx}.png")
        umap_class_fig = plot_embeddings(
            embeddings=features.numpy(),
            labels=labels.numpy(),
            method="umap",
            title=f"UMAP Embeddings by Class - Cycle {cycle_idx}",
            filename=umap_class_file,
            highlight_indices=(
                current_gen_indices if len(current_gen_indices) > 0 else None
            ),
            highlight_label=(
                "Recently Generated Samples" if len(current_gen_indices) > 0 else None
            ),
        )

        # 4. Create UMAP visualization of embeddings colored by cluster
        umap_cluster_file = os.path.join(
            vis_dir, f"umap_by_cluster_cycle_{cycle_idx}.png"
        )
        umap_cluster_fig = plot_embeddings(
            embeddings=features.numpy(),
            labels=cluster_assignments,
            method="umap",
            title=f"UMAP Embeddings by Cluster - Cycle {cycle_idx}",
            filename=umap_cluster_file,
            highlight_indices=(
                current_gen_indices if len(current_gen_indices) > 0 else None
            ),
            highlight_label=(
                "Recently Generated Samples" if len(current_gen_indices) > 0 else None
            ),
        )

        # 5. Create visualization showing OOD samples from previous cycle
        if any(is_ood):
            ood_indices = np.where(np.array(is_ood))[0]
            umap_ood_file = os.path.join(
                vis_dir, f"umap_ood_samples_cycle_{cycle_idx}.png"
            )
            umap_ood_fig = plot_embeddings(
                embeddings=features.numpy(),
                labels=labels.numpy(),
                method="umap",
                title=f"OOD Samples - Cycle {cycle_idx}",
                filename=umap_ood_file,
                highlight_indices=ood_indices,
                highlight_label="OOD Samples",
            )

        # Log to wandb if available
        if self.cfg.use_wandb:
            log_dict = {
                f"class_distribution_cycle_{cycle_idx}": wandb.Image(class_dist_file),
                f"cluster_distribution_cycle_{cycle_idx}": wandb.Image(
                    cluster_dist_file
                ),
                f"umap_by_class_cycle_{cycle_idx}": wandb.Image(umap_class_file),
                f"umap_by_cluster_cycle_{cycle_idx}": wandb.Image(umap_cluster_file),
            }

            if any(is_ood):
                log_dict[f"umap_ood_samples_cycle_{cycle_idx}"] = wandb.Image(
                    umap_ood_file
                )

            # Add additional statistics
            # Number of samples by type
            num_original = sum(1 for idx in cycle_idxs if idx == -1)
            num_generated = sum(1 for gen in is_generated if gen)

            log_dict.update(
                {
                    f"num_original_samples_cycle_{cycle_idx}": num_original,
                    f"num_generated_samples_cycle_{cycle_idx}": num_generated,
                    f"total_samples_cycle_{cycle_idx}": len(dataset),
                }
            )

            wandb.log(log_dict)

        # Return features and cluster assignments for potential further use
        return features, cluster_assignments

    def _extract_dataset_features(self, dataset):
        """Extract features from a dataset using the current model"""
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.model.optimizer.batch_size,
            shuffle=False,
            num_workers=self.cfg.dataset.num_workers,
        )

        features, indices, labels = [], [], []

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting features"):
                images, batch_labels = batch
                batch_indices = torch.arange(len(indices), len(indices) + len(images))

                if torch.cuda.is_available():
                    images = images.cuda()

                # Extract features
                batch_features = self.model.get_features(images).cpu()

                features.append(batch_features)
                indices.append(batch_indices)
                labels.append(batch_labels)

        features = torch.cat(features, dim=0)
        indices = torch.cat(indices, dim=0)
        labels = torch.cat(labels, dim=0)

        return features, indices, labels

    def _get_dataset_metadata(self, dataset):
        """Extract metadata from dataset (is_generated, cycle_idx, etc.)"""
        metadata = {
            "is_generated": [],
            "cycle_idx": [],
            "is_ood": [],
            "cluster_idx": [],
        }

        # For HuggingFace datasets
        if hasattr(dataset, "features") and hasattr(dataset, "__getitem__"):
            # Check if metadata fields exist in dataset
            has_metadata = all(
                field in dataset.features
                for field in ["is_generated", "cycle_idx", "is_ood"]
            )

            if has_metadata:
                # Efficient extraction using dataset.select
                metadata["is_generated"] = [
                    dataset[i].get("is_generated", False) for i in range(len(dataset))
                ]
                metadata["cycle_idx"] = [
                    dataset[i].get("cycle_idx", -1) for i in range(len(dataset))
                ]
                metadata["is_ood"] = [
                    dataset[i].get("is_ood", False) for i in range(len(dataset))
                ]
                metadata["cluster_idx"] = [
                    dataset[i].get("cluster_idx", -1) for i in range(len(dataset))
                ]
            else:
                # Default values if metadata not present
                metadata["is_generated"] = [False] * len(dataset)
                metadata["cycle_idx"] = [-1] * len(dataset)
                metadata["is_ood"] = [False] * len(dataset)
                metadata["cluster_idx"] = [-1] * len(dataset)

        return metadata

    def train(self):
        """Run the full BRIDGE training process"""
        # Calculate epochs per cycle
        num_epochs_per_cycle = self.cfg.experiment.bridge.num_epochs_per_cycle

        # Train for each cycle
        for cycle_idx in range(self.cfg.experiment.bridge.num_cycles):
            print(
                f"\n============ Starting Cycle {cycle_idx + 1}/{self.cfg.experiment.bridge.num_cycles} ============\n"
            )

            # Train for this cycle
            learner = self.train_cycle(
                cycle_idx=cycle_idx,
                num_epochs=num_epochs_per_cycle,
            )

            # Visualize cycle results
            self.visualize_cycle_results(cycle_idx)

            # Skip OOD detection and augmentation for the last cycle
            if cycle_idx < self.cfg.experiment.bridge.num_cycles - 1:
                # Run OOD detection and get cluster assignments
                ood_indices, ood_features, ood_labels, cluster_assignments, dataset = (
                    self.run_ood_detection(cycle_idx)
                )

                # Augment dataset with cluster information
                new_dataset_path = self.augment_dataset(
                    cycle_idx,
                    ood_indices,
                    ood_features,
                    ood_labels,
                    cluster_assignments=cluster_assignments,
                    original_dataset=dataset,
                )

                # Log updated dataset path
                print(
                    f"Updated dataset path for cycle {cycle_idx + 1}: {new_dataset_path}"
                )
                if self.cfg.use_wandb:
                    wandb.log({f"dataset_path_cycle_{cycle_idx + 1}": new_dataset_path})

                # Visualize the dataset after augmentation
                print("Visualizing dataset after augmentation...")
                self.visualize_cycle_results(cycle_idx + 1)

        # Final evaluation
        print("\n============ Final Evaluation ============\n")
        self.evaluate()

        # Close wandb
        if self.cfg.use_wandb:
            wandb.finish()

    def _evaluate_representation(
        self,
        dataset_name: str,
        run_linear: bool = True,
        run_knn: bool = True,
        batch_size: int = 128,
        linear_epochs: int = 10,
        freeze_backbone: bool = True,
        unfreeze_layers: int = 0,
        mixup: float = 0.0,
        learning_rate: Optional[float] = None,
        log_prefix: str = "",
        save_results: bool = True,
    ):
        """Standardized method to evaluate representations on a dataset"""
        results = {}

        try:
            # Create dataloaders for this dataset
            dls = self._create_finetune_dataloaders(dataset_name, batch_size)

            # Extract backbone
            backbone = (
                self.model.backbone if hasattr(self.model, "backbone") else self.model
            )

            # 1. Linear Evaluation
            if run_linear:
                print(f"Running linear evaluation on {dataset_name}...")
                linear_results = self._run_linear_model(
                    dataset_name=dataset_name,
                    dls=dls,
                    backbone=backbone,
                    num_epochs=linear_epochs,
                    freeze_backbone=freeze_backbone,
                    unfreeze_layers=unfreeze_layers,
                    mixup=mixup,
                    learning_rate=learning_rate,
                    log_prefix=log_prefix,
                )
                results["linear"] = linear_results

            # 2. kNN Classification
            if run_knn:
                print(f"Running kNN classification on {dataset_name}...")
                knn_results = self._run_knn_classifier(
                    dataset_name=dataset_name,
                    dls=dls,
                    backbone=backbone,
                    log_prefix=log_prefix,
                )
                results["knn"] = knn_results

            # Save results if requested
            if save_results:
                results_path = os.path.join(
                    self.output_dir, f"{log_prefix}{dataset_name}_results.json"
                )
                with open(results_path, "w") as f:
                    json.dump(
                        {
                            "linear": results.get("linear", {}),
                            "knn": {
                                str(k): v for k, v in results.get("knn", {}).items()
                            },
                        },
                        f,
                        indent=2,
                    )

        except Exception as e:
            print(f"Error evaluating on {dataset_name}: {e}")
            results["error"] = str(e)

        return results

    def _run_linear_model(
        self,
        dataset_name: str,
        dls,
        backbone,
        num_epochs: int = 10,
        freeze_backbone: bool = True,
        unfreeze_layers: int = 0,
        mixup: float = 0.0,
        learning_rate: Optional[float] = None,
        log_prefix: str = "",
    ):
        """Train and evaluate a linear model on top of the backbone"""
        # Create a classifier model
        learn = self._create_finetune_learner(backbone, dls, freeze_backbone)

        # Unfreeze specified layers if requested
        if freeze_backbone and unfreeze_layers > 0:
            learn.unfreeze_to(-unfreeze_layers)

        # Add mixup if specified
        if mixup > 0:
            learn.add_cb(MixUp(mixup))

        # Add CSV logger
        learn.add_cb(CSVLogger(fname=f"{log_prefix}{dataset_name}_metrics"))

        # Add WandbCallback if using wandb
        if self.cfg.use_wandb:
            learn.add_cb(WandbCallback(log_preds=False, log_model=False))

            # Custom callback for nicer metric names
            class WandbMetricsCallback(Callback):
                def after_epoch(callback_self):
                    metrics = learn.recorder.metrics
                    epoch = learn.epoch
                    error_rate = metrics[0] if len(metrics) > 0 else None
                    accuracy = metrics[1] if len(metrics) > 1 else None

                    wandb.log(
                        {
                            f"{log_prefix}{dataset_name}_epoch": epoch,
                            f"{log_prefix}{dataset_name}_error_rate": error_rate,
                            f"{log_prefix}{dataset_name}_accuracy": accuracy,
                        }
                    )

            learn.add_cb(WandbMetricsCallback())

        # Find learning rate if not provided
        if learning_rate is None:
            learning_rate = learn.lr_find().valley
            print(f"Found optimal learning rate: {learning_rate}")

        # Train the model - either fine_tune (with gradual unfreezing) or fit_one_cycle
        if freeze_backbone:
            print(f"Training linear classifier for {num_epochs} epochs...")
            learn.fit_one_cycle(num_epochs, learning_rate)
        else:
            print(f"Fine-tuning model for {num_epochs} epochs...")
            learn.fine_tune(num_epochs, learning_rate)

        # Evaluate
        error_rate, accuracy = learn.validate()
        print(
            f"Final metrics: Error Rate = {error_rate:.4f}, Accuracy = {accuracy:.4f}"
        )

        # Save the model
        model_path = os.path.join(self.output_dir, f"{log_prefix}{dataset_name}_model")
        learn.save(model_path)

        # Log final results
        if self.cfg.use_wandb:
            wandb.log(
                {
                    f"{log_prefix}{dataset_name}_final_error_rate": error_rate,
                    f"{log_prefix}{dataset_name}_final_accuracy": accuracy,
                }
            )

        return {"error_rate": error_rate.item(), "accuracy": accuracy.item()}

    def _run_knn_classifier(
        self, dataset_name: str, dls, backbone, log_prefix: str = ""
    ):
        """Run kNN classification on a dataset"""
        # Create KNN evaluator
        knn_evaluator = KNNEvaluator(
            feature_extractor=backbone,
            k_values=[1, 5, 10, 20, 50, 100],
            batch_size=dls.bs,
            num_workers=self.cfg.dataset.num_workers,
        )

        # Run evaluation
        knn_results = knn_evaluator.evaluate(dls.train, dls.valid)

        # Log results
        if self.cfg.use_wandb:
            # Log accuracy for each k
            for k, result in knn_results.items():
                wandb.log(
                    {
                        f"{log_prefix}{dataset_name}_knn_k{k}_accuracy": result[
                            "accuracy"
                        ]
                    }
                )

            # Create confusion matrix plot for k=20
            if 20 in knn_results:
                cm = knn_results[20]["confusion_matrix"]
                cm_fig = plt.figure(figsize=(10, 10))
                sns.heatmap(cm, annot=False, fmt="d")
                plt.title(f"KNN (k=20) - {dataset_name}")
                plt.ylabel("True label")
                plt.xlabel("Predicted label")

                # Save and log
                cm_path = os.path.join(
                    self.output_dir, f"{log_prefix}{dataset_name}_knn_cm.png"
                )
                plt.savefig(cm_path)
                plt.close()

                wandb.log(
                    {
                        f"{log_prefix}{dataset_name}_knn_confusion_matrix": wandb.Image(
                            cm_path
                        )
                    }
                )

        return knn_results

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
            "birds": {
                "cls": vision_learner,
                "size": 224,
                "func": lambda path: untar_data(URLs.CUB_200_2011),
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
                    self.cfg.dataset.finetune_path
                    if hasattr(self.cfg.dataset, "finetune_path")
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
        learn.model.backbone = backbone

        # Freeze the backbone if needed
        if freeze_backbone:
            learn.freeze()

            # Unfreeze the last few layers if specified
            if unfreeze_layers > 0:
                learn.unfreeze_to(-unfreeze_layers)

        return learn

    def evaluate(self):
        """Comprehensive evaluation of the model on standard benchmarks"""
        print("\n========== STARTING COMPREHENSIVE EVALUATION ==========\n")

        # Define standard benchmark datasets
        standard_benchmarks = [
            "cifar10",
            "cifar100",
            "caltech101",
            "pets",
            "flowers",
            "food",
            "cars",
            "birds",
        ]

        # Results containers
        all_results = {}

        # Setup summary tables for wandb
        if self.cfg.use_wandb:
            linear_table = wandb.Table(columns=["Dataset", "Error Rate", "Accuracy"])
            knn_table = wandb.Table(
                columns=["Dataset"] + [f"k={k}" for k in [1, 5, 20, 100]]
            )

        # Run evaluations for each benchmark
        for dataset_name in standard_benchmarks:
            print(f"\n----- Evaluating on {dataset_name} dataset -----\n")

            # Run standardized evaluation (linear with frozen backbone + kNN)
            results = self._evaluate_representation(
                dataset_name=dataset_name,
                run_linear=True,
                run_knn=True,
                batch_size=128,
                linear_epochs=10,
                freeze_backbone=True,  # Always freeze for evaluation
                unfreeze_layers=0,
                mixup=0.0,  # No mixup for evaluation
                learning_rate=None,  # Auto-find LR
                log_prefix="eval_",
                save_results=True,
            )

            all_results[dataset_name] = results

            # Add to wandb tables if no error occurred
            if self.cfg.use_wandb and "error" not in results:
                # Linear results
                if "linear" in results:
                    linear_table.add_data(
                        dataset_name,
                        results["linear"]["error_rate"],
                        results["linear"]["accuracy"],
                    )

                # kNN results
                if "knn" in results:
                    knn_accs = [
                        results["knn"].get(k, {}).get("accuracy", 0)
                        for k in [1, 5, 20, 100]
                    ]
                    knn_table.add_data(dataset_name, *knn_accs)

        # Print summary of results
        print("\n========== EVALUATION SUMMARY ==========\n")
        print("Linear Evaluation Results (frozen backbone):")
        for dataset, result in all_results.items():
            if "linear" in result:
                print(
                    f"  {dataset}: Error Rate = {result['linear']['error_rate']:.4f}, "
                    f"Accuracy = {result['linear']['accuracy']:.4f}"
                )

        print("\nkNN Classification Results (k=20):")
        for dataset, result in all_results.items():
            if "knn" in result and 20 in result["knn"]:
                print(f"  {dataset}: Accuracy = {result['knn'][20]['accuracy']:.4f}")

        # Log summary tables to wandb
        if self.cfg.use_wandb:
            wandb.log(
                {
                    "linear_evaluation_summary": linear_table,
                    "knn_evaluation_summary": knn_table,
                    "evaluation_completed": True,
                }
            )

        # Save comprehensive results to disk
        results_path = os.path.join(self.output_dir, "evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump(
                {
                    dataset: {
                        "linear": results.get("linear", {}),
                        "knn": (
                            {
                                str(k): v
                                for k, v in results.get("knn", {}).items()
                                if k in [1, 5, 10, 20, 50, 100]
                            }
                            if "knn" in results
                            else {}
                        ),
                    }
                    for dataset, results in all_results.items()
                },
                f,
                indent=2,
            )

        print(f"\nEvaluation results saved to {results_path}")
        return all_results

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
        """Fine-tune model on a downstream task

        Unlike pure evaluation, this allows adaptation to downstream tasks
        with optional backbone unfreezing and mixup augmentation.
        """
        print(f"\n----- Fine-tuning on {dataset_name} dataset -----\n")

        # Load pretrained model if specified
        if pretrained_path:
            print(f"Loading pretrained model from {pretrained_path}")
            self.model.load_state_dict(torch.load(pretrained_path))

        # Use the standardized evaluation method, but with finetuning options
        results = self._evaluate_representation(
            dataset_name=dataset_name,
            run_linear=True,
            run_knn=True,  # Also do kNN to compare
            batch_size=batch_size,
            linear_epochs=num_epochs,
            freeze_backbone=freeze_backbone,
            unfreeze_layers=unfreeze_layers,
            mixup=mixup,
            learning_rate=lr,
            log_prefix="finetune_",
            save_results=True,
        )

        return results

    def fine_tune_all(
        self,
        datasets: List[str],
        num_epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 64,
        mixup: float = 0.0,
        freeze_backbone: bool = True,
        unfreeze_layers: int = 0,
        pretrained_path: Optional[str] = None,
    ):
        """Fine-tune on multiple datasets sequentially"""
        print("\n========== STARTING MULTI-DATASET FINE-TUNING ==========\n")

        # Results container
        all_results = {}

        # Finetune on each dataset
        for dataset_name in datasets:
            results = self.finetune(
                dataset_name=dataset_name,
                num_epochs=num_epochs,
                lr=lr,
                batch_size=batch_size,
                mixup=mixup,
                freeze_backbone=freeze_backbone,
                unfreeze_layers=unfreeze_layers,
                pretrained_path=(
                    pretrained_path if dataset_name == datasets[0] else None
                ),
            )

            all_results[dataset_name] = results

        # Create summary table for wandb
        if self.cfg.use_wandb:
            summary_table = wandb.Table(
                columns=["Dataset", "Mode", "Method", "Accuracy"]
            )

            for dataset, result in all_results.items():
                if "linear" in result:
                    summary_table.add_data(
                        dataset, "Finetune", "Linear", result["linear"]["accuracy"]
                    )

                if "knn" in result and 20 in result["knn"]:
                    summary_table.add_data(
                        dataset, "Finetune", "kNN (k=20)", result["knn"][20]["accuracy"]
                    )

            wandb.log({"finetune_summary": summary_table})

        # Print summary
        print("\n========== FINE-TUNING SUMMARY ==========\n")
        for dataset, result in all_results.items():
            if "linear" in result:
                print(
                    f"  {dataset} Linear: Accuracy = {result['linear']['accuracy']:.4f}"
                )
            if "knn" in result and 20 in result["knn"]:
                print(
                    f"  {dataset} kNN (k=20): Accuracy = {result['knn'][20]['accuracy']:.4f}"
                )

        return all_results
