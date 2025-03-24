import os
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from omegaconf import DictConfig, OmegaConf
import hydra
from pytorch_lightning import Trainer
from datasets import Dataset as HFDataset
from huggingface_hub import HfApi

from solo_learn.methods.base import BaseMethod
from solo_learn.utils.misc import make_contiguous
from solo_learn.data.dataset import prepare_datasets, prepare_hf_datasets
from solo_learn.data.classification_dataloader import prepare_dataloaders
from solo_learn.utils.auto_resumer import AutoResumer
from solo_learn.utils.checkpointer import Checkpointer


class OODDetector:
    """Detects out-of-distribution samples using Gaussian Mixture Models."""

    def __init__(self, num_clusters: int, num_ood_points_per_cluster: int):
        """
        Args:
            num_clusters: number of clusters to use in GMM
            num_ood_points_per_cluster: number of OOD points to select per cluster
        """
        self.num_clusters = num_clusters
        self.num_ood_points_per_cluster = num_ood_points_per_cluster

    def fit_and_detect(
        self, features: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fits a GMM to the features and detects OOD samples.

        Args:
            features: feature vectors from the model (N, D)
            labels: class labels (N,)

        Returns:
            ood_indices: indices of OOD samples (K*C,) where K=num_ood_points_per_cluster, C=num_clusters
            ood_labels: class labels of OOD samples (K*C,)
            cluster_assignments: cluster assignments for all samples (N,)
        """
        from sklearn.mixture import GaussianMixture

        # Fit GMM
        gmm = GaussianMixture(n_components=self.num_clusters, random_state=0)
        cluster_assignments = gmm.fit_predict(features)

        # Get probabilities
        probs = gmm.score_samples(features)

        # For each cluster, find the samples with lowest probability (most OOD)
        ood_indices = []
        for cluster_idx in range(self.num_clusters):
            cluster_mask = cluster_assignments == cluster_idx
            cluster_probs = probs[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]

            # Get indices of samples with lowest probabilities (most OOD)
            if len(cluster_indices) <= self.num_ood_points_per_cluster:
                # If cluster is small, take all samples
                ood_cluster_indices = cluster_indices
            else:
                # Sort by probability (ascending) and take lowest K
                sorted_idx = np.argsort(cluster_probs)[
                    : self.num_ood_points_per_cluster
                ]
                ood_cluster_indices = cluster_indices[sorted_idx]

            ood_indices.extend(ood_cluster_indices)

        ood_indices = np.array(ood_indices)
        ood_labels = labels[ood_indices]

        return ood_indices, ood_labels, cluster_assignments


class DiffusionAugmenter:
    """Augments images using a diffusion model."""

    def __init__(self, model_name: str = "stabilityai/stable-diffusion-2-1"):
        """
        Args:
            model_name: name of the diffusion model to use
        """
        self.model_name = model_name

    def generate_similar_images(self, images: List, n_samples: int = 5) -> List:
        """
        Generates similar images using the diffusion model.

        Args:
            images: list of images to augment
            n_samples: number of new samples to generate per image

        Returns:
            augmented_images: list of augmented images
        """
        # This is a placeholder - you would implement the actual diffusion logic here
        # For example, using diffusers library with Stable Diffusion
        from diffusers import StableDiffusionImg2ImgPipeline
        import torch

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model_name, torch_dtype=torch.float16
        ).to("cuda")

        all_generated = []
        all_original = []  # Store original images for logging

        for image in images:
            all_original.append(image)
            # Generate variations with low strength to keep similarity
            outputs = pipe(
                image=image,
                prompt="same image, high quality",
                num_images_per_prompt=n_samples,
                strength=0.3,  # Low strength preserves more of original
            ).images
            all_generated.extend(outputs)

        # Log to wandb
        self._log_to_wandb(all_original, all_generated, n_samples)

        return all_generated

    def _log_to_wandb(
        self, original_images: List, generated_images: List, n_samples: int
    ):
        """
        Logs original and generated images to wandb.

        Args:
            original_images: list of original images
            generated_images: list of generated images
            n_samples: number of samples generated per original image
        """
        if not wandb.run:
            return  # Skip if wandb is not initialized

        # Convert all images to PIL if they aren't already
        to_pil = ToPILImage()

        for i, orig_img in enumerate(original_images):
            # Ensure it's a PIL image
            if not isinstance(orig_img, Image.Image):
                orig_img = to_pil(orig_img)

            # Resize to smaller dimensions for wandb (optional, to save bandwidth)
            orig_img = orig_img.resize((128, 128), Image.LANCZOS)

            # Get corresponding generated images
            gen_imgs = []
            for j in range(n_samples):
                idx = i * n_samples + j
                if idx < len(generated_images):
                    gen_img = generated_images[idx]
                    if not isinstance(gen_img, Image.Image):
                        gen_img = to_pil(gen_img)
                    gen_img = gen_img.resize((128, 128), Image.LANCZOS)
                    gen_imgs.append(gen_img)

            # Create a wandb Image with caption
            wandb_images = [wandb.Image(orig_img, caption="Original")]
            for j, img in enumerate(gen_imgs):
                wandb_images.append(wandb.Image(img, caption=f"Generated {j+1}"))

            # Log to wandb
            wandb.log(
                {
                    f"diffusion/sample_{i}": wandb_images,
                }
            )


def extract_features(
    model: BaseMethod, loader: DataLoader
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts features from the model for all samples in the loader.

    Args:
        model: trained model
        loader: data loader

    Returns:
        features: feature vectors (N, D)
        labels: class labels (N,)
    """
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for batch in loader:
            img, labels = batch
            img = make_contiguous(img)
            features = model.backbone(img.to(model.device))
            features = features.cpu().numpy()
            features_list.append(features)
            labels_list.append(labels.numpy())

    features = np.concatenate(features_list)
    labels = np.concatenate(labels_list)

    return features, labels


def save_to_huggingface(
    dataset: Dataset,
    original_dataset_name: str,
    cycle_idx: int,
    num_clusters: int,
    num_ood_points: int,
    repo_id: str = None,
) -> str:
    """
    Saves a dataset to HuggingFace.

    Args:
        dataset: dataset to save
        original_dataset_name: name of original dataset
        cycle_idx: current cycle index
        num_clusters: number of clusters used
        num_ood_points: number of OOD points per cluster
        repo_id: repository ID (will be created if None)

    Returns:
        repo_id: repository ID of saved dataset
    """
    # Convert PyTorch dataset to HuggingFace dataset
    images = []
    labels = []

    for i in range(len(dataset)):
        img, label = dataset[i]
        images.append(img)
        labels.append(label)

    hf_dataset = HFDataset.from_dict({"image": images, "label": labels})

    # Create repo ID if not provided
    if repo_id is None:
        # Example: "username/cifar10-bridge-c3-n10-e5"
        dataset_name = original_dataset_name.split("/")[-1]
        repo_id = f"{HfApi().whoami()['name']}/{dataset_name}-bridge-c{cycle_idx}-k{num_clusters}-n{num_ood_points}"

    # Push to hub
    hf_dataset.push_to_hub(repo_id)

    return repo_id


@hydra.main(config_path="config", config_name="pretrain")
def main(cfg: DictConfig):
    # Initialize model and other components as in main_pretrain.py
    # [Same initialization code as in main_pretrain.py]

    # Add specific bridge configurations
    bridge_cfg = OmegaConf.create(
        {
            "num_epochs_per_cycle": 10,
            "num_cycles": 5,
            "num_clusters": 5,
            "num_ood_points_per_cluster": 10,
            "diffusion_model": "stabilityai/stable-diffusion-2-1",
            "n_augmentations_per_sample": 5,
        }
    )

    if "bridge" in cfg:
        bridge_cfg = OmegaConf.merge(bridge_cfg, cfg.bridge)

    # Track the current dataset name
    current_dataset_name = cfg.data.dataset_name

    # Initialize components
    ood_detector = OODDetector(
        num_clusters=bridge_cfg.num_clusters,
        num_ood_points_per_cluster=bridge_cfg.num_ood_points_per_cluster,
    )

    augmenter = DiffusionAugmenter(model_name=bridge_cfg.diffusion_model)

    # Run the bridge cycles
    for cycle_idx in range(bridge_cfg.num_cycles):
        print(f"=== Starting Bridge Cycle {cycle_idx+1}/{bridge_cfg.num_cycles} ===")
        print(f"Training with dataset: {current_dataset_name}")

        # Determine if we're using a HuggingFace dataset
        is_hf_dataset = (
            current_dataset_name.startswith("hf://") or "/" in current_dataset_name
        )

        # Prepare datasets
        if is_hf_dataset:
            train_dataset, val_dataset = prepare_hf_datasets(
                dataset_name=current_dataset_name.replace("hf://", ""),
                train_transform=cfg.data.augmentations.train_transform,
                val_transform=cfg.data.augmentations.val_transform,
                streaming=cfg.data.get("streaming", False),
                image_key=cfg.data.get("image_key", "image"),
                label_key=cfg.data.get("label_key", "label"),
            )
        else:
            train_dataset, val_dataset = prepare_datasets(
                dataset=cfg.data.dataset,
                data_dir=cfg.data.data_dir,
                train_dir=cfg.data.train_dir,
                val_dir=cfg.data.val_dir,
                train_transform=cfg.data.augmentations.train_transform,
                val_transform=cfg.data.augmentations.val_transform,
                no_labels=False,
            )

        # Prepare data loaders
        train_loader, val_loader = prepare_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=cfg.optimizer.batch_size,
            num_workers=cfg.data.num_workers,
        )

        # Create trainer and model (similar to main_pretrain.py)
        # [Same model and trainer setup code as in main_pretrain.py]

        # Train for N epochs
        trainer.fit(model, train_loader, val_loader)

        # Skip OOD detection and augmentation for the last cycle
        if cycle_idx == bridge_cfg.num_cycles - 1:
            break

        # Extract features
        print("Extracting features for OOD detection...")
        features, labels = extract_features(model, train_loader)

        # Detect OOD samples
        print("Detecting OOD samples...")
        ood_indices, ood_labels, cluster_assignments = ood_detector.fit_and_detect(
            features, labels
        )

        # Get OOD images
        ood_images = []
        for idx in ood_indices:
            img, _ = train_dataset[idx]
            ood_images.append(img)

        # Generate similar images
        print("Generating new images with diffusion model...")
        augmented_images = augmenter.generate_similar_images(
            images=ood_images, n_samples=bridge_cfg.n_augmentations_per_sample
        )

        # Create new augmented dataset
        # (In practice, you might want to add these to the original dataset)
        print("Creating and uploading augmented dataset...")

        # Save to HuggingFace
        repo_id = save_to_huggingface(
            dataset=train_dataset,  # You'd add the augmented samples here
            original_dataset_name=current_dataset_name,
            cycle_idx=cycle_idx,
            num_clusters=bridge_cfg.num_clusters,
            num_ood_points=bridge_cfg.num_ood_points_per_cluster,
        )

        # Update current dataset name for next cycle
        current_dataset_name = repo_id

        print(f"Completed cycle {cycle_idx+1}. New dataset: {current_dataset_name}")

    print("Bridge pretraining complete!")


if __name__ == "__main__":
    main()
