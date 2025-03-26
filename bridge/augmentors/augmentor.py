import os
import torch
import numpy as np
from typing import List, Tuple, Dict, Any
from PIL import Image
import wandb
from tqdm import tqdm
from huggingface_hub import HfApi, snapshot_download, login
from datasets import Dataset, load_dataset, Image as HFImage, concatenate_datasets
from tqdm import tqdm

# For diffusion models
from diffusers import (
    StableDiffusionImageVariationPipeline,
    DDPMPipeline,
    DDIMPipeline,
    DiffusionPipeline,
)


class DiffusionAugmentor:
    """Base class for diffusion model-based augmentation"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """Load diffusion model"""
        raise NotImplementedError("Subclasses must implement this method")

    def augment(self, images, num_generations_per_image=1):
        """Generate new samples based on input images"""
        raise NotImplementedError("Subclasses must implement this method")


class StableDiffusionAugmentor(DiffusionAugmentor):
    """Augmentor using Stable Diffusion image variation models"""

    def __init__(self, model_path: str = "lambdalabs/sd-image-variations-diffusers"):
        super().__init__(model_path)
        self.pipe = None
        self.load_model()

    def load_model(self):
        """Load Stable Diffusion image variation model"""
        self.pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            self.model_path,
            safety_checker=None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

    def augment(self, images, num_generations_per_image=1):
        """Generate variations of the input images"""
        # Preprocess images if needed
        pil_images = []
        for img in images:
            # Convert tensor to PIL Image if necessary
            if isinstance(img, torch.Tensor):
                # Ensure shape is [C, H, W]
                if img.dim() == 4:
                    img = img.squeeze(0)

                # Rescale to [0, 1] if in [-1, 1]
                if img.min() < 0:
                    img = (img + 1) / 2

                # Convert to uint8 PIL image
                img = (img * 255).byte().permute(1, 2, 0).cpu().numpy()
                img = Image.fromarray(img)

            # Resize to expected dimensions (512x512 for SD models)
            img = img.resize((512, 512))
            pil_images.append(img)

        # Generate variations
        generated_images = []
        for img in pil_images:
            with torch.no_grad():
                outputs = self.pipe(
                    img,
                    num_images_per_prompt=num_generations_per_image,
                    num_inference_steps=25,
                )
                generated_images.extend(outputs.images)

        return generated_images


class DatasetAugmentor:
    """Augments a dataset with generated samples from underrepresented areas"""

    def __init__(
        self,
        num_generations_per_sample: int,
        diffusion_model: str,
        batch_size: int,
        num_workers: int,
        experiment_name: str,
        cycle_idx: int,
        original_dataset_path: str,
        hf_token: str = None,
        output_dir: str = "/tmp",
    ):
        self.num_generations_per_sample = num_generations_per_sample
        self.diffusion_model = diffusion_model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.experiment_name = experiment_name
        self.cycle_idx = cycle_idx
        self.original_dataset_path = original_dataset_path
        self.hf_token = hf_token
        self.output_dir = output_dir

        # Initialize diffusion augmentor
        self.augmentor = self._get_augmentor(diffusion_model)

        # Set the new dataset path/name
        self.new_dataset_name = f"{experiment_name}_cycle_{cycle_idx}"
        self.new_dataset_path = os.path.join("flaitenberger", self.new_dataset_name)

    def _get_augmentor(self, diffusion_model):
        """Get the appropriate diffusion augmentor"""
        # For now, just use the Stable Diffusion augmentor
        # Future: add more options based on the diffusion_model parameter
        return StableDiffusionAugmentor(diffusion_model)

    def augment_dataset(
        self,
        ood_indices,
        ood_features,
        ood_labels,
        cluster_assignments=None,
        original_dataset=None,
    ):
        """Augment the dataset with generated samples

        Args:
            ood_indices: Indices of OOD samples
            ood_features: Features of OOD samples
            ood_labels: Labels of OOD samples
            cluster_assignments: Optional array of cluster assignments for all samples
            original_dataset: Optional pre-loaded dataset to avoid loading again
        """
        # Use provided dataset or load it
        if original_dataset is None:
            print(f"Loading original dataset from {self.original_dataset_path}")
            original_dataset = load_dataset(self.original_dataset_path, split="train")

        # Extract OOD samples
        ood_samples = original_dataset.select(ood_indices.tolist())

        # Get original images from OOD samples
        ood_images = [sample["image"] for sample in ood_samples]

        # Map OOD indices to their cluster assignments if available
        ood_clusters = {}
        if cluster_assignments is not None:
            for i, idx in enumerate(ood_indices):
                ood_clusters[idx.item()] = cluster_assignments[idx].item()

        # Generate new samples using the diffusion model
        print(
            f"Generating {self.num_generations_per_sample} new samples for each of {len(ood_images)} OOD points"
        )
        generated_images = self.augmentor.augment(
            ood_images, self.num_generations_per_sample
        )

        # Prepare new samples data for merging with original dataset
        new_samples_data = {
            "image": [],
            "label": [],
            "is_generated": [],
            "cycle_idx": [],
            "cluster_idx": [],
            "parent_idx": [],  # Track which OOD sample generated this one
            "is_ood": [],
        }

        # Add generated samples
        for i, (ood_idx, label) in enumerate(zip(ood_indices, ood_labels)):
            # Get the cluster for this OOD sample
            cluster = ood_clusters.get(ood_idx.item(), -1)

            for j in range(self.num_generations_per_sample):
                # Get the corresponding generated image
                gen_idx = i * self.num_generations_per_sample + j
                if gen_idx < len(generated_images):
                    new_samples_data["image"].append(generated_images[gen_idx])
                    new_samples_data["label"].append(label.item())
                    new_samples_data["is_generated"].append(True)
                    new_samples_data["cycle_idx"].append(self.cycle_idx)
                    new_samples_data["cluster_idx"].append(cluster)
                    new_samples_data["parent_idx"].append(ood_idx.item())
                    new_samples_data["is_ood"].append(False)

        # Create dataset with new samples
        new_samples_dataset = Dataset.from_dict(new_samples_data)

        # Add metadata to original dataset if it doesn't have it
        def add_metadata(example, idx):
            return {
                "is_generated": False,
                **example,
                "cycle_idx": -1,  # -1 indicates original data
                "cluster_idx": (
                    cluster_assignments[idx].item()
                    if cluster_assignments is not None
                    else -1
                ),
                "parent_idx": -1,  # Original data has no parent
                "is_ood": idx in ood_indices.tolist(),
            }

    def add_removed_data(self, removed_data_path):
        """Ablation: instead of generating new data, add back previously removed data"""
        # Load original and removed datasets
        original_dataset = load_dataset(self.original_dataset_path, split="train")
        removed_dataset = load_dataset(removed_data_path, split="train")

        # Add metadata
        def add_original_metadata(example):
            return {
                **example,
                "is_generated": False,
                "cycle_idx": -1,
                "is_removed": False,
            }

        def add_removed_metadata(example):
            return {
                **example,
                "is_generated": False,
                "cycle_idx": self.cycle_idx,
                "is_removed": True,
            }

        original_with_meta = original_dataset.map(add_original_metadata)
        removed_with_meta = removed_dataset.map(add_removed_metadata)

        # Concatenate datasets
        augmented_dataset = concatenate_datasets(
            [original_with_meta, removed_with_meta]
        )

        # Save the combined dataset
        local_path = os.path.join(
            self.output_dir, "datasets", f"{self.new_dataset_name}_with_removed"
        )
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        augmented_dataset.save_to_disk(local_path)
        self.new_dataset_path = local_path

        print(f"Created dataset with removed data at {local_path}")
        return local_path
