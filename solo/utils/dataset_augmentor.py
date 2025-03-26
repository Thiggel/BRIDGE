import os
import shutil
import h5py
import wandb
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List
from diffusers import (
    StableDiffusionImageVariationPipeline,
    FluxPriorReduxPipeline,
    FluxPipeline,
)


class FluxAugmentor:
    def __init__(self):
        self.pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Redux-dev", torch_dtype=torch.bfloat16
        ).to("cuda")
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            text_encoder=None,
            text_encoder_2=None,
            torch_dtype=torch.bfloat16,
        ).to("cuda")

    def augment(self, images, num_generations_per_image=1):
        pipe_prior_output = self.pipe_prior_redux(images)

        return self.pipe(
            num_images_per_prompt=num_generations_per_image,
            num_inference_steps=20,
            **pipe_prior_output,
        ).images


class StableDiffusionAugmentor:
    def __init__(self):
        self.pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            "lambdalabs/sd-image-variations-diffusers",
            revision="v2.0",
            safety_checker=None,
        ).to("cuda")

    def augment(self, images, num_generations_per_image=1):
        return self.pipe(
            images,
            num_inference_steps=20,
            num_images_per_prompt=num_generations_per_image,
        ).images


class DatasetAugmentor:
    # 3. for each ood datapoint, generate num_generations_per_ood_sample new datapoints
    #   3.1. load stable diffusion or other model for similar image generation
    #   3.2. generate new datapoints
    #   3.3. copy current dataset to new dataset with all existing plus new datapoints, give it a name based on the cycle, the number of generated data points etc.
    #   3.4. change dataset path in cfg to the new dataset
    def __init__(
        self,
        num_generations_per_ood_sample: int,
        diffusion_model: str,
        batch_size: int,
        num_workers: int,
        experiment_name: str,
        cycle_idx: int,
        dataset_path: str,
    ):
        self.num_generations_per_ood_sample = num_generations_per_ood_sample
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.experiment_name = experiment_name
        self.cycle_idx = cycle_idx
        self.dataset_path = dataset_path
        self.diffusion_augmentor = (
            FluxAugmentor() if diffusion_model == "flux" else StableDiffusionAugmentor()
        )
        # Use a file path for the H5 dataset (with .h5 extension)
        self.new_dataset_path = os.path.join(
            os.environ.get("BASE_CACHE_DIR", "/tmp"),
            "dataset",
            experiment_name,
            f"cycle_{cycle_idx}.h5",
        )

    def augment_dataset(self, ood_images, ood_features, ood_classes):
        # Copy the original H5 dataset file
        shutil.copy(self.dataset_path, self.new_dataset_path)
        generated_images_all = []

        # Open the H5 file once and update groups/datasets inside the loop
        with h5py.File(self.new_dataset_path, "a") as new_dataset:
            for i, (ood_feature, ood_class) in enumerate(
                zip(ood_features, ood_classes)
            ):
                ood_feature = ood_feature.unsqueeze(0).to("cuda")
                new_images = self.diffusion_augmentor.augment(
                    ood_feature, self.num_generations_per_ood_sample
                )
                new_images = new_images.cpu().numpy()

                # Use the ood_class as group name
                group_name = str(ood_class)
                if group_name not in new_dataset:
                    new_dataset.create_group(group_name)

                for j, new_image in enumerate(new_images):
                    img_name = f"{ood_class}_{i}_{j}"
                    new_dataset[group_name].create_dataset(img_name, data=new_image)

                generated_images_all.extend(new_images)

        # Log all generated images instead of just the last iteration
        if wandb.run is not None:
            wandb.log(
                {
                    f"ood_images_{self.cycle_idx}": [
                        wandb.Image(img) for img in ood_images
                    ],
                    f"generated_images_{self.cycle_idx}": [
                        wandb.Image(img) for img in generated_images_all
                    ],
                }
            )

        return self.new_dataset_path
