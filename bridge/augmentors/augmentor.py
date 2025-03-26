import os
import torch
import numpy as np
from typing import List, Tuple, Dict, Any
from PIL import Image
import wandb
from tqdm import tqdm
from huggingface_hub import HfApi, snapshot_download, login
from datasets import Dataset, load_dataset, Image as HFImage

# For diffusion models
from diffusers import (
    StableDiffusionImageVariationPipeline,
    DDPMPipeline,
    DDIMPipeline,
    DiffusionPipeline
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
