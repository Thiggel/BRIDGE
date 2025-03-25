import os
import argparse
import h5py
import io
from PIL import Image
import requests
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from pathlib import Path
import shutil

parser = argparse.ArgumentParser(
    description="Download dataset and convert to h5 format for solo-learn"
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="Name of the dataset (huggingface dataset or 'cifar10', 'cifar100', 'stl10', 'imagenet', 'imagenet100')",
)
parser.add_argument(
    "--output-dir", type=str, required=True, help="Output directory for h5 files"
)
parser.add_argument(
    "--split",
    type=str,
    default="train",
    help="Dataset split (e.g., train, test, validation)",
)
parser.add_argument(
    "--subset",
    type=int,
    default=-1,
    help="Number of examples to use per class (for debugging). Default: -1 (all)",
)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)


def get_image_bytes(img):
    """Convert PIL Image to bytes buffer"""
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    return img_byte_arr.getvalue()


def process_dataset():
    """Download the dataset and convert it to h5 format"""
    print(f"Loading dataset: {args.dataset}, split: {args.split}")

    # Load the dataset
    if args.dataset in ["cifar10", "cifar100", "stl10"]:
        dataset = load_dataset(args.dataset, split=args.split)
    elif args.dataset in ["imagenet", "imagenet100"]:
        if args.dataset == "imagenet100":
            # Use a subset of ImageNet classes
            with open("solo/data/dataset_subset/imagenet100_classes.txt", "r") as f:
                imagenet100_classes = f.readline().strip().split()
            dataset = load_dataset("imagenet-1k", split=args.split)
            # Filter to only include imagenet100 classes
            dataset = dataset.filter(
                lambda example: example["label"] in imagenet100_classes
            )
        else:
            dataset = load_dataset("imagenet-1k", split=args.split)
    else:
        # Use HuggingFace datasets
        dataset = load_dataset(args.dataset, split=args.split)

    # Get class information
    if "label" in dataset.features:
        if hasattr(dataset.features["label"], "names"):
            class_names = dataset.features["label"].names
        else:
            # Create numeric class names if names not available
            unique_labels = set(dataset["label"])
            class_names = [str(i) for i in range(len(unique_labels))]
    else:
        # If no labels, create a single dummy class
        class_names = ["no_label"]

    # Create h5 file
    output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.split}.h5")
    print(f"Creating h5 file: {output_file}")

    with h5py.File(output_file, "w") as h5f:
        # Create group for each class
        for class_name in class_names:
            h5f.create_group(class_name)

        # Process all examples
        for idx, example in enumerate(tqdm(dataset, desc="Processing images")):
            # Get image and label
            if "image" in example:
                image = example["image"]
                if not isinstance(image, Image.Image):
                    image = Image.fromarray(image)
            elif "img" in example:
                image = example["img"]
                if not isinstance(image, Image.Image):
                    image = Image.fromarray(image)
            elif "pixel_values" in example:
                # If the dataset provides pixel values directly
                pixel_values = example["pixel_values"]
                if isinstance(pixel_values, torch.Tensor):
                    pixel_values = pixel_values.numpy()
                # Convert from CHW to HWC if necessary
                if pixel_values.shape[0] == 3:
                    pixel_values = np.transpose(pixel_values, (1, 2, 0))
                image = Image.fromarray((pixel_values * 255).astype(np.uint8))
            else:
                raise ValueError(f"Cannot find image data in example: {example.keys()}")

            # Get label
            if "label" in example:
                label = example["label"]
                class_name = (
                    class_names[label] if isinstance(label, int) else str(label)
                )
            else:
                class_name = "no_label"

            # Convert image to bytes and store in h5 file
            image_bytes = get_image_bytes(image)
            image_name = f"img_{idx}.jpg"
            h5f[class_name][image_name] = np.void(image_bytes)

            # Limit examples per class if subset is specified
            if args.subset > 0 and (idx + 1) >= args.subset * len(class_names):
                break

    print(f"Dataset saved to {output_file}")
    print(f"To use with solo-learn, configure your YAML with:")
    print(f"data:")
    print(f"  dataset: {args.dataset}")
    print(f'  train_path: "{output_file}"')
    print(f'  format: "h5"')


if __name__ == "__main__":
    process_dataset()
