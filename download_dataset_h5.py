#!/usr/bin/env python
import os
import argparse
import h5py
import io
from PIL import Image
import numpy as np
from datasets import load_dataset
from pathlib import Path
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--output-dir", type=str, required=True)
args = parser.parse_args()

# Load dataset
dataset = load_dataset(args.dataset)
print(f"Loaded dataset: {args.dataset}")

if os.path.exists(args.output_dir):
    print(f"Output directory {args.output_dir} already exists. Exiting.")
    exit(1)

# Create base directories
os.makedirs(args.output_dir, exist_ok=True)


# Function to convert PIL Image to bytes
def image_to_bytes(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.uint8(img))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    return img_byte_arr.getvalue()


# Process training data
print("Processing training data...")
if "train" in dataset:
    train_h5_path = os.path.join(args.output_dir, "train.h5")
    with h5py.File(train_h5_path, "w") as h5f:
        # Check if dataset has labels
        has_labels = "label" in dataset["train"].features

        if has_labels:
            # Get class names if available
            if hasattr(dataset["train"].features["label"], "names"):
                class_names = dataset["train"].features["label"].names
            else:
                # Create numeric class names if names not available
                class_names = [
                    str(i) for i in range(len(np.unique(dataset["train"]["label"])))
                ]

            # Create groups for each class
            for class_name in class_names:
                h5f.create_group(class_name)

            # Save images to respective class groups
            for i, (img, label) in enumerate(
                zip(dataset["train"]["image"], dataset["train"]["label"])
            ):
                if i % 100 == 0:
                    print(f"Processing training image {i}/{len(dataset['train'])}")

                class_name = class_names[label]
                img_name = f"{i}.jpg"

                # Convert to bytes
                img_bytes = image_to_bytes(img)

                # Store in h5 file
                h5f[class_name][img_name] = np.void(img_bytes)
        else:
            # No labels - single group
            no_label_group = h5f.create_group("no_label")

            # Save images to the single group
            for i, img in enumerate(dataset["train"]["image"]):
                if i % 100 == 0:
                    print(f"Processing training image {i}/{len(dataset['train'])}")

                img_name = f"{i}.jpg"

                # Convert to bytes
                img_bytes = image_to_bytes(img)

                # Store in h5 file
                no_label_group[img_name] = np.void(img_bytes)

# Process validation data
print("Processing validation data...")
val_key = None
if "validation" in dataset:
    val_key = "validation"
elif "val" in dataset:
    val_key = "val"
elif "test" in dataset:
    val_key = "test"

if val_key:
    val_h5_path = os.path.join(args.output_dir, "val.h5")
    with h5py.File(val_h5_path, "w") as h5f:
        # Check if dataset has labels
        has_labels = "label" in dataset[val_key].features

        if has_labels:
            # Get class names if available
            if hasattr(dataset[val_key].features["label"], "names"):
                class_names = dataset[val_key].features["label"].names
            else:
                # Create numeric class names if names not available
                class_names = [
                    str(i) for i in range(len(np.unique(dataset[val_key]["label"])))
                ]

            # Create groups for each class
            for class_name in class_names:
                h5f.create_group(class_name)

            # Save images to respective class groups
            for i, (img, label) in enumerate(
                zip(dataset[val_key]["image"], dataset[val_key]["label"])
            ):
                if i % 100 == 0:
                    print(f"Processing validation image {i}/{len(dataset[val_key])}")

                class_name = class_names[label]
                img_name = f"{i}.jpg"

                # Convert to bytes
                img_bytes = image_to_bytes(img)

                # Store in h5 file
                h5f[class_name][img_name] = np.void(img_bytes)
        else:
            # No labels - single group
            no_label_group = h5f.create_group("no_label")

            # Save images to the single group
            for i, img in enumerate(dataset[val_key]["image"]):
                if i % 100 == 0:
                    print(f"Processing validation image {i}/{len(dataset[val_key])}")

                img_name = f"{i}.jpg"

                # Convert to bytes
                img_bytes = image_to_bytes(img)

                # Store in h5 file
                no_label_group[img_name] = np.void(img_bytes)

print(f"Dataset converted and saved to {args.output_dir}")
print(f"To use with solo-learn, configure your YAML with:")
print(f"data:")
print(f"  dataset: custom")
print(f'  train_path: "{os.path.join(args.output_dir, "train.h5")}"')
if val_key:
    print(f'  val_path: "{os.path.join(args.output_dir, "val.h5")}"')
print(f'  format: "h5"')
print(f"  no_labels: {not has_labels}")
