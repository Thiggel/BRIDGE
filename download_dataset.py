from datasets import load_dataset
import os
import argparse
import io
from PIL import Image
import shutil
import numpy as np

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
train_dir = os.path.join(args.output_dir, "train")
val_dir = os.path.join(args.output_dir, "val")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Process training data
print("Processing training data...")
if "train" in dataset:
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

        # Create directories for each class
        for class_name in class_names:
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)

        # Save images to respective class directories
        for i, (img, label) in enumerate(
            zip(dataset["train"]["image"], dataset["train"]["label"])
        ):
            if i % 100 == 0:
                print(f"Processing training image {i}/{len(dataset['train'])}")

            class_name = class_names[label]
            img_path = os.path.join(train_dir, class_name, f"{i}.jpg")

            # Handle both PIL images and numpy arrays
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.uint8(img))

            img.save(img_path)
    else:
        # No labels - flat directory structure
        for i, img in enumerate(dataset["train"]["image"]):
            if i % 100 == 0:
                print(f"Processing training image {i}/{len(dataset['train'])}")

            img_path = os.path.join(train_dir, f"{i}.jpg")

            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.uint8(img))

            img.save(img_path)

# Process validation data
print("Processing validation data...")
if "validation" in dataset:
    val_key = "validation"
elif "val" in dataset:
    val_key = "val"
elif "test" in dataset:
    val_key = "test"
else:
    val_key = None

if val_key:
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

        # Create directories for each class
        for class_name in class_names:
            os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        # Save images to respective class directories
        for i, (img, label) in enumerate(
            zip(dataset[val_key]["image"], dataset[val_key]["label"])
        ):
            if i % 100 == 0:
                print(f"Processing validation image {i}/{len(dataset[val_key])}")

            class_name = class_names[label]
            img_path = os.path.join(val_dir, class_name, f"{i}.jpg")

            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.uint8(img))

            img.save(img_path)
    else:
        # No labels - flat directory structure
        for i, img in enumerate(dataset[val_key]["image"]):
            if i % 100 == 0:
                print(f"Processing validation image {i}/{len(dataset[val_key])}")

            img_path = os.path.join(val_dir, f"{i}.jpg")

            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.uint8(img))

            img.save(img_path)

print(f"Dataset converted and saved to {args.output_dir}")
print(f"To use with solo-learn, configure your YAML with:")
print(f"data:")
print(f"  dataset: custom")
print(f'  train_path: "{train_dir}"')
print(f'  val_path: "{val_dir}"')
print(f'  format: "image_folder"')
print(f"  no_labels: {not has_labels}")
