from datasets import load_dataset
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--output-dir", type=str, required=True)
args = parser.parse_args()

# Load a dataset (e.g., cifar10, beans, food101, etc.)
dataset = load_dataset(args.dataset)

if os.path.exists(args.output_dir):
    print(f"Output directory {args.output_dir} already exists. Exiting.")
    exit(1)

# Create directories for the dataset
os.makedirs(args.output_dir, exist_ok=True)
train_h5_path = os.path.join(args.output_dir, "train.h5")
val_h5_path = os.path.join(args.output_dir, "val.h5")

# Convert the training set
with h5py.File(train_h5_path, "w") as h5:
    # Assuming there's a 'label' feature and class names are available
    if "label" in dataset["train"].features:
        class_names = dataset["train"].features["label"].names

        # Create a group for each class
        for idx, class_name in enumerate(class_names):
            class_group = h5.create_group(class_name)

            # Find examples of this class
            indices = [
                i for i, label in enumerate(dataset["train"]["label"]) if label == idx
            ]

            # Store each image
            for i, data_idx in enumerate(indices):
                img = dataset["train"]["image"][data_idx]

                # Convert PIL image to bytes
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG")
                data = np.frombuffer(buffer.getvalue(), dtype="uint8")

                # Store in h5 file
                class_group.create_dataset(
                    f"{i}.jpg", data=data, compression="gzip", compression_opts=9
                )
    else:
        # Handle datasets without explicit labels
        pass

# Convert the training set
with h5py.File(val_h5_path, "w") as h5:
    # Assuming there's a 'label' feature and class names are available
    if "label" in dataset["val"].features:
        class_names = dataset["val"].features["label"].names

        # Create a group for each class
        for idx, class_name in enumerate(class_names):
            class_group = h5.create_group(class_name)

            # Find examples of this class
            indices = [
                i for i, label in enumerate(dataset["val"]["label"]) if label == idx
            ]

            # Store each image
            for i, data_idx in enumerate(indices):
                img = dataset["val"]["image"][data_idx]

                # Convert PIL image to bytes
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG")
                data = np.frombuffer(buffer.getvalue(), dtype="uint8")

                # Store in h5 file
                class_group.create_dataset(
                    f"{i}.jpg", data=data, compression="gzip", compression_opts=9
                )
    else:
        # Handle datasets without explicit labels
        pass
