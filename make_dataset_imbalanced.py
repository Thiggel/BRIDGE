import torch
import numpy as np
import os
import argparse
from huggingface_hub import login
from datasets import load_dataset, concatenate_datasets
from dotenv import load_dotenv


class Imbalancedness:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def get_imbalance(self, class_index: int) -> float:
        pass


class PowerLawImbalancedness(Imbalancedness):
    """
    Implements ImageNet-LT style imbalance following Pareto distribution with α=6.
    Ensures 1280 samples for most frequent class and 5 samples for least frequent.
    """

    def __init__(
        self, num_classes: int, alpha: float, max_samples: int, min_samples: int
    ):
        super().__init__(num_classes)
        self.alpha = alpha
        self.max_samples = max_samples
        self.min_samples = min_samples

        self.class_ratios = self._calculate_class_ratios()

    def _calculate_class_ratios(self) -> torch.Tensor:
        """
        Calculate the ratio of samples to keep for each class following Pareto distribution.
        Returns ratios in range [0, 1] where 1 corresponds to max_samples.
        """
        # Generate Pareto distribution values
        x = np.arange(1, self.num_classes + 1, dtype=np.float32)
        pareto_vals = x ** (-1 / self.alpha)

        # Scale to [min_samples, max_samples]
        scaled_vals = (pareto_vals - pareto_vals.min()) / (
            pareto_vals.max() - pareto_vals.min()
        )
        scaled_vals = (
            scaled_vals * (self.max_samples - self.min_samples) + self.min_samples
        )

        # Convert to ratios
        ratios = scaled_vals / self.max_samples

        return torch.from_numpy(ratios)

    def get_imbalance(self, class_indices: torch.Tensor) -> torch.Tensor:
        """
        Get imbalance scores for given class indices.
        Args:
            class_indices: Tensor of class indices
        Returns:
            Tensor of same shape as class_indices with imbalance scores
        """
        if not isinstance(class_indices, torch.Tensor):
            class_indices = torch.tensor(class_indices)

        # Move ratios to same device as indices
        ratios = self.class_ratios.to(class_indices.device)

        # Return ratio for each class index
        return ratios[class_indices]


class NoImbalancedness(Imbalancedness):
    def __init__(
        self, num_classes: int, alpha: float, max_samples: int, min_samples: int
    ):
        super().__init__(num_classes)

        self.power_law_imbalance = PowerLawImbalancedness(
            num_classes, alpha, max_samples, min_samples
        )
        self.total_imbalance = self.get_total_imbalance()

    def get_power_law_imbalance(self, class_index: int) -> float:
        return self.power_law_imbalance.get_imbalance(class_index)

    def get_total_imbalance(self) -> float:
        """
        We want the same amount of data in both the imbalanced
        and balanced case. Therefore, we calculate the average
        imbalance across classes and take away this data in
        a uniform way.
        """
        total_imbalance = 0.0
        for class_idx in range(self.num_classes):
            total_imbalance += self.get_power_law_imbalance(class_idx)

        return total_imbalance / self.num_classes

    def get_imbalance(self, class_index: int) -> float:
        return self.total_imbalance


class Imbalancer:
    def __init__(self, dataset, imbalancedness):
        self.dataset = dataset
        self.imbalancedness = imbalancedness

    def _create_indices(self) -> list[int]:
        """
        Create imbalanced dataset indices using GPU acceleration.
        """
        print("Creating dataset indices on GPU...")

        # Load labels to a GPU tensor
        labels = torch.tensor(self.dataset["label"], device="cuda")

        # Get imbalance probabilities for all labels
        imbalance_probs = self.imbalancedness.get_imbalance(labels)

        # Generate random numbers for each sample
        random_numbers = torch.rand(len(labels), device="cuda")

        # Create mask to filter indices
        mask = imbalance_probs >= random_numbers
        selected_indices = torch.nonzero(mask, as_tuple=True)[0]
        non_selected_indices = torch.nonzero(~mask, as_tuple=True)[0]

        # Convert to CPU and Python list
        selected_indices_list = selected_indices.cpu().tolist()
        non_selected_indices_list = non_selected_indices.cpu().tolist()

        remaining = len(selected_indices_list) / labels.size(0)
        print("\n--- Portion of dataset remaining: ", remaining, "\n ---")

        return selected_indices_list, non_selected_indices_list

    def get_filtered_dataset(self):
        indices, remaining = self._create_indices()
        return self.dataset.select(indices), self.dataset.select(remaining)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="clane9/imagenet-100")
    parser.add_argument("--pareto-alpha", type=float, default=2.5)
    parser.add_argument("--max-samples", type=int, default=1280)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--dataset-name", type=str, default="imagenet-100")
    args = parser.parse_args()

    load_dotenv()
    login(token=os.environ["HUGGINGFACE_TOKEN"])

    dataset = load_dataset(args.dataset, trust_remote_code=True)
    all_splits = [dataset[split] for split in dataset.keys()]
    combined_dataset = concatenate_datasets(all_splits)

    num_classes = len(combined_dataset.features["label"].names)

    power_law_imbalance = PowerLawImbalancedness(
        num_classes, args.pareto_alpha, args.max_samples, args.min_samples
    )

    power_law_imbalancer = Imbalancer(combined_dataset, power_law_imbalance)
    imbalanced_dataset, removed_dataset = power_law_imbalancer.get_filtered_dataset()

    imbalanced_dataset.push_to_hub("flaitenberger/imagenet-100-LT")
    removed_dataset.push_to_hub("flaitenberger/imagenet-100-LT-removed-data")

    balanced_imbalance = NoImbalancedness(
        num_classes, args.pareto_alpha, args.max_samples, args.min_samples
    )

    balanced_imbalancer = Imbalancer(combined_dataset, balanced_imbalance)

    balanced_dataset, removed_dataset = balanced_imbalancer.get_filtered_dataset()

    balanced_dataset.push_to_hub("flaitenberger/imagenet-100-LT-balanced")


if __name__ == "__main__":
    main()
