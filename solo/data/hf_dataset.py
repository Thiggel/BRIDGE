from typing import Callable, Optional, Union
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Dataset as HFDataset


class HuggingFaceDatasetWrapper(Dataset):
    """Wrapper for HuggingFace datasets to be used with solo-learn."""

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        streaming: bool = False,
        image_key: str = "image",
        label_key: str = "label",
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        **dataset_kwargs
    ):
        """
        Args:
            dataset_name: name or path of the huggingface dataset
            split: dataset split to use (train, validation, test)
            transform: image transformations
            streaming: whether to use streaming mode
            image_key: key for the image in the dataset
            label_key: key for the label in the dataset
            cache_dir: directory to cache the dataset
            token: HuggingFace API token for private datasets
            dataset_kwargs: additional kwargs for load_dataset
        """
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key

        # Load dataset
        self.dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=streaming,
            cache_dir=cache_dir,
            token=token,
            **dataset_kwargs
        )

        if streaming:
            # Buffer some samples to determine features
            self._buffer = []
            self._buffer_size = 100
            self._buffer_idx = 0
            self._iter = iter(self.dataset)
            self._fill_buffer()

    def _fill_buffer(self):
        """Fill buffer with samples in streaming mode."""
        if not hasattr(self, "_iter"):
            return

        while len(self._buffer) < self._buffer_size:
            try:
                self._buffer.append(next(self._iter))
            except StopIteration:
                self._iter = iter(self.dataset)  # Restart iterator
                break

    def __len__(self):
        if hasattr(self.dataset, "__len__"):
            return len(self.dataset)
        else:
            # For streaming datasets, we don't know the length
            return int(1e9)  # Return a large number

    def __getitem__(self, idx):
        if hasattr(self.dataset, "__getitem__"):
            # Non-streaming dataset
            item = self.dataset[idx]
        else:
            # Streaming dataset - use buffer
            if idx >= len(self._buffer):
                # We need more samples
                self._fill_buffer()

            # Get from buffer, cycling if needed
            buffer_idx = idx % len(self._buffer)
            item = self._buffer[buffer_idx]

        # Get image
        img = item[self.image_key]

        # Apply transform
        if self.transform is not None:
            img = self.transform(img)

        # Get label
        label = item.get(self.label_key, -1)
        if isinstance(label, (list, tuple)) and len(label) == 1:
            label = label[0]

        return img, label
