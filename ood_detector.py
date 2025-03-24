import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class OODDetector:
    def __init__(
        self,
        feature_extractor: nn.Module,
        num_clusters: int,
        num_ood_points_per_cluster: int,
        batch_size: int,
        num_workers: int,
        device: torch.device,
    ):
        self.feature_extractor = feature_extractor
        self.num_clusters = num_clusters
        self.num_ood_points_per_cluster = num_ood_points_per_cluster
        self.device = device

    def extract_features(self, dataset: Dataset) -> torch.Tensor:
        dataloader = DataLoader(dataset, batch_size=32, num_workers=4)

        features = []
        for batch in dataloader:
            features.append(self.feature_extractor(batch.to(self.device)))

        del dataloader

        return torch.cat(features)

    def detect(self, dataset: Dataset) -> torch.Tensor:
        return torch.rand(len(dataset), self.num_clusters)
