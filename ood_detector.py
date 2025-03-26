import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from umap import UMAP
from matplotlib import pyplot as plt
import wandb

# 2. get ood datapoints
#   2.1. get embeddings for all datapoints
#   2.2. fit GMM with num_clusters
#   2.3. for each cluster, get num_ood_samples_per_cluster samples based on the lowest prob
#   2.4. plot UMAP with all embeddings, their cluster assignments, and ood datapoints
#   2.5. make same UMAP plot but with class assignment instead of clusters
#   2.6 upload to wandb
#   2.4. return ood datapoint indices, embeddings and their classes


class OODDetector:
    def __init__(
        self,
        feature_extractor: nn.Module,
        num_clusters: int,
        num_ood_samples_per_cluster: int,
        batch_size: int,
        num_workers: int,
        experiment_name: str,
        cycle_idx: int,
    ):
        self.feature_extractor = feature_extractor
        self.num_clusters = num_clusters
        self.num_ood_samples_per_cluster = num_ood_samples_per_cluster
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.umap_filename = f"{os.environ.get('BASE_CACHE_DIR')}/umap/{experiment_name}/ood_umap_{cycle_idx}.png"
        self.cycle_idx = cycle_idx

    def _extract_features(self, dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        features, indices, targets = [], [], []

        self.feature_extractor.eval()
        for batch in tqdm(dataloader, desc="Extracting features"):
            images, labels = batch
            batch_indices = torch.arange(len(indices), len(indices) + len(images))

            with torch.no_grad():
                if torch.cuda.is_available():
                    images = images.cuda()
                feat = self.feature_extractor(images).cpu()

            features.append(feat)
            indices.append(batch_indices)
            targets.append(labels)

        features = torch.cat(features, dim=0)
        indices = torch.cat(indices, dim=0)
        targets = torch.cat(targets, dim=0)

        return features, indices, targets

    def _fit_gmm(self, features):
        gmm = GaussianMixture(n_components=self.num_clusters)
        gmm.fit(features)

        return gmm

    def _get_ood_indices(self, gmm, features):
        probs = gmm.predict_proba(features.numpy())

        ood_indices = []
        for cluster_idx in range(self.num_clusters):
            cluster_mask = gmm.predict(features.numpy()) == cluster_idx
            cluster_indices = np.where(cluster_mask)[0]
            cluster_probs = probs[cluster_mask, cluster_idx]

            # Get lowest probability samples within this cluster
            if len(cluster_indices) > self.num_ood_samples_per_cluster:
                sorted_idx = np.argsort(cluster_probs)[
                    : self.num_ood_samples_per_cluster
                ]
                ood_indices.extend(cluster_indices[sorted_idx])

        return np.array(ood_indices)

    def _plot_umap(self, features, targets, ood_indices, cluster_assignments):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.umap_filename), exist_ok=True)

        # Convert to numpy for UMAP
        features_np = features.numpy()

        # Fit UMAP
        reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        umap_features = reducer.fit_transform(features_np)

        # Create mask for OOD points
        ood_mask = np.zeros(len(features), dtype=bool)
        ood_mask[ood_indices] = True

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Plot 1: Cluster assignments
        for i in range(self.num_clusters):
            mask = (cluster_assignments == i) & ~ood_mask
            ax1.scatter(
                umap_features[mask, 0],
                umap_features[mask, 1],
                s=5,
                alpha=0.5,
                label=f"Cluster {i}",
            )

        # Plot OOD points
        ax1.scatter(
            umap_features[ood_mask, 0],
            umap_features[ood_mask, 1],
            c="red",
            s=20,
            marker="x",
            label="OOD",
        )
        ax1.set_title("Cluster assignments with OOD points")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Plot 2: Class assignments
        num_classes = len(torch.unique(targets))
        for i in range(num_classes):
            mask = (targets == i) & ~ood_mask
            ax2.scatter(
                umap_features[mask, 0],
                umap_features[mask, 1],
                s=5,
                alpha=0.5,
                label=f"Class {i}",
            )

        # Plot OOD points
        ax2.scatter(
            umap_features[ood_mask, 0],
            umap_features[ood_mask, 1],
            c="red",
            s=20,
            marker="x",
            label="OOD",
        )
        ax2.set_title("Class assignments with OOD points")
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig(self.umap_filename, dpi=300)
        plt.close()

        return umap_features

    def get_ood_datapoints(self, dataset):
        # Get features and metadata
        features, indices, targets = self._extract_features(dataset)

        # Normalize features for GMM
        features_norm = F.normalize(features, dim=1)

        # Fit GMM on normalized features
        gmm = self._fit_gmm(features_norm)

        # Get cluster assignments
        cluster_assignments = gmm.predict(features_norm.numpy())

        # Get OOD indices
        ood_indices = self._get_ood_indices(gmm, features_norm)

        # Plot and log
        umap_features = self._plot_umap(
            features, targets, ood_indices, cluster_assignments
        )

        if wandb.run is not None:
            wandb.log(
                {
                    f"ood_umap_cycle_{self.cycle_idx}": wandb.Image(self.umap_filename),
                    f"num_ood_points_cycle_{self.cycle_idx}": len(ood_indices),
                }
            )

        # Return OOD information
        return indices[ood_indices], features[ood_indices], targets[ood_indices]
