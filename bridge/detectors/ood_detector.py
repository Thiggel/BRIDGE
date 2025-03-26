import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture
from umap import UMAP
import matplotlib.pyplot as plt
from tqdm import tqdm
from fastai.vision.all import *
import wandb


class OODDetector:
    """Detector for out-of-distribution samples in the latent space using GMM clustering"""

    def __init__(
        self,
        feature_extractor: nn.Module,
        num_clusters: int,
        num_ood_samples_per_cluster: int,
        batch_size: int,
        num_workers: int,
        experiment_name: str,
        cycle_idx: int,
        output_dir: str = "/tmp",
    ):
        self.feature_extractor = feature_extractor
        self.num_clusters = num_clusters
        self.num_ood_samples_per_cluster = num_ood_samples_per_cluster
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.experiment_name = experiment_name
        self.cycle_idx = cycle_idx
        self.output_dir = output_dir

        # Create output directory for plots
        self.plot_dir = os.path.join(output_dir, "plots", experiment_name)
        os.makedirs(self.plot_dir, exist_ok=True)

        # Set plot filename
        self.umap_filename = os.path.join(
            self.plot_dir, f"ood_umap_cycle_{cycle_idx}.png"
        )

    def _extract_features(self, dataset):
        """Extract features from dataset using the feature extractor"""
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
                feat = self.feature_extractor.get_features(images).cpu()

            features.append(feat)
            indices.append(batch_indices)
            targets.append(labels)

        features = torch.cat(features, dim=0)
        indices = torch.cat(indices, dim=0)
        targets = torch.cat(targets, dim=0)

        return features, indices, targets

    def _fit_gmm(self, features):
        """Fit Gaussian Mixture Model to features"""
        # Normalize features for GMM (important for high-dimensional data)
        features_norm = F.normalize(features, dim=1)

        # Fit GMM
        gmm = GaussianMixture(
            n_components=self.num_clusters,
            random_state=42,
            verbose=1,
            verbose_interval=10,
        )
        gmm.fit(features_norm.numpy())

        return gmm

    def _get_ood_indices(self, gmm, features):
        """Get indices of OOD samples for each cluster based on lowest probability"""
        # Normalize features for consistency with GMM fitting
        features_norm = F.normalize(features, dim=1)

        # Get cluster assignments and probabilities
        cluster_assignments = gmm.predict(features_norm.numpy())
        probs = gmm.predict_proba(features_norm.numpy())

        # Find OOD samples for each cluster
        ood_indices = []
        for cluster_idx in range(self.num_clusters):
            # Get samples belonging to this cluster
            cluster_mask = cluster_assignments == cluster_idx
            cluster_indices = np.where(cluster_mask)[0]

            # Get probabilities of these samples belonging to this cluster
            cluster_probs = probs[cluster_mask, cluster_idx]

            # Get the samples with lowest probability (edge cases / underrepresented)
            if len(cluster_indices) > self.num_ood_samples_per_cluster:
                # Sort by probability and get the lowest probability samples
                sorted_idx = np.argsort(cluster_probs)[
                    : self.num_ood_samples_per_cluster
                ]
                ood_indices.extend(cluster_indices[sorted_idx])
            else:
                # If cluster is smaller than requested samples, take all
                ood_indices.extend(cluster_indices)

        return np.array(ood_indices), cluster_assignments

    def _plot_umap(self, features, targets, ood_indices, cluster_assignments):
        """Create UMAP visualization of latent space"""
        # Convert features to numpy for UMAP
        features_np = features.numpy()

        # Fit UMAP
        reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        umap_features = reducer.fit_transform(features_np)

        # Create mask for OOD points
        ood_mask = np.zeros(len(features), dtype=bool)
        ood_mask[ood_indices] = True

        # Create plot
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
        unique_classes = np.unique(targets.numpy())
        num_classes = len(unique_classes)

        # Use a colormap suitable for many classes
        cmap = plt.get_cmap("tab20" if num_classes <= 20 else "gist_rainbow")

        for i, class_idx in enumerate(unique_classes):
            mask = (targets.numpy() == class_idx) & ~ood_mask
            ax2.scatter(
                umap_features[mask, 0],
                umap_features[mask, 1],
                s=5,
                alpha=0.5,
                color=cmap(i / num_classes),
                label=f"Class {class_idx}" if i < 10 else None,  # Limit legend entries
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
        if num_classes <= 10:
            ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            ax2.legend(
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                title="First 10 classes + OOD",
            )

        plt.tight_layout()
        plt.savefig(self.umap_filename, dpi=300)
        plt.close()

        return umap_features

    def get_ood_datapoints(self, dataset):
        """Get OOD samples from dataset"""
        # Extract features
        features, indices, labels = self._extract_features(dataset)

        # Fit GMM
        gmm = self._fit_gmm(features)

        # Get OOD indices and cluster assignments
        ood_indices, cluster_assignments = self._get_ood_indices(gmm, features)

        # Plot UMAP
        umap_features = self._plot_umap(
            features, labels, ood_indices, cluster_assignments
        )

        # Log to wandb if available
        if wandb.run is not None:
            wandb.log(
                {
                    f"ood_umap_cycle_{self.cycle_idx}": wandb.Image(self.umap_filename),
                    f"num_ood_points_cycle_{self.cycle_idx}": len(ood_indices),
                }
            )

        # Return OOD samples
        return indices[ood_indices], features[ood_indices], labels[ood_indices]

    def get_random_samples(self, dataset, num_samples):
        """Get random samples for ablation study"""
        # Extract features
        features, indices, labels = self._extract_features(dataset)

        # Randomly select indices
        np.random.seed(
            42 + self.cycle_idx
        )  # Ensure reproducibility with different seed per cycle
        random_indices = np.random.choice(
            len(indices), size=min(num_samples, len(indices)), replace=False
        )

        # Return random samples
        return indices[random_indices], features[random_indices], labels[random_indices]
