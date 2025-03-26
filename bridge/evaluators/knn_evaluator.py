import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


class KNNEvaluator:
    """Evaluates feature quality using kNN classification"""

    def __init__(
        self,
        feature_extractor: nn.Module,
        k_values: list = [1, 5, 10, 20, 50, 100],
        batch_size: int = 256,
        num_workers: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize KNN evaluator

        Args:
            feature_extractor: Model that extracts features from images
            k_values: List of k values to evaluate
            batch_size: Batch size for feature extraction
            num_workers: Number of workers for data loading
            device: Device to use for inference
        """
        self.feature_extractor = feature_extractor
        self.k_values = k_values
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

    def extract_features(self, dataloader):
        """Extract features from dataloader"""
        features = []
        targets = []

        # Set model to evaluation mode
        self.feature_extractor.eval()

        # Extract features
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                # Get images and targets
                images, batch_targets = batch
                images = images.to(self.device)

                # Extract features
                batch_features = self.feature_extractor(images)

                # Move features and targets to CPU
                batch_features = batch_features.cpu()

                # Append to lists
                features.append(batch_features)
                targets.append(batch_targets)

        # Concatenate features and targets
        features = torch.cat(features, dim=0)
        targets = torch.cat(targets, dim=0)

        return features, targets

    def evaluate(self, train_dataloader, test_dataloader):
        """
        Evaluate features using kNN classification

        Args:
            train_dataloader: DataLoader for training data
            test_dataloader: DataLoader for test data

        Returns:
            Dictionary with accuracy for each k value
        """
        # Extract features
        print("Extracting features from training set...")
        train_features, train_targets = self.extract_features(train_dataloader)

        print("Extracting features from test set...")
        test_features, test_targets = self.extract_features(test_dataloader)

        # Convert to numpy
        train_features_np = train_features.numpy()
        train_targets_np = train_targets.numpy()
        test_features_np = test_features.numpy()
        test_targets_np = test_targets.numpy()

        # Normalize features (important for kNN with cosine distance)
        train_features_np = train_features_np / np.linalg.norm(
            train_features_np, axis=1, keepdims=True
        )
        test_features_np = test_features_np / np.linalg.norm(
            test_features_np, axis=1, keepdims=True
        )

        # Evaluate for each k value
        results = {}
        for k in self.k_values:
            print(f"Evaluating with k={k}...")
            # Create kNN classifier
            knn = KNeighborsClassifier(
                n_neighbors=k,
                algorithm="auto",
                metric="cosine",
                n_jobs=-1,  # Use all available cores
            )

            # Fit kNN classifier
            knn.fit(train_features_np, train_targets_np)

            # Predict
            test_preds = knn.predict(test_features_np)

            # Calculate accuracy
            accuracy = np.mean(test_preds == test_targets_np)

            # Calculate confusion matrix
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(test_targets_np, test_preds)

            # Store results
            results[k] = {"accuracy": accuracy, "confusion_matrix": cm}

            print(f"k={k}, Accuracy: {accuracy:.4f}")

        return results
