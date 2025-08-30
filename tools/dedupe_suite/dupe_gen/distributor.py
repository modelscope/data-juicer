# dupgen/distributors.py
import random
from typing import Dict, List, Tuple

import numpy as np


class DuplicateDistributor:
    """Controls how duplicates are distributed in the dataset"""

    @staticmethod
    def random_distribution(originals: List[Dict],
                            num_duplicates: int) -> List[Tuple[Dict, int]]:
        """Distribute duplicates randomly across originals

        Args:
            originals: List of original documents
            num_duplicates: Number of duplicates to create

        Returns:
            List of (original, count) tuples indicating how many duplicates
            to create
        """
        distribution = []

        # Simple random selection with replacement
        for _ in range(num_duplicates):
            original = random.choice(originals)
            distribution.append((original, 1))

        return distribution

    @staticmethod
    def clustered_distribution(
            originals: List[Dict],
            num_duplicates: int,
            avg_cluster_size: float = 5.0,
            variance: float = 0.5) -> List[Tuple[Dict, int]]:
        """Distribute duplicates in clusters

        Args:
            originals: List of original documents
            num_duplicates: Number of duplicates to create
            avg_cluster_size: Average number of duplicates per original
            variance: Variance in cluster sizes (0-1, higher means more
            variance)

        Returns:
            List of (original, count) tuples indicating how many duplicates
              to create
        """
        # Determine how many originals to use
        num_clusters = max(1, int(num_duplicates / avg_cluster_size))

        # Select originals to duplicate
        selected_originals = random.sample(originals,
                                           k=min(num_clusters, len(originals)))

        # Generate cluster sizes following a power law distribution
        alpha = 2.0  # Power law exponent (adjust for different distributions)
        sizes = np.random.power(
            alpha, size=len(selected_originals)) * avg_cluster_size * 2
        sizes = np.maximum(sizes, 1)  # Ensure at least size 1

        # Apply variance
        if variance > 0:
            # Add noise proportional to variance
            noise = np.random.normal(0,
                                     variance * avg_cluster_size,
                                     size=len(sizes))
            sizes = np.maximum(sizes + noise, 1)

        # Convert to integers
        sizes = sizes.astype(int)

        # Adjust to match total required duplicates
        total = sum(sizes)
        if total > num_duplicates:
            # Scale down
            sizes = np.floor(sizes * (num_duplicates / total)).astype(int)
            # Distribute remaining
            remainder = num_duplicates - sum(sizes)
            for i in range(remainder):
                sizes[i % len(sizes)] += 1
        elif total < num_duplicates:
            # Scale up
            deficit = num_duplicates - total
            # Distribute deficit
            for i in range(deficit):
                sizes[i % len(sizes)] += 1

        # Create distribution
        distribution = [(original, int(size))
                        for original, size in zip(selected_originals, sizes)]

        return distribution

    @staticmethod
    def power_law_distribution(
            originals: List[Dict],
            num_duplicates: int,
            exponent: float = 2.0) -> List[Tuple[Dict, int]]:
        """Distribute duplicates following a power law (few originals get
        many duplicates)

        Args:
            originals: List of original documents
            num_duplicates: Number of duplicates to create
            exponent: Power law exponent (higher means more skewed
              distribution)

        Returns:
            List of (original, count) tuples indicating how many duplicates
            to create
        """
        # Select a subset of originals to duplicate
        num_to_duplicate = min(len(originals), max(1,
                                                   int(len(originals) * 0.1)))
        selected_originals = random.sample(originals, k=num_to_duplicate)

        # Generate power law weights
        weights = np.power(np.arange(1, num_to_duplicate + 1), -exponent)
        weights = weights / np.sum(weights)  # Normalize

        # Distribute duplicates according to weights
        counts = np.zeros(num_to_duplicate, dtype=int)
        for _ in range(num_duplicates):
            idx = np.random.choice(num_to_duplicate, p=weights)
            counts[idx] += 1

        # Create distribution
        distribution = [(original, int(count))
                        for original, count in zip(selected_originals, counts)]

        return distribution
