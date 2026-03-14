"""Symbolic discretization engine for TokenCast.

Provides multiple quantization strategies to map continuous time-series
values into discrete symbol IDs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseQuantizer(ABC):
    """Abstract quantizer: fit on data, then quantize/dequantize."""

    def __init__(self, n_bins: int = 1024):
        self.n_bins = n_bins
        self._fitted = False
        self._bin_edges: np.ndarray | None = None

    @abstractmethod
    def fit(self, data: np.ndarray) -> BaseQuantizer:
        ...

    def quantize(self, values: np.ndarray) -> np.ndarray:
        """Map continuous values to bin indices in [0, n_bins)."""
        if self._bin_edges is None:
            raise RuntimeError("Quantizer not fitted")
        indices = np.digitize(values, self._bin_edges[1:-1])
        return np.clip(indices, 0, self.n_bins - 1).astype(np.int32)

    def dequantize(self, indices: np.ndarray) -> np.ndarray:
        """Map bin indices back to bin-center values."""
        if self._bin_edges is None:
            raise RuntimeError("Quantizer not fitted")
        centers = (self._bin_edges[:-1] + self._bin_edges[1:]) / 2.0
        return centers[np.clip(indices, 0, len(centers) - 1)]

    @property
    def bin_edges(self) -> np.ndarray | None:
        return self._bin_edges

    @property
    def bin_centers(self) -> np.ndarray | None:
        if self._bin_edges is None:
            return None
        return (self._bin_edges[:-1] + self._bin_edges[1:]) / 2.0


class UniformQuantizer(BaseQuantizer):
    """Equal-width binning."""

    def fit(self, data: np.ndarray) -> UniformQuantizer:
        flat = np.asarray(data, dtype=float).ravel()
        flat = flat[np.isfinite(flat)]
        if len(flat) == 0:
            raise ValueError("No finite values to fit")
        self._bin_edges = np.linspace(flat.min(), flat.max(), self.n_bins + 1)
        self._fitted = True
        return self


class QuantileQuantizer(BaseQuantizer):
    """Equal-frequency (quantile) binning."""

    def fit(self, data: np.ndarray) -> QuantileQuantizer:
        flat = np.asarray(data, dtype=float).ravel()
        flat = flat[np.isfinite(flat)]
        if len(flat) == 0:
            raise ValueError("No finite values to fit")
        percentiles = np.linspace(0, 100, self.n_bins + 1)
        edges = np.percentile(flat, percentiles)
        # Ensure strictly increasing edges by adding tiny offsets for duplicates
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-10
        self._bin_edges = edges
        self._fitted = True
        return self


class KMeansQuantizer(BaseQuantizer):
    """K-means clustering quantization.

    Clusters data values into n_bins groups, then derives bin edges
    from the midpoints between sorted cluster centers.
    """

    def __init__(self, n_bins: int = 1024, max_iter: int = 50, sample_size: int = 100_000):
        super().__init__(n_bins)
        self.max_iter = max_iter
        self.sample_size = sample_size

    def fit(self, data: np.ndarray) -> KMeansQuantizer:
        flat = np.asarray(data, dtype=float).ravel()
        flat = flat[np.isfinite(flat)]
        if len(flat) == 0:
            raise ValueError("No finite values to fit")

        if len(flat) > self.sample_size:
            rng = np.random.default_rng(42)
            flat = rng.choice(flat, self.sample_size, replace=False)

        n_clusters = min(self.n_bins, len(np.unique(flat)))
        if n_clusters < self.n_bins:
            return UniformQuantizer(self.n_bins).fit(data)

        # Mini-batch 1D K-means via sorted initialization
        centers = np.percentile(flat, np.linspace(0, 100, n_clusters))

        for _ in range(self.max_iter):
            # Assign each point to nearest center
            dists = np.abs(flat[:, None] - centers[None, :])
            labels = np.argmin(dists, axis=1)
            new_centers = np.array([
                flat[labels == k].mean() if np.any(labels == k) else centers[k]
                for k in range(n_clusters)
            ])
            if np.allclose(centers, new_centers, atol=1e-8):
                break
            centers = new_centers

        centers = np.sort(centers)
        # Derive bin edges from midpoints between centers
        midpoints = (centers[:-1] + centers[1:]) / 2.0
        self._bin_edges = np.concatenate([
            [flat.min() - 1e-10],
            midpoints,
            [flat.max() + 1e-10],
        ])
        self._fitted = True
        return self


class AdaptiveQuantizer(BaseQuantizer):
    """Adaptive binning based on data density.

    Uses a histogram-based approach: regions with higher density get
    finer bins, regions with lower density get coarser bins.
    """

    def __init__(self, n_bins: int = 1024, density_bins: int = 256):
        super().__init__(n_bins)
        self.density_bins = density_bins

    def fit(self, data: np.ndarray) -> AdaptiveQuantizer:
        flat = np.asarray(data, dtype=float).ravel()
        flat = flat[np.isfinite(flat)]
        if len(flat) == 0:
            raise ValueError("No finite values to fit")

        # Step 1: estimate density via histogram
        counts, coarse_edges = np.histogram(flat, bins=self.density_bins)
        density = counts / counts.sum()

        # Step 2: allocate bins proportional to sqrt(density)
        # sqrt gives a balance between uniform and fully density-proportional
        weights = np.sqrt(density + 1e-12)
        weights /= weights.sum()
        bin_alloc = np.round(weights * self.n_bins).astype(int)
        bin_alloc = np.maximum(bin_alloc, 1)

        # Adjust to exactly n_bins
        while bin_alloc.sum() != self.n_bins:
            diff = self.n_bins - bin_alloc.sum()
            if diff > 0:
                indices = np.argsort(-weights)[:diff]
                bin_alloc[indices] += 1
            else:
                candidates = np.where(bin_alloc > 1)[0]
                if len(candidates) == 0:
                    break
                sorted_cand = candidates[np.argsort(weights[candidates])]
                for idx in sorted_cand[:abs(diff)]:
                    bin_alloc[idx] -= 1

        # Step 3: build fine edges within each coarse bin
        all_edges = [coarse_edges[0]]
        for i in range(self.density_bins):
            lo, hi = coarse_edges[i], coarse_edges[i + 1]
            n_sub = int(bin_alloc[i])
            sub_edges = np.linspace(lo, hi, n_sub + 1)
            all_edges.extend(sub_edges[1:].tolist())

        self._bin_edges = np.array(all_edges)
        self.n_bins = len(self._bin_edges) - 1
        self._fitted = True
        return self


def get_quantizer(strategy: str = "uniform", n_bins: int = 1024, **kwargs: Any) -> BaseQuantizer:
    """Factory function to create a quantizer by strategy name."""
    strategies = {
        "uniform": UniformQuantizer,
        "quantile": QuantileQuantizer,
        "kmeans": KMeansQuantizer,
        "adaptive": AdaptiveQuantizer,
    }
    cls = strategies.get(strategy)
    if cls is None:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {list(strategies.keys())}")
    return cls(n_bins=n_bins, **kwargs)
