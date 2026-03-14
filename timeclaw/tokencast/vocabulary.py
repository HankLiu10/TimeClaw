"""Time series vocabulary management for TokenCast.

Manages the mapping between discrete token IDs and their semantic
meaning (bin centers / special tokens), supporting multi-granularity
vocabularies for coarse-to-fine encoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# Reserved special token IDs (always occupy the first slots)
PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3
MASK_TOKEN = 4
NUM_SPECIAL_TOKENS = 5

SPECIAL_TOKEN_NAMES = {
    PAD_TOKEN: "<PAD>",
    BOS_TOKEN: "<BOS>",
    EOS_TOKEN: "<EOS>",
    UNK_TOKEN: "<UNK>",
    MASK_TOKEN: "<MASK>",
}


@dataclass
class TimeSeriesVocabulary:
    """Vocabulary that maps token IDs to continuous value ranges.

    Token layout: [special tokens (0..4)] + [value bins (5..5+n_bins-1)]
    """

    n_bins: int = 1024
    bin_edges: np.ndarray | None = None
    bin_centers: np.ndarray | None = None
    _name: str = "default"

    @property
    def vocab_size(self) -> int:
        return NUM_SPECIAL_TOKENS + self.n_bins

    @property
    def value_offset(self) -> int:
        """First token ID that represents a real value (not special)."""
        return NUM_SPECIAL_TOKENS

    def build_from_edges(self, edges: np.ndarray) -> TimeSeriesVocabulary:
        """Build vocabulary from pre-computed bin edges."""
        self.bin_edges = np.asarray(edges, dtype=float)
        self.n_bins = len(self.bin_edges) - 1
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0
        return self

    def value_to_token(self, values: np.ndarray) -> np.ndarray:
        """Map continuous values to token IDs (offset by special tokens)."""
        if self.bin_edges is None:
            raise RuntimeError("Vocabulary not built")
        indices = np.digitize(values, self.bin_edges[1:-1])
        indices = np.clip(indices, 0, self.n_bins - 1)
        return (indices + self.value_offset).astype(np.int32)

    def token_to_value(self, tokens: np.ndarray) -> np.ndarray:
        """Map token IDs back to continuous values (bin centers)."""
        if self.bin_centers is None:
            raise RuntimeError("Vocabulary not built")
        tokens = np.asarray(tokens, dtype=np.int32)
        bin_indices = tokens - self.value_offset
        # Special tokens map to NaN
        mask = bin_indices < 0
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        values = self.bin_centers[bin_indices]
        values[mask] = np.nan
        return values

    def is_special(self, token_id: int) -> bool:
        return 0 <= token_id < NUM_SPECIAL_TOKENS

    def token_name(self, token_id: int) -> str:
        if token_id in SPECIAL_TOKEN_NAMES:
            return SPECIAL_TOKEN_NAMES[token_id]
        bin_idx = token_id - self.value_offset
        if self.bin_centers is not None and 0 <= bin_idx < len(self.bin_centers):
            return f"v{self.bin_centers[bin_idx]:.4f}"
        return f"t{token_id}"


@dataclass
class MultiGranularityVocabulary:
    """Manages coarse + fine vocabularies for hierarchical encoding.

    Coarse vocabulary captures broad value ranges (fewer bins),
    fine vocabulary captures precise values (more bins).
    """

    coarse: TimeSeriesVocabulary = field(default_factory=lambda: TimeSeriesVocabulary(n_bins=64))
    fine: TimeSeriesVocabulary = field(default_factory=lambda: TimeSeriesVocabulary(n_bins=1024))

    def build_from_data(
        self,
        data: np.ndarray,
        coarse_bins: int = 64,
        fine_bins: int = 1024,
    ) -> MultiGranularityVocabulary:
        """Build both vocabularies from data using uniform binning."""
        flat = np.asarray(data, dtype=float).ravel()
        flat = flat[np.isfinite(flat)]

        coarse_edges = np.linspace(flat.min(), flat.max(), coarse_bins + 1)
        fine_edges = np.linspace(flat.min(), flat.max(), fine_bins + 1)

        self.coarse = TimeSeriesVocabulary(n_bins=coarse_bins, _name="coarse")
        self.coarse.build_from_edges(coarse_edges)

        self.fine = TimeSeriesVocabulary(n_bins=fine_bins, _name="fine")
        self.fine.build_from_edges(fine_edges)

        return self

    def encode_coarse(self, values: np.ndarray) -> np.ndarray:
        return self.coarse.value_to_token(values)

    def encode_fine(self, values: np.ndarray) -> np.ndarray:
        return self.fine.value_to_token(values)

    def decode_coarse(self, tokens: np.ndarray) -> np.ndarray:
        return self.coarse.token_to_value(tokens)

    def decode_fine(self, tokens: np.ndarray) -> np.ndarray:
        return self.fine.token_to_value(tokens)
