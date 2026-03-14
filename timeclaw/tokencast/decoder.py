"""TokenCast decoder: semantic token -> signal reconstruction.

Reconstructs continuous signals from discrete token IDs, with optional
interpolation smoothing for higher-quality output.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from timeclaw.tokencast.vocabulary import (
    NUM_SPECIAL_TOKENS,
    TimeSeriesVocabulary,
    MultiGranularityVocabulary,
)


class TokenCastDecoder:
    """Decodes semantic tokens back to continuous signals.

    Supports:
    - Single-vocabulary decoding
    - Multi-granularity fusion (coarse + fine)
    - Channel-aware reconstruction
    - Optional interpolation smoothing
    """

    def __init__(
        self,
        vocabulary: TimeSeriesVocabulary | None = None,
        multi_vocab: MultiGranularityVocabulary | None = None,
        channel_vocabs: list[TimeSeriesVocabulary] | None = None,
        smooth: bool = False,
        smooth_window: int = 3,
    ):
        self.vocabulary = vocabulary
        self.multi_vocab = multi_vocab
        self.channel_vocabs = channel_vocabs or []
        self.smooth = smooth
        self.smooth_window = smooth_window

    def decode(
        self,
        tokens: np.ndarray,
        strip_special: bool = True,
    ) -> np.ndarray:
        """Decode token IDs to continuous values."""
        if self.vocabulary is None:
            raise RuntimeError("Decoder has no vocabulary set")

        tokens = np.asarray(tokens, dtype=np.int32)
        if strip_special:
            tokens = self._strip_special(tokens)

        values = self.vocabulary.token_to_value(tokens)

        if self.smooth:
            values = self._smooth_signal(values)

        return values

    def decode_multi_granularity(
        self,
        coarse_tokens: np.ndarray,
        fine_tokens: np.ndarray,
        alpha: float = 0.8,
        strip_special: bool = True,
    ) -> np.ndarray:
        """Decode using both coarse and fine tokens with weighted fusion.

        Args:
            coarse_tokens: Coarse-grained token IDs.
            fine_tokens: Fine-grained token IDs.
            alpha: Weight for fine tokens (1-alpha for coarse).
            strip_special: Whether to strip BOS/EOS tokens.
        """
        if self.multi_vocab is None:
            raise RuntimeError("Decoder has no multi-granularity vocabulary")

        coarse_tokens = np.asarray(coarse_tokens, dtype=np.int32)
        fine_tokens = np.asarray(fine_tokens, dtype=np.int32)

        if strip_special:
            coarse_tokens = self._strip_special(coarse_tokens)
            fine_tokens = self._strip_special(fine_tokens)

        coarse_vals = self.multi_vocab.decode_coarse(coarse_tokens)
        fine_vals = self.multi_vocab.decode_fine(fine_tokens)

        # Weighted fusion: prefer fine but use coarse as anchor
        fused = alpha * fine_vals + (1 - alpha) * coarse_vals

        if self.smooth:
            fused = self._smooth_signal(fused)

        return fused

    def decode_channels(
        self,
        channel_tokens: list[np.ndarray],
        strip_special: bool = True,
    ) -> np.ndarray:
        """Decode per-channel tokens back to multivariate signal."""
        if not self.channel_vocabs:
            raise RuntimeError("Decoder has no channel vocabularies")
        if len(channel_tokens) != len(self.channel_vocabs):
            raise ValueError(
                f"Got {len(channel_tokens)} channels but {len(self.channel_vocabs)} vocabs"
            )

        channels = []
        for ch_tokens, vocab in zip(channel_tokens, self.channel_vocabs):
            ch_tokens = np.asarray(ch_tokens, dtype=np.int32)
            if strip_special:
                ch_tokens = self._strip_special(ch_tokens)
            channels.append(vocab.token_to_value(ch_tokens))

        result = np.column_stack(channels)
        if self.smooth:
            for col in range(result.shape[1]):
                result[:, col] = self._smooth_signal(result[:, col])

        return result

    @staticmethod
    def _strip_special(tokens: np.ndarray) -> np.ndarray:
        """Remove leading/trailing special tokens (IDs < NUM_SPECIAL_TOKENS)."""
        if tokens.ndim == 1:
            mask = tokens >= NUM_SPECIAL_TOKENS
            return tokens[mask]
        # 2D: strip rows where all values are special
        row_mask = np.any(tokens >= NUM_SPECIAL_TOKENS, axis=-1) if tokens.ndim > 1 else tokens >= NUM_SPECIAL_TOKENS
        return tokens[row_mask]

    def _smooth_signal(self, values: np.ndarray) -> np.ndarray:
        """Apply simple moving-average smoothing."""
        if values.ndim != 1 or len(values) < self.smooth_window:
            return values
        kernel = np.ones(self.smooth_window) / self.smooth_window
        # Pad to preserve length
        padded = np.pad(values, (self.smooth_window // 2, self.smooth_window // 2), mode="edge")
        return np.convolve(padded, kernel, mode="valid")[: len(values)]
