"""TokenCast encoder: signal -> semantic token mapping.

Supports multiple quantization strategies, channel-aware encoding
for multivariate data, and multi-granularity (coarse + fine) output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from timeclaw.tokencast.quantizer import BaseQuantizer, get_quantizer
from timeclaw.tokencast.vocabulary import (
    BOS_TOKEN,
    EOS_TOKEN,
    MultiGranularityVocabulary,
    TimeSeriesVocabulary,
)


@dataclass
class EncodedOutput:
    """Result of encoding a time series."""

    tokens: np.ndarray
    coarse_tokens: np.ndarray | None = None
    channel_tokens: list[np.ndarray] | None = None
    vocabulary: TimeSeriesVocabulary | None = None
    multi_vocab: MultiGranularityVocabulary | None = None
    shape: tuple[int, ...] = ()
    add_special: bool = False


class TokenCastEncoder:
    """Encodes raw time series signals into semantic tokens.

    Supports:
    - Multiple quantization strategies (uniform, quantile, kmeans, adaptive)
    - Channel-aware encoding for multivariate data
    - Multi-granularity encoding (coarse + fine tokens simultaneously)
    - Optional BOS/EOS wrapping
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        strategy: str = "uniform",
        channel_aware: bool = False,
        multi_granularity: bool = False,
        coarse_bins: int = 64,
        add_special_tokens: bool = False,
        **quantizer_kwargs: Any,
    ):
        self.vocab_size = vocab_size
        self.strategy = strategy
        self.channel_aware = channel_aware
        self.multi_granularity = multi_granularity
        self.coarse_bins = coarse_bins
        self.add_special_tokens = add_special_tokens
        self.quantizer_kwargs = quantizer_kwargs

        self._fitted = False
        self._quantizer: BaseQuantizer | None = None
        self._channel_quantizers: list[BaseQuantizer] = []
        self._vocabulary: TimeSeriesVocabulary | None = None
        self._channel_vocabs: list[TimeSeriesVocabulary] = []
        self._multi_vocab: MultiGranularityVocabulary | None = None

    def fit(self, data: Any) -> TokenCastEncoder:
        """Fit the encoder to data to learn quantization boundaries."""
        arr = np.asarray(data, dtype=float)

        if self.channel_aware and arr.ndim == 2:
            self._fit_channel_aware(arr)
        else:
            flat = arr.ravel()
            self._quantizer = get_quantizer(self.strategy, self.vocab_size, **self.quantizer_kwargs)
            self._quantizer.fit(flat)
            self._vocabulary = TimeSeriesVocabulary(n_bins=self._quantizer.n_bins)
            self._vocabulary.build_from_edges(self._quantizer.bin_edges)

        if self.multi_granularity:
            self._multi_vocab = MultiGranularityVocabulary()
            self._multi_vocab.build_from_data(arr, self.coarse_bins, self.vocab_size)

        self._fitted = True
        return self

    def _fit_channel_aware(self, arr: np.ndarray) -> None:
        """Fit independent quantizers per channel (column)."""
        n_channels = arr.shape[1]
        self._channel_quantizers = []
        self._channel_vocabs = []
        for ch in range(n_channels):
            q = get_quantizer(self.strategy, self.vocab_size, **self.quantizer_kwargs)
            q.fit(arr[:, ch])
            v = TimeSeriesVocabulary(n_bins=q.n_bins, _name=f"ch{ch}")
            v.build_from_edges(q.bin_edges)
            self._channel_quantizers.append(q)
            self._channel_vocabs.append(v)
        # Also fit a global quantizer for the combined view
        self._quantizer = get_quantizer(self.strategy, self.vocab_size, **self.quantizer_kwargs)
        self._quantizer.fit(arr.ravel())
        self._vocabulary = TimeSeriesVocabulary(n_bins=self._quantizer.n_bins)
        self._vocabulary.build_from_edges(self._quantizer.bin_edges)

    def encode(self, data: Any) -> EncodedOutput:
        """Encode continuous values into discrete token IDs."""
        if not self._fitted:
            self.fit(data)

        arr = np.asarray(data, dtype=float)
        original_shape = arr.shape

        # Main tokens via global vocabulary
        tokens = self._vocabulary.value_to_token(arr)

        # Channel-specific tokens
        channel_tokens = None
        if self.channel_aware and arr.ndim == 2 and self._channel_vocabs:
            channel_tokens = []
            for ch in range(arr.shape[1]):
                ch_tok = self._channel_vocabs[ch].value_to_token(arr[:, ch])
                channel_tokens.append(ch_tok)

        # Multi-granularity
        coarse_tokens = None
        if self.multi_granularity and self._multi_vocab is not None:
            coarse_tokens = self._multi_vocab.encode_coarse(arr)

        # Optionally wrap with BOS/EOS
        if self.add_special_tokens:
            tokens = self._wrap_special(tokens)
            if coarse_tokens is not None:
                coarse_tokens = self._wrap_special(coarse_tokens)

        return EncodedOutput(
            tokens=tokens,
            coarse_tokens=coarse_tokens,
            channel_tokens=channel_tokens,
            vocabulary=self._vocabulary,
            multi_vocab=self._multi_vocab,
            shape=original_shape,
            add_special=self.add_special_tokens,
        )

    @staticmethod
    def _wrap_special(tokens: np.ndarray) -> np.ndarray:
        """Prepend BOS and append EOS to a 1D token sequence."""
        if tokens.ndim == 1:
            return np.concatenate([[BOS_TOKEN], tokens, [EOS_TOKEN]]).astype(np.int32)
        # For 2D (time x channels), wrap along time axis
        bos = np.full((1, *tokens.shape[1:]), BOS_TOKEN, dtype=np.int32)
        eos = np.full((1, *tokens.shape[1:]), EOS_TOKEN, dtype=np.int32)
        return np.concatenate([bos, tokens, eos], axis=0)

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        """Decode token IDs back to continuous values (bin centers)."""
        if self._vocabulary is None:
            raise RuntimeError("Encoder not fitted")
        return self._vocabulary.token_to_value(np.asarray(tokens, dtype=np.int32))

    @property
    def vocabulary(self) -> TimeSeriesVocabulary | None:
        return self._vocabulary
