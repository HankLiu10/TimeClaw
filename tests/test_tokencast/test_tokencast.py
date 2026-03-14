"""Comprehensive tests for the TokenCast pipeline.

Tests quantizer strategies, encoder/decoder round-trips, channel-aware encoding,
multi-granularity encoding, special token wrapping, and vocabulary mappings.
"""

from __future__ import annotations

import numpy as np
import pytest

from timeclaw.tokencast.quantizer import (
    AdaptiveQuantizer,
    KMeansQuantizer,
    QuantileQuantizer,
    UniformQuantizer,
    get_quantizer,
)
from timeclaw.tokencast.encoder import TokenCastEncoder
from timeclaw.tokencast.decoder import TokenCastDecoder
from timeclaw.tokencast.vocabulary import (
    BOS_TOKEN,
    EOS_TOKEN,
    NUM_SPECIAL_TOKENS,
    MultiGranularityVocabulary,
    TimeSeriesVocabulary,
)


@pytest.fixture
def sin_data():
    return np.sin(np.linspace(0, 4 * np.pi, 200))


@pytest.fixture
def sin_2d(sin_data):
    cos_data = np.cos(np.linspace(0, 4 * np.pi, 200))
    return np.column_stack([sin_data, cos_data])


# ---------------------------------------------------------------------------
# Quantizer strategy tests
# ---------------------------------------------------------------------------

class TestUniformQuantizer:
    def test_fit_quantize_dequantize(self, sin_data):
        q = UniformQuantizer(n_bins=64)
        q.fit(sin_data)
        indices = q.quantize(sin_data)
        reconstructed = q.dequantize(indices)

        assert indices.shape == sin_data.shape
        assert indices.dtype == np.int32
        assert np.all(indices >= 0) and np.all(indices < 64)
        assert np.mean(np.abs(sin_data - reconstructed)) < 0.1

    def test_bin_edges_count(self, sin_data):
        q = UniformQuantizer(n_bins=128).fit(sin_data)
        assert q.bin_edges is not None
        assert len(q.bin_edges) == 129

    def test_bin_centers(self, sin_data):
        q = UniformQuantizer(n_bins=32).fit(sin_data)
        centers = q.bin_centers
        assert centers is not None
        assert len(centers) == 32


class TestQuantileQuantizer:
    def test_fit_quantize_dequantize(self, sin_data):
        q = QuantileQuantizer(n_bins=64)
        q.fit(sin_data)
        indices = q.quantize(sin_data)
        reconstructed = q.dequantize(indices)

        assert indices.shape == sin_data.shape
        assert np.all(indices >= 0) and np.all(indices < 64)
        mae = np.mean(np.abs(sin_data - reconstructed))
        assert mae < 0.15

    def test_edges_monotonic(self, sin_data):
        q = QuantileQuantizer(n_bins=64).fit(sin_data)
        edges = q.bin_edges
        assert np.all(np.diff(edges) > 0)


class TestKMeansQuantizer:
    def test_fit_quantize_dequantize(self, sin_data):
        q = KMeansQuantizer(n_bins=32, max_iter=20)
        q.fit(sin_data)
        indices = q.quantize(sin_data)
        reconstructed = q.dequantize(indices)

        assert indices.shape == sin_data.shape
        assert np.all(indices >= 0) and np.all(indices < 32)
        mae = np.mean(np.abs(sin_data - reconstructed))
        assert mae < 0.2

    def test_small_data_fallback(self):
        data = np.array([1.0, 2.0, 3.0])
        q = KMeansQuantizer(n_bins=64)
        result = q.fit(data)
        indices = result.quantize(data)
        assert indices.shape == data.shape


class TestAdaptiveQuantizer:
    def test_fit_quantize_dequantize(self, sin_data):
        q = AdaptiveQuantizer(n_bins=64, density_bins=16)
        q.fit(sin_data)
        indices = q.quantize(sin_data)
        reconstructed = q.dequantize(indices)

        assert indices.shape == sin_data.shape
        mae = np.mean(np.abs(sin_data - reconstructed))
        assert mae < 0.2

    def test_density_aware_allocation(self, sin_data):
        q = AdaptiveQuantizer(n_bins=128, density_bins=32).fit(sin_data)
        assert q._fitted
        assert q.bin_edges is not None
        assert len(q.bin_edges) == q.n_bins + 1


class TestGetQuantizer:
    @pytest.mark.parametrize("strategy", ["uniform", "quantile", "kmeans", "adaptive"])
    def test_factory(self, strategy, sin_data):
        q = get_quantizer(strategy, n_bins=32)
        q.fit(sin_data)
        indices = q.quantize(sin_data)
        reconstructed = q.dequantize(indices)
        mae = np.mean(np.abs(sin_data - reconstructed))
        assert mae < 0.3

    def test_unknown_strategy(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_quantizer("bogus")


# ---------------------------------------------------------------------------
# TokenCastEncoder tests
# ---------------------------------------------------------------------------

class TestTokenCastEncoder:
    @pytest.mark.parametrize("strategy", ["uniform", "quantile", "kmeans", "adaptive"])
    def test_encode_decode_roundtrip(self, sin_data, strategy):
        enc = TokenCastEncoder(vocab_size=128, strategy=strategy)
        enc.fit(sin_data)
        encoded = enc.encode(sin_data)
        decoded = enc.decode(encoded.tokens)

        assert encoded.tokens.shape == sin_data.shape
        assert np.all(encoded.tokens >= NUM_SPECIAL_TOKENS)
        mae = np.mean(np.abs(sin_data - decoded))
        assert mae < 0.15, f"MAE too high for strategy={strategy}: {mae}"

    def test_auto_fit_on_encode(self, sin_data):
        enc = TokenCastEncoder(vocab_size=64, strategy="uniform")
        encoded = enc.encode(sin_data)
        assert enc._fitted
        assert encoded.vocabulary is not None

    def test_channel_aware_encoding(self, sin_2d):
        enc = TokenCastEncoder(vocab_size=64, strategy="uniform", channel_aware=True)
        enc.fit(sin_2d)
        encoded = enc.encode(sin_2d)

        assert encoded.channel_tokens is not None
        assert len(encoded.channel_tokens) == 2
        for ch_tok in encoded.channel_tokens:
            assert ch_tok.shape == (200,)
            assert np.all(ch_tok >= NUM_SPECIAL_TOKENS)

    def test_multi_granularity_encoding(self, sin_data):
        enc = TokenCastEncoder(
            vocab_size=256,
            strategy="uniform",
            multi_granularity=True,
            coarse_bins=32,
        )
        enc.fit(sin_data)
        encoded = enc.encode(sin_data)

        assert encoded.coarse_tokens is not None
        assert encoded.multi_vocab is not None
        assert encoded.coarse_tokens.shape == sin_data.shape

    def test_special_token_wrapping_1d(self, sin_data):
        enc = TokenCastEncoder(vocab_size=64, strategy="uniform", add_special_tokens=True)
        enc.fit(sin_data)
        encoded = enc.encode(sin_data)

        assert encoded.add_special
        assert encoded.tokens[0] == BOS_TOKEN
        assert encoded.tokens[-1] == EOS_TOKEN
        assert len(encoded.tokens) == len(sin_data) + 2

    def test_special_token_wrapping_with_multi_granularity(self, sin_data):
        enc = TokenCastEncoder(
            vocab_size=64,
            strategy="uniform",
            multi_granularity=True,
            coarse_bins=16,
            add_special_tokens=True,
        )
        enc.fit(sin_data)
        encoded = enc.encode(sin_data)

        assert encoded.coarse_tokens is not None
        assert encoded.coarse_tokens[0] == BOS_TOKEN
        assert encoded.coarse_tokens[-1] == EOS_TOKEN

    def test_vocabulary_property(self, sin_data):
        enc = TokenCastEncoder(vocab_size=64, strategy="uniform")
        assert enc.vocabulary is None
        enc.fit(sin_data)
        assert enc.vocabulary is not None
        assert enc.vocabulary.vocab_size == NUM_SPECIAL_TOKENS + 64


# ---------------------------------------------------------------------------
# TokenCastDecoder tests
# ---------------------------------------------------------------------------

class TestTokenCastDecoder:
    def test_basic_decode(self, sin_data):
        enc = TokenCastEncoder(vocab_size=128, strategy="uniform")
        enc.fit(sin_data)
        encoded = enc.encode(sin_data)

        dec = TokenCastDecoder(vocabulary=encoded.vocabulary)
        decoded = dec.decode(encoded.tokens, strip_special=False)

        mae = np.mean(np.abs(sin_data - decoded))
        assert mae < 0.1

    def test_decode_strips_special(self, sin_data):
        enc = TokenCastEncoder(vocab_size=128, strategy="uniform", add_special_tokens=True)
        enc.fit(sin_data)
        encoded = enc.encode(sin_data)

        dec = TokenCastDecoder(vocabulary=encoded.vocabulary)
        decoded = dec.decode(encoded.tokens, strip_special=True)

        assert len(decoded) == len(sin_data)

    def test_multi_granularity_decode(self, sin_data):
        enc = TokenCastEncoder(
            vocab_size=256,
            strategy="uniform",
            multi_granularity=True,
            coarse_bins=32,
        )
        enc.fit(sin_data)
        encoded = enc.encode(sin_data)

        dec = TokenCastDecoder(
            vocabulary=encoded.vocabulary,
            multi_vocab=encoded.multi_vocab,
        )
        fine_tokens = encoded.tokens
        coarse_tokens = encoded.coarse_tokens

        fused = dec.decode_multi_granularity(
            coarse_tokens, fine_tokens, alpha=0.8, strip_special=False
        )
        assert fused.shape == sin_data.shape
        mae = np.mean(np.abs(sin_data - fused))
        assert mae < 0.15

    def test_decode_channels(self, sin_2d):
        enc = TokenCastEncoder(vocab_size=64, strategy="uniform", channel_aware=True)
        enc.fit(sin_2d)
        encoded = enc.encode(sin_2d)

        dec = TokenCastDecoder(channel_vocabs=enc._channel_vocabs)
        decoded = dec.decode_channels(encoded.channel_tokens, strip_special=False)

        assert decoded.shape == sin_2d.shape
        mae = np.mean(np.abs(sin_2d - decoded))
        assert mae < 0.2

    def test_smooth_decode(self, sin_data):
        enc = TokenCastEncoder(vocab_size=64, strategy="uniform")
        enc.fit(sin_data)
        encoded = enc.encode(sin_data)

        dec = TokenCastDecoder(vocabulary=encoded.vocabulary, smooth=True, smooth_window=5)
        decoded = dec.decode(encoded.tokens, strip_special=False)
        assert decoded.shape == sin_data.shape

    def test_no_vocabulary_raises(self):
        dec = TokenCastDecoder()
        with pytest.raises(RuntimeError, match="no vocabulary"):
            dec.decode(np.array([5, 6, 7]))

    def test_no_multi_vocab_raises(self):
        dec = TokenCastDecoder()
        with pytest.raises(RuntimeError, match="no multi-granularity"):
            dec.decode_multi_granularity(np.array([5]), np.array([5]))


# ---------------------------------------------------------------------------
# TimeSeriesVocabulary tests
# ---------------------------------------------------------------------------

class TestTimeSeriesVocabulary:
    def test_value_to_token_and_back(self, sin_data):
        edges = np.linspace(-1.0, 1.0, 65)
        vocab = TimeSeriesVocabulary(n_bins=64)
        vocab.build_from_edges(edges)

        tokens = vocab.value_to_token(sin_data)
        values = vocab.token_to_value(tokens)

        assert tokens.dtype == np.int32
        assert np.all(tokens >= NUM_SPECIAL_TOKENS)
        assert np.all(tokens < NUM_SPECIAL_TOKENS + 64)
        mae = np.mean(np.abs(sin_data - values))
        assert mae < 0.1

    def test_special_tokens_map_to_nan(self):
        edges = np.linspace(0, 10, 33)
        vocab = TimeSeriesVocabulary(n_bins=32)
        vocab.build_from_edges(edges)

        special_ids = np.array([BOS_TOKEN, EOS_TOKEN, 0, 3, 4], dtype=np.int32)
        values = vocab.token_to_value(special_ids)
        assert np.all(np.isnan(values))

    def test_vocab_size(self):
        vocab = TimeSeriesVocabulary(n_bins=100)
        assert vocab.vocab_size == NUM_SPECIAL_TOKENS + 100

    def test_is_special(self):
        vocab = TimeSeriesVocabulary()
        assert vocab.is_special(0)
        assert vocab.is_special(BOS_TOKEN)
        assert vocab.is_special(EOS_TOKEN)
        assert not vocab.is_special(NUM_SPECIAL_TOKENS)

    def test_token_name(self):
        edges = np.linspace(0, 1, 11)
        vocab = TimeSeriesVocabulary(n_bins=10)
        vocab.build_from_edges(edges)

        assert vocab.token_name(BOS_TOKEN) == "<BOS>"
        assert vocab.token_name(EOS_TOKEN) == "<EOS>"
        name = vocab.token_name(NUM_SPECIAL_TOKENS)
        assert name.startswith("v")

    def test_not_built_raises(self):
        vocab = TimeSeriesVocabulary()
        with pytest.raises(RuntimeError, match="not built"):
            vocab.value_to_token(np.array([1.0]))


# ---------------------------------------------------------------------------
# MultiGranularityVocabulary tests
# ---------------------------------------------------------------------------

class TestMultiGranularityVocabulary:
    def test_build_and_encode_decode(self, sin_data):
        mv = MultiGranularityVocabulary()
        mv.build_from_data(sin_data, coarse_bins=16, fine_bins=128)

        coarse_tokens = mv.encode_coarse(sin_data)
        fine_tokens = mv.encode_fine(sin_data)

        assert coarse_tokens.shape == sin_data.shape
        assert fine_tokens.shape == sin_data.shape

        coarse_vals = mv.decode_coarse(coarse_tokens)
        fine_vals = mv.decode_fine(fine_tokens)

        coarse_mae = np.mean(np.abs(sin_data - coarse_vals))
        fine_mae = np.mean(np.abs(sin_data - fine_vals))

        assert fine_mae < coarse_mae
        assert fine_mae < 0.05

    def test_coarse_fewer_unique_tokens(self, sin_data):
        mv = MultiGranularityVocabulary()
        mv.build_from_data(sin_data, coarse_bins=8, fine_bins=256)

        coarse_tokens = mv.encode_coarse(sin_data)
        fine_tokens = mv.encode_fine(sin_data)

        assert len(np.unique(coarse_tokens)) <= len(np.unique(fine_tokens))

    def test_vocab_sizes(self):
        mv = MultiGranularityVocabulary()
        data = np.random.randn(100)
        mv.build_from_data(data, coarse_bins=16, fine_bins=64)

        assert mv.coarse.vocab_size == NUM_SPECIAL_TOKENS + 16
        assert mv.fine.vocab_size == NUM_SPECIAL_TOKENS + 64
