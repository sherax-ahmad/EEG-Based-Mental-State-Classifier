"""
Tests for src/features/time_domain.py, frequency_domain.py, extractor.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from src.features.time_domain import (
    extract_time_features,
    hjorth_parameters,
    zero_crossing_rate,
    get_feature_names as time_feature_names,
)
from src.features.frequency_domain import (
    compute_psd,
    band_power,
    relative_band_power,
    spectral_entropy,
    extract_frequency_features,
    FREQUENCY_BANDS,
    get_feature_names as freq_feature_names,
)
from src.features.extractor import EEGFeatureExtractor, make_feature_matrix


SFREQ = 256
N_CHANNELS = 4
N_SAMPLES = SFREQ * 4   # 4-second epoch


@pytest.fixture
def epoch():
    """Synthetic epoch: alpha + beta sine waves + noise."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, N_SAMPLES / SFREQ, N_SAMPLES, endpoint=False)
    data = np.zeros((N_CHANNELS, N_SAMPLES))
    for ch in range(N_CHANNELS):
        data[ch] = (
            5.0 * np.sin(2 * np.pi * 10 * t)
            + 2.0 * np.sin(2 * np.pi * 20 * t)
            + rng.normal(0, 0.3, N_SAMPLES)
        )
    return data


@pytest.fixture
def epochs_batch(epoch):
    """Batch of 10 epochs."""
    rng = np.random.default_rng(7)
    return epoch[np.newaxis] + rng.normal(0, 0.1, (10, N_CHANNELS, N_SAMPLES))


# ── Time-domain features ──────────────────────────────────────────────────────

class TestTimeDomainFeatures:
    def test_output_is_1d(self, epoch):
        feat = extract_time_features(epoch)
        assert feat.ndim == 1

    def test_output_length(self, epoch):
        feat = extract_time_features(epoch)
        # 8 scalar features × N_CHANNELS + 3 Hjorth × N_CHANNELS
        expected = (8 + 3) * N_CHANNELS
        assert len(feat) == expected, f"Expected {expected}, got {len(feat)}"

    def test_no_nan_inf(self, epoch):
        feat = extract_time_features(epoch)
        assert np.all(np.isfinite(feat)), "NaN or Inf in time features"

    def test_zero_crossing_rate_range(self, epoch):
        zcr = zero_crossing_rate(epoch)
        assert zcr.shape == (N_CHANNELS,)
        assert np.all(zcr >= 0) and np.all(zcr <= 1)

    def test_hjorth_shape(self, epoch):
        h = hjorth_parameters(epoch)
        assert h.shape == (N_CHANNELS * 3,)

    def test_hjorth_activity_positive(self, epoch):
        h = hjorth_parameters(epoch).reshape(N_CHANNELS, 3)
        assert np.all(h[:, 0] >= 0), "Hjorth activity must be non-negative"

    def test_feature_names_length(self):
        names = time_feature_names(N_CHANNELS)
        feat = extract_time_features(np.zeros((N_CHANNELS, N_SAMPLES)))
        assert len(names) == len(feat)


# ── Frequency-domain features ─────────────────────────────────────────────────

class TestFrequencyDomainFeatures:
    def test_psd_shape(self, epoch):
        freqs, psd = compute_psd(epoch, SFREQ)
        assert psd.shape[0] == N_CHANNELS
        assert len(freqs) == psd.shape[1]

    def test_psd_non_negative(self, epoch):
        _, psd = compute_psd(epoch, SFREQ)
        assert np.all(psd >= 0)

    def test_alpha_power_dominant(self, epoch):
        """Epoch has strong 10 Hz → alpha power should be highest."""
        freqs, psd = compute_psd(epoch, SFREQ)
        alpha = band_power(psd, freqs, 8, 13)
        delta = band_power(psd, freqs, 0.5, 4)
        assert np.all(alpha > delta), "Alpha should dominate over delta"

    def test_relative_power_sums_to_one(self, epoch):
        freqs, psd = compute_psd(epoch, SFREQ)
        total_rel = sum(
            relative_band_power(psd, freqs, lo, hi)
            for _, (lo, hi) in FREQUENCY_BANDS.items()
        )
        assert np.allclose(total_rel, 1.0, atol=0.05)

    def test_spectral_entropy_range(self, epoch):
        freqs, psd = compute_psd(epoch, SFREQ)
        ent = spectral_entropy(psd, freqs)
        assert ent.shape == (N_CHANNELS,)
        assert np.all(ent >= 0) and np.all(ent <= 1)

    def test_extract_frequency_features_1d(self, epoch):
        feat = extract_frequency_features(epoch, SFREQ)
        assert feat.ndim == 1

    def test_no_nan_inf_freq(self, epoch):
        feat = extract_frequency_features(epoch, SFREQ)
        assert np.all(np.isfinite(feat))

    def test_freq_feature_names_length(self, epoch):
        feat = extract_frequency_features(epoch, SFREQ)
        names = freq_feature_names(N_CHANNELS)
        assert len(names) == len(feat)


# ── Combined extractor ────────────────────────────────────────────────────────

class TestEEGFeatureExtractor:
    def test_transform_shape(self, epochs_batch):
        ext = EEGFeatureExtractor(sfreq=SFREQ)
        X = ext.transform(epochs_batch)
        assert X.shape[0] == len(epochs_batch)
        assert X.ndim == 2

    def test_feature_names_match_columns(self, epochs_batch):
        ext = EEGFeatureExtractor(sfreq=SFREQ)
        X = ext.transform(epochs_batch)
        names = ext.get_feature_names()
        assert len(names) == X.shape[1]

    def test_time_only(self, epochs_batch):
        ext = EEGFeatureExtractor(sfreq=SFREQ, use_time=True, use_frequency=False)
        X = ext.transform(epochs_batch)
        assert X.shape[1] == (8 + 3) * N_CHANNELS

    def test_freq_only(self, epochs_batch):
        ext = EEGFeatureExtractor(sfreq=SFREQ, use_time=False, use_frequency=True)
        X = ext.transform(epochs_batch)
        assert X.shape[1] > 0

    def test_raises_if_both_disabled(self, epochs_batch):
        ext = EEGFeatureExtractor(sfreq=SFREQ, use_time=False, use_frequency=False)
        with pytest.raises(ValueError):
            ext.transform(epochs_batch)

    def test_get_feature_names_before_transform_raises(self):
        ext = EEGFeatureExtractor(sfreq=SFREQ)
        with pytest.raises(RuntimeError):
            ext.get_feature_names()

    def test_no_nan_inf_combined(self, epochs_batch):
        ext = EEGFeatureExtractor(sfreq=SFREQ)
        X = ext.transform(epochs_batch)
        assert np.all(np.isfinite(X))

    def test_make_feature_matrix(self, epochs_batch):
        labels = np.zeros(len(epochs_batch), dtype=int)
        X, y = make_feature_matrix(epochs_batch, labels, sfreq=SFREQ)
        assert X.shape[0] == len(labels)
        assert len(y) == len(labels)
