"""
Tests for src/data/preprocessor.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from src.data.preprocessor import EEGPreprocessor


# ── Fixtures ──────────────────────────────────────────────────────────────────

SFREQ = 256
N_CHANNELS = 4
DURATION_SEC = 30
N_SAMPLES = SFREQ * DURATION_SEC


@pytest.fixture
def synthetic_eeg():
    """Synthetic multi-channel EEG: sum of sine waves at alpha + beta."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, DURATION_SEC, N_SAMPLES, endpoint=False)
    data = np.zeros((N_CHANNELS, N_SAMPLES))
    for ch in range(N_CHANNELS):
        data[ch] = (
            5.0  * np.sin(2 * np.pi * 10 * t)   # alpha
            + 3.0 * np.sin(2 * np.pi * 20 * t)  # beta
            + rng.normal(0, 0.5, N_SAMPLES)      # noise
        )
    return data


@pytest.fixture
def preprocessor():
    return EEGPreprocessor(
        sfreq=SFREQ,
        lowcut=0.5,
        highcut=45.0,
        notch_freq=50.0,
        epoch_length=4.0,
        overlap=0.5,
        amplitude_threshold=100.0,
    )


# ── Filtering ─────────────────────────────────────────────────────────────────

class TestFiltering:
    def test_bandpass_preserves_shape(self, preprocessor, synthetic_eeg):
        filtered = preprocessor.bandpass_filter(synthetic_eeg)
        assert filtered.shape == synthetic_eeg.shape

    def test_bandpass_attenuates_dc(self, preprocessor, synthetic_eeg):
        """DC component (0 Hz) should be largely removed by the highpass."""
        dc_data = synthetic_eeg + 50.0          # add large DC offset
        filtered = preprocessor.bandpass_filter(dc_data)
        assert abs(filtered.mean()) < abs(dc_data.mean())

    def test_notch_preserves_shape(self, preprocessor, synthetic_eeg):
        filtered = preprocessor.notch_filter(synthetic_eeg)
        assert filtered.shape == synthetic_eeg.shape

    def test_notch_attenuates_50hz(self, preprocessor):
        """50 Hz sine should be strongly attenuated after notch filter."""
        t = np.linspace(0, 10, SFREQ * 10, endpoint=False)
        noise_50 = np.tile(np.sin(2 * np.pi * 50 * t), (N_CHANNELS, 1))
        filtered = preprocessor.notch_filter(noise_50)
        ratio = filtered.std() / (noise_50.std() + 1e-10)
        assert ratio < 0.05, f"50 Hz not sufficiently attenuated (ratio={ratio:.4f})"


# ── Epoching ──────────────────────────────────────────────────────────────────

class TestEpoching:
    def test_epoch_shape(self, preprocessor, synthetic_eeg):
        epochs, _ = preprocessor.epoch(synthetic_eeg)
        epoch_samples = int(4.0 * SFREQ)
        assert epochs.ndim == 3
        assert epochs.shape[1] == N_CHANNELS
        assert epochs.shape[2] == epoch_samples

    def test_epoch_count_with_overlap(self, preprocessor, synthetic_eeg):
        epochs, _ = preprocessor.epoch(synthetic_eeg)
        # At least 1 epoch must be produced
        assert epochs.shape[0] > 0

    def test_epoch_with_labels(self, preprocessor, synthetic_eeg):
        labels = np.zeros(N_SAMPLES, dtype=int)
        labels[N_SAMPLES // 2 :] = 1     # second half is class 1
        epochs, ep_labels = preprocessor.epoch(synthetic_eeg, labels)
        assert ep_labels is not None
        assert len(ep_labels) == epochs.shape[0]

    def test_epoch_no_labels(self, preprocessor, synthetic_eeg):
        epochs, ep_labels = preprocessor.epoch(synthetic_eeg)
        assert ep_labels is None

    def test_epoch_too_short_raises(self, preprocessor):
        short_data = np.zeros((N_CHANNELS, 10))  # only 10 samples
        with pytest.raises(ValueError, match="too short"):
            preprocessor.epoch(short_data)


# ── Artifact Rejection ────────────────────────────────────────────────────────

class TestArtifactRejection:
    def test_removes_high_amplitude_epochs(self, preprocessor, synthetic_eeg):
        epochs, _ = preprocessor.epoch(synthetic_eeg)
        n_before = len(epochs)

        # Inject an obvious artifact into the first epoch
        epochs_with_artifact = epochs.copy()
        epochs_with_artifact[0] += 500.0   # way above threshold

        clean, _ = preprocessor.reject_artifacts(epochs_with_artifact)
        assert len(clean) < n_before

    def test_keeps_clean_epochs(self, preprocessor, synthetic_eeg):
        epochs, labels = preprocessor.epoch(synthetic_eeg, np.zeros(N_SAMPLES, int))
        clean, clean_labels = preprocessor.reject_artifacts(epochs, labels)
        # Synthetic data is well within threshold → nothing removed
        assert len(clean) == len(epochs)
        assert len(clean_labels) == len(clean)


# ── Normalization ─────────────────────────────────────────────────────────────

class TestNormalization:
    def test_zscore_mean_zero(self, preprocessor, synthetic_eeg):
        epochs, _ = preprocessor.epoch(synthetic_eeg)
        normed = EEGPreprocessor.normalize(epochs, method="zscore")
        assert abs(normed.mean()) < 0.1

    def test_zscore_std_one(self, preprocessor, synthetic_eeg):
        epochs, _ = preprocessor.epoch(synthetic_eeg)
        normed = EEGPreprocessor.normalize(epochs, method="zscore")
        # std per epoch-channel should be close to 1
        per_epoch_std = normed.std(axis=-1)
        assert np.allclose(per_epoch_std, 1.0, atol=0.1)

    def test_minmax_range(self, preprocessor, synthetic_eeg):
        epochs, _ = preprocessor.epoch(synthetic_eeg)
        normed = EEGPreprocessor.normalize(epochs, method="minmax")
        assert normed.min() >= 0.0 - 1e-6
        assert normed.max() <= 1.0 + 1e-6

    def test_unknown_method_raises(self, preprocessor, synthetic_eeg):
        epochs, _ = preprocessor.epoch(synthetic_eeg)
        with pytest.raises(ValueError, match="Unknown normalization"):
            EEGPreprocessor.normalize(epochs, method="mystery_norm")


# ── Full Pipeline ─────────────────────────────────────────────────────────────

class TestFullPipeline:
    def test_fit_transform_shape(self, preprocessor, synthetic_eeg):
        labels = np.zeros(N_SAMPLES, dtype=int)
        X, y = preprocessor.fit_transform(synthetic_eeg, labels)
        assert X.ndim == 3
        assert X.shape[1] == N_CHANNELS
        assert len(y) == X.shape[0]

    def test_fit_transform_no_labels(self, preprocessor, synthetic_eeg):
        X, y = preprocessor.fit_transform(synthetic_eeg)
        assert X.ndim == 3
        assert y is None
