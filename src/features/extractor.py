"""
Combined feature extraction pipeline for EEG epochs.

Combines time-domain and frequency-domain features into a single
feature matrix suitable for scikit-learn classifiers.
"""

import numpy as np
from typing import List, Optional, Tuple

from .time_domain import extract_time_features, get_feature_names as time_names
from .frequency_domain import (
    extract_frequency_features,
    get_feature_names as freq_names,
    FREQUENCY_BANDS,
)


class EEGFeatureExtractor:
    """
    Extract a combined feature vector from EEG epochs.

    Args:
        sfreq         : Sampling frequency in Hz.
        use_time      : Include time-domain features.
        use_frequency : Include frequency-domain features.
        ch_names      : Optional list of channel names for labeling.
    """

    def __init__(
        self,
        sfreq: int = 256,
        use_time: bool = True,
        use_frequency: bool = True,
        ch_names: Optional[List[str]] = None,
    ):
        self.sfreq = sfreq
        self.use_time = use_time
        self.use_frequency = use_frequency
        self.ch_names = ch_names
        self._n_channels: Optional[int] = None

    def _extract_one(self, epoch: np.ndarray) -> np.ndarray:
        """Extract features from a single epoch (n_channels, n_samples)."""
        parts = []
        if self.use_time:
            parts.append(extract_time_features(epoch))
        if self.use_frequency:
            parts.append(extract_frequency_features(epoch, self.sfreq))
        if not parts:
            raise ValueError("At least one of use_time or use_frequency must be True.")
        return np.concatenate(parts)

    def transform(self, epochs: np.ndarray) -> np.ndarray:
        """
        Extract features from all epochs.

        Args:
            epochs : (n_epochs, n_channels, n_samples)

        Returns:
            X : (n_epochs, n_features) feature matrix
        """
        self._n_channels = epochs.shape[1]
        features = np.array([self._extract_one(ep) for ep in epochs])
        print(f"  Feature extraction complete: {features.shape}")
        return features

    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        if self._n_channels is None:
            raise RuntimeError("Call transform() before get_feature_names().")
        names = []
        if self.use_time:
            names += time_names(self._n_channels, self.ch_names)
        if self.use_frequency:
            names += freq_names(self._n_channels, self.ch_names)
        return names


def make_feature_matrix(
    epochs: np.ndarray,
    labels: np.ndarray,
    sfreq: int = 256,
    use_time: bool = True,
    use_frequency: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function: preprocess epochs → feature matrix.

    Args:
        epochs : (n_epochs, n_channels, n_samples)
        labels : (n_epochs,)
        sfreq  : Sampling frequency

    Returns:
        X : (n_epochs, n_features)
        y : (n_epochs,) same labels
    """
    extractor = EEGFeatureExtractor(
        sfreq=sfreq, use_time=use_time, use_frequency=use_frequency
    )
    X = extractor.transform(epochs)
    return X, labels
