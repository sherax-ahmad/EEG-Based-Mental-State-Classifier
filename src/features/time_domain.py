"""
Time-domain EEG feature extraction.

Features extracted per channel:
- Mean, variance, standard deviation
- Skewness, kurtosis
- Root Mean Square (RMS)
- Zero-crossing rate
- Hjorth parameters (activity, mobility, complexity)
- Peak-to-peak amplitude
"""

import numpy as np
from scipy.stats import skew, kurtosis
from typing import List, Optional


def mean_signal(epoch: np.ndarray) -> np.ndarray:
    """Mean amplitude per channel. Shape: (n_channels,)"""
    return epoch.mean(axis=-1)


def variance_signal(epoch: np.ndarray) -> np.ndarray:
    """Signal variance per channel."""
    return epoch.var(axis=-1)


def std_signal(epoch: np.ndarray) -> np.ndarray:
    """Standard deviation per channel."""
    return epoch.std(axis=-1)


def rms_signal(epoch: np.ndarray) -> np.ndarray:
    """Root Mean Square amplitude per channel."""
    return np.sqrt(np.mean(epoch ** 2, axis=-1))


def skewness_signal(epoch: np.ndarray) -> np.ndarray:
    """Skewness per channel."""
    return skew(epoch, axis=-1)


def kurtosis_signal(epoch: np.ndarray) -> np.ndarray:
    """Kurtosis per channel."""
    return kurtosis(epoch, axis=-1)


def zero_crossing_rate(epoch: np.ndarray) -> np.ndarray:
    """
    Zero-crossing rate: number of times signal crosses zero, normalized by length.

    Returns: (n_channels,)
    """
    signs = np.sign(epoch)
    crossings = np.diff(signs, axis=-1)
    zcr = (crossings != 0).sum(axis=-1) / epoch.shape[-1]
    return zcr


def peak_to_peak(epoch: np.ndarray) -> np.ndarray:
    """Peak-to-peak amplitude per channel."""
    return epoch.max(axis=-1) - epoch.min(axis=-1)


def hjorth_parameters(epoch: np.ndarray) -> np.ndarray:
    """
    Compute Hjorth Activity, Mobility, and Complexity.

    - Activity   : variance of the signal
    - Mobility   : sqrt(var(1st derivative) / var(signal))
    - Complexity : mobility(1st derivative) / mobility(signal)

    Returns:
        features: (n_channels * 3,) — [activity_ch0, mobility_ch0, complexity_ch0, ...]
    """
    activity = epoch.var(axis=-1)

    d1 = np.diff(epoch, axis=-1)
    mobility = np.sqrt(d1.var(axis=-1) / (activity + 1e-10))

    d2 = np.diff(d1, axis=-1)
    mobility_d1 = np.sqrt(d2.var(axis=-1) / (d1.var(axis=-1) + 1e-10))
    complexity = mobility_d1 / (mobility + 1e-10)

    # Interleave: [act_0, mob_0, comp_0, act_1, ...]
    return np.stack([activity, mobility, complexity], axis=-1).flatten()


def extract_time_features(epoch: np.ndarray) -> np.ndarray:
    """
    Extract all time-domain features from a single epoch.

    Args:
        epoch : (n_channels, n_samples)

    Returns:
        features : 1D vector of shape (n_channels * n_features,)
    """
    feature_list = [
        mean_signal(epoch),
        variance_signal(epoch),
        std_signal(epoch),
        rms_signal(epoch),
        skewness_signal(epoch),
        kurtosis_signal(epoch),
        zero_crossing_rate(epoch),
        peak_to_peak(epoch),
        hjorth_parameters(epoch),   # already flattened (3 params * n_channels)
    ]
    return np.concatenate(feature_list)


def get_feature_names(
    n_channels: int, ch_names: Optional[List[str]] = None
) -> List[str]:
    """Return human-readable feature names for time-domain features."""
    if ch_names is None:
        ch_names = [f"ch{i}" for i in range(n_channels)]

    base_features = ["mean", "variance", "std", "rms", "skewness", "kurtosis", "zcr", "ptp"]
    names = [f"{ch}_{feat}" for feat in base_features for ch in ch_names]

    # Hjorth
    for ch in ch_names:
        names += [f"{ch}_hjorth_activity", f"{ch}_hjorth_mobility", f"{ch}_hjorth_complexity"]

    return names
