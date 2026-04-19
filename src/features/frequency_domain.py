"""
Frequency-domain EEG feature extraction.

Features extracted:
- Power Spectral Density (PSD) via Welch's method
- Absolute and relative band power (delta, theta, alpha, beta, gamma)
- Spectral entropy
- Peak frequency per band
"""

import numpy as np
from scipy import signal
from typing import Dict, List, Optional, Tuple


# Standard EEG frequency bands (Hz)
FREQUENCY_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def compute_psd(
    epoch: np.ndarray,
    sfreq: int,
    nperseg: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density using Welch's method.

    Args:
        epoch  : (n_channels, n_samples)
        sfreq  : Sampling frequency in Hz
        nperseg: Samples per segment (default: sfreq → 1-second segments)

    Returns:
        freqs : (n_freqs,) frequency bins
        psd   : (n_channels, n_freqs) power values
    """
    if nperseg is None:
        nperseg = sfreq  # 1-second segments

    freqs, psd = signal.welch(epoch, fs=sfreq, nperseg=nperseg, axis=-1)
    return freqs, psd


def band_power(
    psd: np.ndarray,
    freqs: np.ndarray,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """
    Compute absolute band power by integrating PSD over [fmin, fmax].

    Args:
        psd   : (n_channels, n_freqs)
        freqs : (n_freqs,)
        fmin, fmax: frequency range

    Returns:
        power : (n_channels,) absolute power per channel
    """
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    freq_res = freqs[1] - freqs[0]
    return np.sum(psd[:, idx], axis=-1) * freq_res


def relative_band_power(
    psd: np.ndarray,
    freqs: np.ndarray,
    fmin: float,
    fmax: float,
    total_fmin: float = 0.5,
    total_fmax: float = 45.0,
) -> np.ndarray:
    """
    Compute relative band power (band_power / total_power).

    Returns:
        rel_power : (n_channels,)
    """
    bp = band_power(psd, freqs, fmin, fmax)
    total = band_power(psd, freqs, total_fmin, total_fmax) + 1e-10
    return bp / total


def spectral_entropy(psd: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """
    Compute normalized spectral entropy.

    Args:
        psd   : (n_channels, n_freqs)
        freqs : (n_freqs,)

    Returns:
        entropy : (n_channels,)
    """
    psd_norm = psd / (psd.sum(axis=-1, keepdims=True) + 1e-10)
    ent = -np.sum(psd_norm * np.log2(psd_norm + 1e-10), axis=-1)
    max_ent = np.log2(psd.shape[-1])
    return ent / (max_ent + 1e-10)


def peak_frequency(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    """
    Return the frequency of maximum power within [fmin, fmax].

    Returns:
        peak_freq : (n_channels,)
    """
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    band_psd = psd[:, idx]
    band_freqs = freqs[idx]
    peak_idx = np.argmax(band_psd, axis=-1)
    return band_freqs[peak_idx]


def extract_frequency_features(
    epoch: np.ndarray,
    sfreq: int,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    include_relative: bool = True,
    include_entropy: bool = True,
    include_peak_freq: bool = True,
) -> np.ndarray:
    """
    Extract all frequency-domain features from a single epoch.

    Args:
        epoch  : (n_channels, n_samples)
        sfreq  : Sampling frequency
        bands  : Dict of band_name -> (fmin, fmax). Defaults to FREQUENCY_BANDS.
        include_relative : Include relative band power features
        include_entropy  : Include spectral entropy
        include_peak_freq: Include peak frequency per band

    Returns:
        features : 1D feature vector (flattened across channels and features)
    """
    if bands is None:
        bands = FREQUENCY_BANDS

    freqs, psd = compute_psd(epoch, sfreq)
    feature_list = []

    for band_name, (fmin, fmax) in bands.items():
        # Absolute power
        abs_pow = band_power(psd, freqs, fmin, fmax)
        feature_list.append(abs_pow)

        # Relative power
        if include_relative:
            rel_pow = relative_band_power(psd, freqs, fmin, fmax)
            feature_list.append(rel_pow)

        # Peak frequency
        if include_peak_freq:
            pk_freq = peak_frequency(psd, freqs, fmin, fmax)
            feature_list.append(pk_freq)

    # Spectral entropy (one value per channel)
    if include_entropy:
        ent = spectral_entropy(psd, freqs)
        feature_list.append(ent)

    return np.concatenate(feature_list)


def get_feature_names(
    n_channels: int,
    ch_names: Optional[List[str]] = None,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    include_relative: bool = True,
    include_entropy: bool = True,
    include_peak_freq: bool = True,
) -> List[str]:
    """Generate human-readable feature names."""
    if bands is None:
        bands = FREQUENCY_BANDS
    if ch_names is None:
        ch_names = [f"ch{i}" for i in range(n_channels)]

    names = []
    for band_name in bands:
        for ch in ch_names:
            names.append(f"{ch}_{band_name}_abs_power")
        if include_relative:
            for ch in ch_names:
                names.append(f"{ch}_{band_name}_rel_power")
        if include_peak_freq:
            for ch in ch_names:
                names.append(f"{ch}_{band_name}_peak_freq")
    if include_entropy:
        for ch in ch_names:
            names.append(f"{ch}_spectral_entropy")
    return names
