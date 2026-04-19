"""
EEG Signal Preprocessing Pipeline.

Steps:
1. Bandpass filtering
2. Notch filtering (50/60 Hz powerline)
3. Artifact rejection (amplitude thresholding)
4. Epoching (segmentation into windows)
5. Normalization
"""

import numpy as np
from scipy import signal
from typing import Optional, Tuple, List


class EEGPreprocessor:
    """
    Complete preprocessing pipeline for raw EEG signals.

    Args:
        sfreq       : Sampling frequency in Hz.
        lowcut      : Low cutoff frequency for bandpass filter (Hz).
        highcut     : High cutoff frequency for bandpass filter (Hz).
        notch_freq  : Powerline noise frequency to notch out (50 or 60 Hz).
        epoch_length: Window length in seconds for segmentation.
        overlap     : Overlap between consecutive epochs (0.0–1.0).
        amplitude_threshold: Reject epochs with peak-to-peak amplitude above this (µV).
    """

    def __init__(
        self,
        sfreq: int = 256,
        lowcut: float = 0.5,
        highcut: float = 45.0,
        notch_freq: float = 50.0,
        epoch_length: float = 4.0,
        overlap: float = 0.5,
        amplitude_threshold: float = 100.0,
    ):
        self.sfreq = sfreq
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq
        self.epoch_length = epoch_length
        self.overlap = overlap
        self.amplitude_threshold = amplitude_threshold

        self._epoch_samples = int(epoch_length * sfreq)
        self._step = int(self._epoch_samples * (1 - overlap))

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply a zero-phase Butterworth bandpass filter.

        Args:
            data: (n_channels, n_samples) or (n_samples,)
        Returns:
            Filtered data of the same shape.
        """
        nyq = self.sfreq / 2.0
        low = self.lowcut / nyq
        high = self.highcut / nyq
        # Clamp to valid range
        low = max(low, 1e-6)
        high = min(high, 0.9999)
        b, a = signal.butter(4, [low, high], btype="band")
        return signal.filtfilt(b, a, data, axis=-1)

    def notch_filter(self, data: np.ndarray) -> np.ndarray:
        """Remove powerline interference at notch_freq Hz."""
        b, a = signal.iirnotch(self.notch_freq, Q=30, fs=self.sfreq)
        return signal.filtfilt(b, a, data, axis=-1)

    # ------------------------------------------------------------------
    # Epoching
    # ------------------------------------------------------------------

    def epoch(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Segment continuous EEG into fixed-length overlapping windows.

        Args:
            data  : (n_channels, n_samples)
            labels: (n_samples,) sample-level labels. If provided, the most
                    common label in each epoch is used as the epoch label.

        Returns:
            epochs  : (n_epochs, n_channels, epoch_samples)
            ep_labels: (n_epochs,) or None
        """
        n_channels, n_samples = data.shape
        epoch_list, label_list = [], []

        start = 0
        while start + self._epoch_samples <= n_samples:
            end = start + self._epoch_samples
            epoch = data[:, start:end]
            epoch_list.append(epoch)

            if labels is not None:
                ep_label = int(np.bincount(labels[start:end].astype(int)).argmax())
                label_list.append(ep_label)

            start += self._step

        if not epoch_list:
            raise ValueError("Data too short to create even one epoch.")

        epochs = np.stack(epoch_list, axis=0)
        ep_labels = np.array(label_list) if label_list else None
        return epochs, ep_labels

    # ------------------------------------------------------------------
    # Artifact rejection
    # ------------------------------------------------------------------

    def reject_artifacts(
        self,
        epochs: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Remove epochs that exceed the amplitude threshold.

        Args:
            epochs : (n_epochs, n_channels, epoch_samples)
            labels : (n_epochs,) optional

        Returns:
            clean_epochs, clean_labels (artifacts removed)
        """
        ptp = epochs.ptp(axis=-1).max(axis=-1)   # peak-to-peak per epoch
        mask = ptp < self.amplitude_threshold
        n_removed = (~mask).sum()
        if n_removed > 0:
            print(f"  Artifact rejection: removed {n_removed}/{len(epochs)} epochs")
        clean_labels = labels[mask] if labels is not None else None
        return epochs[mask], clean_labels

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def normalize(epochs: np.ndarray, method: str = "zscore") -> np.ndarray:
        """
        Normalize epoch data.

        Args:
            epochs: (n_epochs, n_channels, n_samples)
            method: 'zscore' | 'minmax'
        """
        if method == "zscore":
            mu = epochs.mean(axis=-1, keepdims=True)
            sd = epochs.std(axis=-1, keepdims=True) + 1e-8
            return (epochs - mu) / sd
        elif method == "minmax":
            mn = epochs.min(axis=-1, keepdims=True)
            mx = epochs.max(axis=-1, keepdims=True)
            return (epochs - mn) / (mx - mn + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        normalize: str = "zscore",
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Run the full preprocessing pipeline.

        Args:
            data   : (n_channels, n_samples) raw EEG
            labels : (n_samples,) sample-level labels (optional)
            normalize: normalization method

        Returns:
            X : (n_epochs, n_channels, epoch_samples) — preprocessed epochs
            y : (n_epochs,) or None
        """
        print("Preprocessing EEG data...")
        print(f"  Input shape: {data.shape}, sfreq={self.sfreq} Hz")

        data = self.bandpass_filter(data)
        data = self.notch_filter(data)

        epochs, ep_labels = self.epoch(data, labels)
        print(f"  Epochs created: {epochs.shape}")

        epochs, ep_labels = self.reject_artifacts(epochs, ep_labels)
        epochs = self.normalize(epochs, method=normalize)

        print(f"  Final shape: {epochs.shape}")
        return epochs, ep_labels
