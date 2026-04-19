"""
Data loaders for EEG datasets (EDF format and CSV).
Supports PhysioNet EDF files and generic CSV EEG data.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional, Dict

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("Warning: MNE not installed. EDF loading will be unavailable.")

try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False


class EEGDataLoader:
    """
    Unified loader for EEG data from multiple formats.

    Supports:
    - EDF/EDF+ files (PhysioNet, Sleep-EDF)
    - CSV files
    - NumPy .npy arrays
    """

    def __init__(self, data_dir: str, sampling_rate: Optional[int] = None):
        self.data_dir = Path(data_dir)
        self.sampling_rate = sampling_rate

    # ------------------------------------------------------------------
    # EDF loading (MNE)
    # ------------------------------------------------------------------

    def load_edf(
        self,
        filepath: str,
        channels: Optional[List[str]] = None,
        tmin: float = 0.0,
        tmax: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Load an EDF file using MNE.

        Args:
            filepath: Path to .edf file.
            channels: List of channel names to load. None = all channels.
            tmin: Start time in seconds.
            tmax: End time in seconds. None = end of recording.

        Returns:
            data   : ndarray (n_channels, n_samples)
            times  : ndarray (n_samples,)
            sfreq  : int sampling frequency
        """
        if not MNE_AVAILABLE:
            raise ImportError("MNE is required for EDF loading. Run: pip install mne")

        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)

        if channels:
            available = [ch for ch in channels if ch in raw.ch_names]
            if available:
                raw.pick_channels(available)

        raw.crop(tmin=tmin, tmax=tmax)
        data, times = raw.get_data(return_times=True)
        sfreq = int(raw.info["sfreq"])

        return data, times, sfreq

    def load_sleep_edf(
        self, subject_id: int, data_dir: Optional[str] = None
    ) -> Dict:
        """
        Load a Sleep-EDF subject with annotations.

        Returns dict with keys: data, labels, sfreq, ch_names
        """
        if not MNE_AVAILABLE:
            raise ImportError("MNE is required.")

        base = Path(data_dir) if data_dir else self.data_dir
        edf_files = sorted(base.glob(f"SC4{subject_id:02d}*PSG.edf"))
        ann_files = sorted(base.glob(f"SC4{subject_id:02d}*Hypnogram.edf"))

        if not edf_files:
            raise FileNotFoundError(f"No PSG file found for subject {subject_id}")

        raw = mne.io.read_raw_edf(str(edf_files[0]), preload=True, verbose=False)
        sfreq = int(raw.info["sfreq"])

        # Load annotations
        labels = []
        if ann_files:
            ann = mne.read_annotations(str(ann_files[0]))
            stage_map = {
                "Sleep stage W": 0,
                "Sleep stage 1": 1,
                "Sleep stage 2": 2,
                "Sleep stage 3": 3,
                "Sleep stage 4": 3,   # N3 = stage 3+4
                "Sleep stage R": 4,
            }
            for desc, onset, duration in zip(ann.description, ann.onset, ann.duration):
                stage = stage_map.get(desc, -1)
                n_epochs = int(duration // 30)
                labels.extend([stage] * n_epochs)

        return {
            "data": raw.get_data(),
            "labels": np.array(labels),
            "sfreq": sfreq,
            "ch_names": raw.ch_names,
        }

    # ------------------------------------------------------------------
    # CSV loading
    # ------------------------------------------------------------------

    def load_csv(
        self,
        filepath: str,
        label_col: Optional[str] = "label",
        drop_cols: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load EEG data from a CSV file.

        Returns:
            X : ndarray (n_samples, n_features_or_channels)
            y : ndarray (n_samples,) or None if no label column
        """
        df = pd.read_csv(filepath)
        if drop_cols:
            df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

        y = None
        if label_col and label_col in df.columns:
            y = df.pop(label_col).values

        return df.values, y

    # ------------------------------------------------------------------
    # Numpy loading
    # ------------------------------------------------------------------

    def load_numpy(self, filepath: str) -> np.ndarray:
        return np.load(filepath, allow_pickle=True)

    # ------------------------------------------------------------------
    # Batch loading
    # ------------------------------------------------------------------

    def load_directory(
        self, pattern: str = "*.csv", label_col: str = "label"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load all matching files in data_dir and concatenate."""
        X_list, y_list = [], []
        for fp in sorted(self.data_dir.glob(pattern)):
            X, y = self.load_csv(str(fp), label_col=label_col)
            if y is not None:
                X_list.append(X)
                y_list.append(y)
        if not X_list:
            raise FileNotFoundError(f"No files matching '{pattern}' in {self.data_dir}")
        return np.concatenate(X_list), np.concatenate(y_list)
