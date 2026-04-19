"""
EEG Signal visualization utilities.

- Raw multi-channel EEG traces
- Power spectral density plots
- Band power topomaps (if channel positions available)
- Spectrogram (time-frequency)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy import signal
from typing import List, Optional


def plot_eeg_traces(
    data: np.ndarray,
    sfreq: int,
    ch_names: Optional[List[str]] = None,
    duration: float = 10.0,
    offset_uv: float = 100.0,
    title: str = "EEG Signal",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot stacked EEG channel traces (waterfall style).

    Args:
        data      : (n_channels, n_samples)
        sfreq     : Sampling frequency in Hz
        ch_names  : Channel names (optional)
        duration  : Seconds to display
        offset_uv : Vertical offset between channels (µV)
        save_path : If given, save figure here
    """
    n_channels, n_samples = data.shape
    n_show = min(n_samples, int(duration * sfreq))
    t = np.arange(n_show) / sfreq

    if ch_names is None:
        ch_names = [f"CH{i+1}" for i in range(n_channels)]

    fig, ax = plt.subplots(figsize=(14, max(4, n_channels * 0.7)))

    for i, (ch, name) in enumerate(zip(data[:, :n_show], ch_names)):
        offset = (n_channels - 1 - i) * offset_uv
        ax.plot(t, ch + offset, lw=0.8, color="#2c7bb6", alpha=0.85)
        ax.text(-0.01, offset, name, ha="right", va="center", fontsize=9, transform=ax.get_yaxis_transform())

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_psd(
    data: np.ndarray,
    sfreq: int,
    ch_names: Optional[List[str]] = None,
    fmax: float = 50.0,
    title: str = "Power Spectral Density",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot PSD (Welch) for each channel with EEG band shading.

    Args:
        data  : (n_channels, n_samples)
        sfreq : Sampling frequency
        fmax  : Maximum frequency to display
    """
    BANDS = {
        "Delta\n(0.5–4)": (0.5, 4, "#d4e6f1"),
        "Theta\n(4–8)":   (4, 8,  "#d5f5e3"),
        "Alpha\n(8–13)":  (8, 13, "#fdf2e9"),
        "Beta\n(13–30)":  (13, 30, "#f9ebea"),
        "Gamma\n(30–45)": (30, 45, "#f5eef8"),
    }

    if ch_names is None:
        ch_names = [f"CH{i+1}" for i in range(data.shape[0])]

    fig, ax = plt.subplots(figsize=(12, 5))

    for ch_data, name in zip(data, ch_names):
        freqs, psd = signal.welch(ch_data, fs=sfreq, nperseg=sfreq)
        mask = freqs <= fmax
        ax.semilogy(freqs[mask], psd[mask], lw=1.2, alpha=0.7, label=name)

    # Shade frequency bands
    ymin, ymax = ax.get_ylim()
    for label, (flo, fhi, color) in BANDS.items():
        if fhi <= fmax:
            ax.axvspan(flo, fhi, alpha=0.15, color=color, label=None)
            ax.text((flo + fhi) / 2, ymax * 0.6, label, ha="center", va="top", fontsize=8, color="gray")

    ax.set_xlim(0, fmax)
    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("Power Spectral Density (µV²/Hz)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    if len(ch_names) <= 10:
        ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_spectrogram(
    data: np.ndarray,
    sfreq: int,
    channel: int = 0,
    fmax: float = 50.0,
    title: str = "Spectrogram",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot spectrogram (time-frequency representation) for a single channel.

    Args:
        data   : (n_channels, n_samples) or (n_samples,)
        sfreq  : Sampling frequency
        channel: Channel index to plot
        fmax   : Maximum frequency to display
    """
    ch_data = data[channel] if data.ndim == 2 else data

    f, t, Sxx = signal.spectrogram(ch_data, fs=sfreq, nperseg=sfreq, noverlap=sfreq // 2)
    mask = f <= fmax

    fig, ax = plt.subplots(figsize=(14, 5))
    pcm = ax.pcolormesh(t, f[mask], 10 * np.log10(Sxx[mask] + 1e-12), shading="gouraud", cmap="inferno")
    fig.colorbar(pcm, ax=ax, label="Power (dB)")
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Frequency (Hz)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_band_power_over_time(
    data: np.ndarray,
    sfreq: int,
    window_sec: float = 2.0,
    title: str = "Band Power Over Time",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot how EEG band power evolves over time (averaged across channels).

    Args:
        data       : (n_channels, n_samples)
        sfreq      : Sampling frequency
        window_sec : Window size in seconds for short-time PSD
    """
    BANDS = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30)}
    colors = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c"]

    win = int(window_sec * sfreq)
    step = win // 2
    n_samples = data.shape[1]
    times, band_powers = [], {b: [] for b in BANDS}

    for start in range(0, n_samples - win, step):
        seg = data[:, start : start + win]
        freqs, psd = signal.welch(seg, fs=sfreq, nperseg=min(win, sfreq), axis=-1)
        mean_psd = psd.mean(axis=0)
        times.append((start + win / 2) / sfreq)
        for band, (lo, hi) in BANDS.items():
            idx = (freqs >= lo) & (freqs <= hi)
            band_powers[band].append(np.trapz(mean_psd[idx], freqs[idx]))

    fig, ax = plt.subplots(figsize=(14, 5))
    for (band, vals), color in zip(band_powers.items(), colors):
        ax.plot(times, vals, label=band, lw=1.5, color=color)

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Band Power (µV²)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
