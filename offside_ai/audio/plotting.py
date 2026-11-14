"""Spectrogram + onset peak visualisation for the audio stage."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_spectrogram_with_peaks(
    spectrogram_db: np.ndarray,
    times_s: np.ndarray,
    freqs_hz: np.ndarray,
    peak_times_s: Sequence[float],
    onset_env: np.ndarray | None = None,
    onset_times_s: np.ndarray | None = None,
    output_path: str | Path | None = None,
    title: str = "Pass / kick onset detection",
    max_freq_hz: float = 5000.0,
) -> Path | plt.Figure:
    """Plot spectrogram with vertical lines at detected onset peaks.

    If ``output_path`` is set, saves a PNG and returns that path; otherwise
    returns the matplotlib Figure.
    """
    freq_mask = freqs_hz <= max_freq_hz
    S = spectrogram_db[freq_mask]
    F = freqs_hz[freq_mask]

    n_rows = 2 if onset_env is not None and onset_times_s is not None else 1
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(10, 3.2 * n_rows), sharex=True, constrained_layout=True
    )
    if n_rows == 1:
        axes = [axes]

    ax0 = axes[0]
    extent = [float(times_s[0]), float(times_s[-1]), float(F[0]), float(F[-1])]
    im = ax0.imshow(
        S,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="magma",
        interpolation="nearest",
    )
    ax0.set_ylabel("Hz")
    ax0.set_title(title)
    for t in peak_times_s:
        ax0.axvline(float(t), color="cyan", linewidth=1.2, alpha=0.9)
    fig.colorbar(im, ax=ax0, fraction=0.02, pad=0.02, label="dB")

    if n_rows == 2:
        ax1 = axes[1]
        ax1.plot(onset_times_s, onset_env, color="steelblue", linewidth=1.0)
        for t in peak_times_s:
            ax1.axvline(float(t), color="crimson", linewidth=1.2, alpha=0.9)
        ax1.set_ylabel("Onset strength")
        ax1.set_xlabel("Time (s)")
    else:
        ax0.set_xlabel("Time (s)")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=140)
        plt.close(fig)
        return output_path
    return fig
