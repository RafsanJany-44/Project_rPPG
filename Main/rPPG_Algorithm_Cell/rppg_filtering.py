# rppg_filtering.py
"""
Zero-phase bandpass filtering for rPPG signals.

Implements:
- bandpass_zero_phase(): Butterworth bandpass filter with filtfilt
- Optional helpers for normalization

Typical rPPG band: 0.7–4.0 Hz (42–240 BPM)
"""

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_zero_phase(
    sig: np.ndarray,
    fs: float,
    low: float = 0.7,
    high: float = 4.0,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass filter.

    Parameters
    ----------
    sig : np.ndarray
        Input signal (1D), raw or unnormalized.
    fs : float
        Sampling rate in Hz (frames per second).
    low : float
        Low cutoff frequency in Hz (default 0.7).
    high : float
        High cutoff frequency in Hz (default 4.0).
    order : int
        Filter order (default 4).

    Returns
    -------
    sig_filt : np.ndarray
        Zero-phase filtered signal (same shape as input).
    """
    sig = np.asarray(sig, dtype=np.float32)

    if fs <= 0:
        return sig

    nyq = fs / 2.0
    low_n = low / nyq
    high_n = high / nyq

    # Avoid invalid frequency errors for high FPS or low fps
    if high_n >= 1.0:
        high_n = 0.999
    if low_n <= 0:
        low_n = 0.0001

    # Design Butterworth bandpass filter
    b, a = butter(order, [low_n, high_n], btype="band")

    # Zero-phase filtering (same as MATLAB filtfilt)
    try:
        sig_filt = filtfilt(b, a, sig, method="pad")
    except ValueError:
        # fallback: return raw signal
        return sig

    return sig_filt.astype(np.float32)
