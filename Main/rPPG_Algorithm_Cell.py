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


# rppg_methods.py
"""
Standard rPPG methods: POS and CHROM.

Implements:
- rPPG POS (Plane-Orthogonal-to-Skin)
- rPPG CHROM (chrominance-based rPPG)

Inputs:
    rgb_ts: np.ndarray, shape (T, 3), RGB mean values per frame.

Outputs:
    1D rPPG signal, shape (T,).

Reference:
    POS: Wang et al., "Algorithmic principles of remote PPG", IEEE TBME 2017.
    CHROM: de Haan & Jeanne, "Robust pulse rate from chrominance...", 2013.
"""

import numpy as np

# -------------------------------------------------------------------------
# Helper: detrend or normalize a time-series using zero-mean
# -------------------------------------------------------------------------

def _zero_mean(sig):
    sig = sig.astype(np.float32)
    return sig - np.mean(sig)


# -------------------------------------------------------------------------
# POS Method
# -------------------------------------------------------------------------

def rppg_pos(rgb_ts: np.ndarray) -> np.ndarray:
    """
    POS (Plane-Orthogonal-to-Skin) rPPG method.
    
    Parameters
    ----------
    rgb_ts : np.ndarray
        Shape (T, 3). Columns are [R, G, B].

    Returns
    -------
    rppg_sig : np.ndarray
        Pulse signal extracted by POS, shape (T,).
    """
    rgb_ts = rgb_ts.astype(np.float32)
    R = _zero_mean(rgb_ts[:, 0])
    G = _zero_mean(rgb_ts[:, 1])
    B = _zero_mean(rgb_ts[:, 2])

    # 2-D chrominance signals
    Xs = 3*R - 2*G
    Ys = 1.5*R + G - 1.5*B

    # Normalize (standard deviation normalization)
    std_X = np.std(Xs) + 1e-8
    std_Y = np.std(Ys) + 1e-8
    Xs /= std_X
    Ys /= std_Y

    # Combine into POS signal
    S = Xs - Ys

    return _zero_mean(S)


# -------------------------------------------------------------------------
# CHROM Method
# -------------------------------------------------------------------------

def rppg_chrom(rgb_ts: np.ndarray) -> np.ndarray:
    """
    CHROM method (de Haan & Jeanne, 2013).

    Parameters
    ----------
    rgb_ts : np.ndarray
        Shape (T, 3). Columns are [R, G, B].

    Returns
    -------
    rppg_sig : np.ndarray
        Pulse signal extracted by CHROM, shape (T,).
    """
    rgb_ts = rgb_ts.astype(np.float32)
    R = _zero_mean(rgb_ts[:, 0])
    G = _zero_mean(rgb_ts[:, 1])
    B = _zero_mean(rgb_ts[:, 2])

    # Chrominance signals
    Xs = 0.77*R - 0.51*G
    Ys = 0.77*R + 0.51*G - 0.77*B

    std_X = np.std(Xs) + 1e-8
    std_Y = np.std(Ys) + 1e-8

    S = Xs / std_X - Ys / std_Y

    return _zero_mean(S)
