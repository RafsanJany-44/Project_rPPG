# pure_alignment.py
"""
Utilities for aligning ground-truth PPG samples to video frame timestamps.
"""

from typing import Tuple

import numpy as np


def pure_align_ppg_to_frame_times(
    t_ppg_s: np.ndarray,
    wave: np.ndarray,
    t_vid_s: np.ndarray,
) -> np.ndarray:
    """
    Align ground-truth PPG waveform to video frame times using linear interpolation.

    Parameters
    ----------
    t_ppg_s : np.ndarray
        Time stamps of PPG samples in seconds (monotonically increasing).
    wave : np.ndarray
        PPG waveform samples corresponding to t_ppg_s.
    t_vid_s : np.ndarray
        Time stamps of video frames in seconds.

    Returns
    -------
    wave_interp : np.ndarray
        PPG waveform values interpolated at each video frame time.
    """
    # Use numpy interpolation to obtain PPG values at video frame times.
    # This assumes that t_ppg_s covers the full range of t_vid_s.
    wave_interp = np.interp(t_vid_s, t_ppg_s, wave)
    return wave_interp


def pure_select_time_window(
    t: np.ndarray,
    *arrays: np.ndarray,
    t_start: float,
    t_end: float,
) -> Tuple[np.ndarray, ...]:
    """
    Select a time window [t_start, t_end] from a time axis and corresponding arrays.

    Parameters
    ----------
    t : np.ndarray
        Time axis (seconds).
    arrays : np.ndarray
        One or more arrays with the same length as t.
    t_start : float
        Start time of the window (seconds).
    t_end : float
        End time of the window (seconds).

    Returns
    -------
    Tuple[np.ndarray, ...]
        Subset of (t, *arrays) within the specified time window.
    """
    mask = (t >= t_start) & (t <= t_end)
    out = [t[mask]]
    for arr in arrays:
        out.append(arr[mask])
    return tuple(out)
