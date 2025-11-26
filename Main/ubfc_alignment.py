# Main/Data_Read_Engine/ubfc_alignment.py
"""
Helpers for UBFC GT PPG + video alignment.

Reuses your existing ubfc_gt_io.load_ubfc_ground_truth, and provides:
- timestamp fixing (non-strictly increasing),
- convenience loader,
- generic interpolation helper.
"""

from typing import Tuple, List
from pathlib import Path
import numpy as np

from ubfc_gt_io import load_ubfc_ground_truth  


def fix_timestamps_inplace(t_s: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, List[int]]:
    """
    Detect and fix non-strictly increasing timestamps in place.

    Whenever t_s[i] <= t_s[i-1], t_s[i] is nudged forward to
    (t_s[i-1] + eps). Returns corrected timestamps and indices.
    """
    t_s = np.asarray(t_s, dtype=np.float64)
    corrected_indices: List[int] = []
    for i in range(1, len(t_s)):
        if t_s[i] <= t_s[i - 1]:
            t_s[i] = t_s[i - 1] + eps
            corrected_indices.append(i)
    return t_s, corrected_indices


def load_ubfc_gt_and_fix(path: Path) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Convenience wrapper:
    - load UBFC GT file (PPG + timestamps),
    - fix non-increasing timestamps.

    Returns
    -------
    t_ppg_s : np.ndarray
    ppg_wave : np.ndarray
    corrected_indices : list[int]
    """
    t_ppg_s, ppg_wave = load_ubfc_ground_truth(str(path))
    t_ppg_s_fixed, corrected = fix_timestamps_inplace(t_ppg_s, eps=1e-6)
    return t_ppg_s_fixed, ppg_wave, corrected


def ubfc_align_ppg_to_frame_times(
    t_ppg_s: np.ndarray,
    ppg_wave: np.ndarray,
    t_frame_s: np.ndarray,
) -> np.ndarray:
    """
    Align UBFC GT PPG to video frame timestamps using interpolation.

    Equivalent to your repeated np.interp usage.

    Parameters
    ----------
    t_ppg_s : np.ndarray
        GT PPG timestamps (seconds, strictly increasing).
    ppg_wave : np.ndarray
        GT PPG waveform samples, same length as t_ppg_s.
    t_frame_s : np.ndarray
        Video frame times (seconds).

    Returns
    -------
    ppg_at_frames : np.ndarray
        PPG signal interpolated at frame times.
    """
    t_ppg_s = np.asarray(t_ppg_s, dtype=np.float64)
    ppg_wave = np.asarray(ppg_wave, dtype=np.float64)
    t_frame_s = np.asarray(t_frame_s, dtype=np.float64)

    if len(t_ppg_s) < 2:
        raise RuntimeError("Not enough GT samples for interpolation.")

    return np.interp(t_frame_s, t_ppg_s, ppg_wave)
