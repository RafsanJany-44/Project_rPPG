# step5_rppg_vs_gt_phase_lag.py
"""
STEP 5: Extract rPPG (POS/CHROM) from PURE frames in a given time window
and compare with ground-truth PPG for phase-lag analysis.

Pipeline:
1) Choose sequence (01-01) and window [t_start, t_end].
2) From JSON:
   - load PPG timestamps and waveform (/FullPackage).
3) From image filenames:
   - load frame timestamps and extract RGB time series using a configurable ROI.
4) Interpolate GT PPG at frame timestamps (physically correct alignment).
5) Apply POS and CHROM to RGB time series to obtain rPPG signals.
6) Plot rPPG vs GT PPG and compute cross-correlation-based phase lag.

This version is aligned with the new modular architecture:
- PUREFrameSource (dataset abstraction)
- RoiExtractor implementations (central, OpenCV face, MediaPipe face regions)
- extract_rgb_timeseries (dataset-agnostic, ROI-agnostic)
"""
import sys
from pathlib import Path
# Add project root (Project_rPPG) to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from Main.Data_Read_Engine import DEFAULT_SEQ_ID
from Main.Data_Read_Engine import load_pure_json, extract_streams_from_pure_json
from Main.Data_Read_Engine import pure_align_ppg_to_frame_times, pure_select_time_window

from Main.Signal_Processing_Engine.pure_dataset import PUREFrameSource
from Main.Signal_Processing_Engine.roi_central import CentralRoiExtractor
from Main.Signal_Processing_Engine.roi_face_opencv import OpenCVFaceBoxRoi
from Main.Signal_Processing_Engine.roi_face_mediapipe import MediaPipeFaceRegionsRoi
from Main.Signal_Processing_Engine.rgb_extractor import extract_rgb_timeseries

from  Main.rPPG_Algorithm_Cell import rppg_pos, rppg_chrom

# ---------------------------------------------------------------------
# Helper: choose ROI strategy & extract RGB with new architecture
# ---------------------------con------------------------------------------

def extract_pure_rgb_timeseries_modular(
    seq_id: str,
    t_start: float,
    t_end: float,
    roi_mode: str = "central",
    roi_frac: float = 0.5,
):
    """
    Wrapper that wires PUREFrameSource + chosen RoiExtractor into the
    generic extract_rgb_timeseries() function.

    Parameters
    ----------
    seq_id : str
        PURE sequence ID, e.g., "01-01".
    t_start : float
        Start time of the window in seconds (relative to first video frame).
    t_end : float
        End time of the window in seconds.
    roi_mode : {"central", "face_opencv", "face_mediapipe"}
        Which ROI strategy to use.
    roi_frac : float
        Fraction for central ROI (only used when roi_mode == "central").

    Returns
    -------
    t_frame_s_win : np.ndarray
        Frame timestamps in seconds, shape (T,).
    rgb_ts : np.ndarray
        Mean RGB per frame, shape (T, 3), float32, RGB order.
    """
    # Dataset abstraction
    source = PUREFrameSource(seq_id)

    # ROI strategy selection (Open/Closed: add more without changing caller)
    if roi_mode == "central":
        roi = CentralRoiExtractor(frac=roi_frac)
    elif roi_mode == "face_opencv":
        roi = OpenCVFaceBoxRoi()
    elif roi_mode == "face_mediapipe":
        roi = MediaPipeFaceRegionsRoi()
    else:
        raise ValueError(f"Unknown roi_mode: {roi_mode}")

    # Generic extractor (dataset-agnostic + ROI-agnostic)
    t_frame_s_win, rgb_ts = extract_rgb_timeseries(
        source=source,
        roi_extractor=roi,
        t_start=t_start,
        t_end=t_end,
    )
    return t_frame_s_win, rgb_ts


# ---------------------------------------------------------------------
# Phase lag estimation (unchanged)
# ---------------------------------------------------------------------

def estimate_phase_lag(
    s1: np.ndarray,
    s2: np.ndarray,
    dt: float,
    max_lag_seconds: float = 2.0,
) -> float:
    """
    Estimate phase lag between two signals using cross-correlation.

    Parameters
    ----------
    s1 : np.ndarray
        First signal, shape (T,).
    s2 : np.ndarray
        Second signal, shape (T,). Should be same length as s1.
    dt : float
        Sampling interval in seconds (assumed constant).
    max_lag_seconds : float
        Maximum lag to consider in either direction.

    Returns
    -------
    lag_sec : float
        Estimated lag in seconds (positive means s1 lags behind s2).
    """
    s1 = np.asarray(s1, dtype=np.float32)
    s2 = np.asarray(s2, dtype=np.float32)
    assert s1.shape == s2.shape, "Signals must have the same shape."

    # Remove mean and normalize to unit variance
    s1 = s1 - np.mean(s1)
    s2 = s2 - np.mean(s2)
    s1_std = np.std(s1)
    s2_std = np.std(s2)
    if s1_std == 0 or s2_std == 0:
        return 0.0
    s1 /= s1_std
    s2 /= s2_std

    T = len(s1)
    max_lag_samples = int(max_lag_seconds / dt)
    max_lag_samples = min(max_lag_samples, T - 1)

    # Full cross-correlation
    corr = np.correlate(s1, s2, mode="full")
    lags = np.arange(-T + 1, T)

    # Restrict to a window around zero lag
    mask = (lags >= -max_lag_samples) & (lags <= max_lag_samples)
    corr_win = corr[mask]
    lags_win = lags[mask]

    # Find lag of maximum correlation
    best_idx = np.argmax(corr_win)
    best_lag_samples = lags_win[best_idx]
    lag_sec = best_lag_samples * dt

    return float(lag_sec)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    seq_id = DEFAULT_SEQ_ID
    print(f"=== STEP 5: rPPG (POS/CHROM) vs GT PPG for sequence {seq_id} ===")

    # Choose ROI strategy here:
    #   "central"      -> central rectangular ROI (similar to your old code)
    #   "face_opencv"  -> OpenCV Haar face box
    #   "face_mediapipe" -> MediaPipe face regions (forehead + cheeks approx)
    ROI_MODE = "central"
    ROI_FRAC = 0.5   # used only for central ROI

    # 1) Choose a window in *actual recording time* (no resampling)
    t_start = 5.0   # seconds
    t_end = 11.0    # seconds
    print(f"Window: [{t_start:.1f}, {t_end:.1f}] s (duration {t_end - t_start:.1f} s)")
    print(f"ROI mode: {ROI_MODE}")

    # 2) Load JSON and extract PPG and frame timestamps
    data = load_pure_json(seq_id)
    t_ppg_ns, wave, t_vid_ns_json, hr_dev = extract_streams_from_pure_json(data)

    # Use first video frame as time origin (consistent with Steps 1–4)
    t0_ns = t_vid_ns_json[0]
    t_ppg_s = (t_ppg_ns - t0_ns) * 1e-9

    # 3) Extract RGB time series from frames in the chosen window
    t_frame_s_win, rgb_ts = extract_pure_rgb_timeseries_modular(
        seq_id=seq_id,
        t_start=t_start,
        t_end=t_end,
        roi_mode=ROI_MODE,
        roi_frac=ROI_FRAC,
    )

    print(f"Frames in window: {len(t_frame_s_win)}")
    print(f"RGB time series shape: {rgb_ts.shape}")  # (T, 3)

    # Estimate frame interval from timestamps (no artificial correction)
    dt_frames = np.diff(t_frame_s_win)
    dt_mean = float(np.mean(dt_frames))
    fps_est = 1.0 / dt_mean
    print(f"Estimated frame rate in window: {fps_est:.3f} Hz")

    # 4) Interpolate GT PPG at these frame timestamps
    #    (physically correct: we evaluate PPG(t) at the exact frame times)
    ppg_gt_at_frames = pure_align_ppg_to_frame_times(
        t_ppg_s=t_ppg_s,
        wave=wave,
        t_vid_s=t_frame_s_win,
    )

    # Detrend GT PPG slightly by removing its mean
    ppg_gt_at_frames = ppg_gt_at_frames - np.mean(ppg_gt_at_frames)

    # 5) Compute rPPG signals using POS and CHROM
    rppg_pos_sig = rppg_pos(rgb_ts)
    rppg_chrom_sig = rppg_chrom(rgb_ts)

    # Optionally normalize all signals to unit variance for fair comparison
    def _normalize(sig):
        sig = sig - np.mean(sig)
        std = np.std(sig)
        if std == 0:
            return sig
        return sig / std

    ppg_gt_n = _normalize(ppg_gt_at_frames)
    rppg_pos_n = _normalize(rppg_pos_sig)
    rppg_chrom_n = _normalize(rppg_chrom_sig)

    # 6) Estimate phase lag using cross-correlation
    lag_pos_sec = estimate_phase_lag(rppg_pos_n, ppg_gt_n, dt=dt_mean)
    lag_chrom_sec = estimate_phase_lag(rppg_chrom_n, ppg_gt_n, dt=dt_mean)

    print("\n--- Phase lag estimates (rPPG vs GT PPG) ---")
    print(
        f"POS   lag: {lag_pos_sec*1000:.2f} ms "
        f"(positive means POS lags behind PPG)"
    )
    print(
        f"CHROM lag: {lag_chrom_sec*1000:.2f} ms "
        f"(positive means CHROM lags behind PPG)"
    )

    # 7) Visualization
    t_rel = t_frame_s_win  # already relative to first frame and clipped to window

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_rel, ppg_gt_n, label="GT PPG (interp at frame times)")
    ax.plot(t_rel, rppg_pos_n, label="rPPG POS", alpha=0.8)
    ax.plot(t_rel, rppg_chrom_n, label="rPPG CHROM", alpha=0.8)

    ax.set_title(
        f"Sequence {seq_id} rPPG vs GT PPG\n"
        f"Window {t_start:.1f}-{t_end:.1f} s, ROI={ROI_MODE}"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized amplitude (a.u.)")
    ax.grid(True)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
