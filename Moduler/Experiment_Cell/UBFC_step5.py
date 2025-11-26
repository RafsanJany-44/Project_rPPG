# UBFC_step5_rppg_vs_gt_phase_lag.py
"""
UBFC STEP 5: Extract rPPG (POS/CHROM) from UBFC video in a given time window
and compare with ground-truth PPG for phase-lag analysis.

Pipeline:
1) Choose a UBFC sequence (vid_1) and a window [t_start, t_end] in seconds.
2) Load GT PPG + timestamps and fix non-increasing timestamps.
3) Build UBFCFrameSource (video frames + timestamps).
4) Extract RGB time series from frames in the window (via ROI strategy).
5) Interpolate GT PPG at frame timestamps.
6) Apply POS or CHROM to RGB time series to obtain rPPG.
7) Plot rPPG vs GT PPG and compute cross-correlation-based phase lag.
"""

import sys
from pathlib import Path
# Add project root (Project_rPPG) to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from Main.Data_Read_Engine.ubfc_alignment import (
    load_ubfc_gt_and_fix,
    ubfc_align_ppg_to_frame_times,
)

from Main.Signal_Processing_Engine.ubfc_dataset import UBFCFrameSource
from Main.Signal_Processing_Engine.roi_central import CentralRoiExtractor
from Main.Signal_Processing_Engine.roi_face_opencv import OpenCVFaceBoxRoi
from Main.Signal_Processing_Engine.roi_face_mediapipe import MediaPipeFaceRegionsRoi
from Main.Signal_Processing_Engine.rgb_extractor import extract_rgb_timeseries

from Main.rPPG_Algorithm_Cell import rppg_chrom, rppg_pos
from Main.rPPG_Algorithm_Cell import bandpass_zero_phase



# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
UBFC_ROOT = Path(r"D:\Data\UBFC\Dataset_3")
SEQ_ID = "vid_1"

SEQ_DIR = UBFC_ROOT / SEQ_ID
GT_FILE = SEQ_DIR / "ground_truth_1.txt"
VIDEO_FILE = SEQ_DIR / f"{SEQ_ID}.avi"

# Time window in seconds (you can tweak these)
T_START = 5.0
T_END = 11.0

# rPPG method: "POS" or "CHROM"
RPPG_METHOD = "CHROM"

# ROI strategy: "central", "face_opencv", "face_mediapipe"
ROI_MODE = "central"
ROI_FRAC = 0.5   # only used for central ROI


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _normalize(sig: np.ndarray) -> np.ndarray:
    sig = np.asarray(sig, dtype=np.float32)
    sig = sig - np.mean(sig)
    std = np.std(sig)
    if std == 0:
        return sig
    return sig / std


def estimate_phase_lag_sign_invariant(
    s_rppg: np.ndarray,
    s_ppg: np.ndarray,
    dt: float,
    max_lag_seconds: float = 2.0,
) -> float:
    """
    Estimate phase lag using sign-invariant cross-correlation (same as PURE).

    Returns
    -------
    lag_sec : float
        Estimated lag in seconds (positive means rPPG lags PPG).
    """
    s1 = _normalize(s_rppg)
    s2 = _normalize(s_ppg)

    T = len(s1)
    if T < 2:
        return 0.0

    corr = np.correlate(s1, s2, mode="full")
    lags = np.arange(-T + 1, T)

    max_lag_samples = int(max_lag_seconds / dt)
    max_lag_samples = min(max_lag_samples, T - 1)

    mask = (lags >= -max_lag_samples) & (lags <= max_lag_samples)
    corr_win = corr[mask]
    lags_win = lags[mask]

    abs_corr_win = np.abs(corr_win)
    best_idx = int(np.argmax(abs_corr_win))
    best_lag_samples = int(lags_win[best_idx])

    lag_sec = float(best_lag_samples * dt)
    return lag_sec


def make_roi(roi_mode: str, roi_frac: float):
    if roi_mode == "central":
        return CentralRoiExtractor(frac=roi_frac)
    elif roi_mode == "face_opencv":
        return OpenCVFaceBoxRoi()
    elif roi_mode == "face_mediapipe":
        return MediaPipeFaceRegionsRoi()
    else:
        raise ValueError(f"Unknown ROI_MODE: {roi_mode}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    method_label = RPPG_METHOD.upper()
    print(
        f"=== UBFC STEP 5: rPPG ({method_label}) vs GT PPG "
        f"for {SEQ_ID}, window [{T_START}, {T_END}] s, ROI={ROI_MODE} ==="
    )

    # 1) Load GT PPG + timestamps (with fixes)
    t_ppg_s, ppg_wave, corrected_idx = load_ubfc_gt_and_fix(GT_FILE)
    print(f"Loaded GT: {len(ppg_wave)} samples")
    if corrected_idx:
        print(f"Corrected {len(corrected_idx)} non-increasing timestamps.")

    # 2) Build UBFCFrameSource + ROI
    source = UBFCFrameSource(VIDEO_FILE)
    roi_extractor = make_roi(ROI_MODE, ROI_FRAC)

    # 3) Extract RGB time series in the chosen window
    t_frame_s_win, rgb_ts = extract_rgb_timeseries(
        source=source,
        roi_extractor=roi_extractor,
        t_start=T_START,
        t_end=T_END,
    )

    if len(t_frame_s_win) == 0:
        raise RuntimeError(
            f"No frames in window [{T_START}, {T_END}] s for {SEQ_ID}"
        )

    print(f"Frames in window: {len(t_frame_s_win)}")
    print(f"RGB time series shape: {rgb_ts.shape}")  # (T, 3)

    dt_frames = np.diff(t_frame_s_win)
    dt_mean = float(np.mean(dt_frames))
    fps_est = 1.0 / dt_mean
    print(f"Estimated frame rate in window: {fps_est:.3f} Hz")

    # 4) Interpolate GT PPG at frame timestamps
    ppg_gt_at_frames = ubfc_align_ppg_to_frame_times(
        t_ppg_s=t_ppg_s,
        ppg_wave=ppg_wave,
        t_frame_s=t_frame_s_win,
    )

    # Detrend GT PPG slightly by removing its mean
    ppg_gt_at_frames = ppg_gt_at_frames - np.mean(ppg_gt_at_frames)

    # 5) Compute rPPG using POS / CHROM
    if method_label == "POS":
        rppg_sig = rppg_pos(rgb_ts)
    elif method_label == "CHROM":
        rppg_sig = rppg_chrom(rgb_ts)
    else:
        raise ValueError("RPPG_METHOD must be 'POS' or 'CHROM'.")

    # 6) Optional filtering (same cardiac band as PURE)
    fs = fps_est
    ppg_filt = bandpass_zero_phase(ppg_gt_at_frames, fs=fs)
    rppg_filt = bandpass_zero_phase(rppg_sig, fs=fs)

    # Normalize
    ppg_n = _normalize(ppg_filt)
    rppg_n = _normalize(rppg_filt)

    # 7) Estimate phase lag
    lag_sec = estimate_phase_lag_sign_invariant(
        s_rppg=rppg_n,
        s_ppg=ppg_n,
        dt=dt_mean,
        max_lag_seconds=2.0,
    )

    print("\n--- Phase lag estimate (rPPG vs GT PPG) ---")
    print(
        f"{method_label} lag: {lag_sec * 1000:.2f} ms "
        f"(positive means {method_label} lags behind PPG)"
    )

    # 8) Visualization
    t_rel = t_frame_s_win  # already in seconds from first frame

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_rel, ppg_n, label="GT PPG (bandpassed, norm.)")
    ax.plot(t_rel, rppg_n, label=f"rPPG {method_label} (bandpassed, norm.)", alpha=0.8)

    ax.set_title(
        f"UBFC {SEQ_ID}: rPPG ({method_label}) vs GT PPG\n"
        f"Window {T_START:.1f}–{T_END:.1f} s, ROI={ROI_MODE}\n"
        f"Estimated lag ≈ {lag_sec*1000:.1f} ms"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized amplitude (a.u.)")
    ax.grid(True)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
