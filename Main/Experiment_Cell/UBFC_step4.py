"""
UBFC STEP 4 (modular): Sliding-window phase-lag analysis (8 s, 1 s stride).

This mirrors PURE Step 8 but using the new architecture:
- UBFCFrameSource (FrameSource implementation)
- RoiExtractor (central / face-based)
- extract_rgb_timeseries (dataset-agnostic)
- rPPG methods (POS/CHROM)
- bandpass_zero_phase
"""

import sys
from pathlib import Path
# Add project root (Project_rPPG) to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from Main.Data_Read_Engine import load_ubfc_ground_truth  

# Dataset abstraction + ROI + RGB extractor
from Main.Signal_Processing_Engine.ubfc_dataset import UBFCFrameSource
from Main.Signal_Processing_Engine.roi_central import CentralRoiExtractor
from Main.Signal_Processing_Engine.roi_face_opencv import OpenCVFaceBoxRoi
from Main.Signal_Processing_Engine.roi_face_mediapipe import MediaPipeFaceRegionsRoi
from Main.Signal_Processing_Engine.rgb_extractor import extract_rgb_timeseries

# rPPG + filtering
from Main.rPPG_Algorithm_Cell import rppg_chrom, rppg_pos
from Main.rPPG_Algorithm_Cell import bandpass_zero_phase


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
SEQ_ID = "vid_1"
UBFC_ROOT = Path(r"D:\Data\UBFC\Dataset_3")
SEQ_DIR = UBFC_ROOT / SEQ_ID
GT_FILE = SEQ_DIR / "ground_truth_1.txt"
VIDEO_FILE = SEQ_DIR / f"{SEQ_ID}.avi"

RPPG_METHOD = "CHROM"  # "CHROM" or "POS"

WIN_LEN_SEC = 8.0   # window length in seconds
STRIDE_SEC = 1.0    # stride in seconds
T_START_MIN = 0.0   # earliest window start time (or 5.0 if you prefer)

# ROI strategy (same pattern as PURE)
ROI_MODE = "central"       # "central", "face_opencv", "face_mediapipe"
ROI_FRAC = 0.5             # only used for central ROI


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def _normalize(sig: np.ndarray) -> np.ndarray:
    sig = np.asarray(sig, dtype=np.float64)
    sig = sig - np.mean(sig)
    std = np.std(sig)
    if std > 0:
        sig = sig / std
    return sig


def estimate_phase_lag_sign_invariant(
    s_rppg: np.ndarray,
    s_ppg: np.ndarray,
    dt: float,
    max_lag_seconds: float = 2.0,
) -> float:
    """
    Same sign-invariant lag estimator as in PURE steps.
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


def fix_timestamps_inplace(t_s: np.ndarray, eps: float = 1e-6):
    t_s = np.asarray(t_s, dtype=np.float64)
    corrected_indices = []
    for i in range(1, len(t_s)):
        if t_s[i] <= t_s[i - 1]:
            t_s[i] = t_s[i - 1] + eps
            corrected_indices.append(i)
    return t_s, corrected_indices


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
# Main sliding-window analysis (modular version)
# ----------------------------------------------------------------------
def main():
    print(
        f"=== UBFC STEP 4 (modular): Sliding-window lag analysis "
        f"(win={WIN_LEN_SEC:.1f} s, stride={STRIDE_SEC:.1f} s) "
        f"for {SEQ_ID} [{RPPG_METHOD}], ROI={ROI_MODE} ==="
    )

    # 1) Load GT PPG and timestamps
    t_ppg_s, ppg_wave = load_ubfc_ground_truth(str(GT_FILE))
    print(f"Loaded GT: {len(ppg_wave)} PPG samples")

    t_ppg_s, corrected_idx = fix_timestamps_inplace(t_ppg_s, eps=1e-6)
    if corrected_idx:
        print(f"Corrected non-increasing timestamps at indices: {corrected_idx[:10]}")
        if len(corrected_idx) > 10:
            print("...(more corrections omitted)")

    # 2) Build UBFCFrameSource and ROI
    source = UBFCFrameSource(VIDEO_FILE)
    roi_extractor = make_roi(ROI_MODE, ROI_FRAC)

    # Global frame timeline (from source)
    t_frame_s_all = source.t_frame_s
    fps_nominal = source.fps
    print(f"Video info: {len(t_frame_s_all)} frames @ {fps_nominal:.3f} fps")

    # 3) Determine overlap between GT and video
    t_ppg_min, t_ppg_max = float(t_ppg_s[0]), float(t_ppg_s[-1])
    t_vid_min, t_vid_max = float(t_frame_s_all[0]), float(t_frame_s_all[-1])

    overlap_start = max(t_ppg_min, t_vid_min)
    overlap_end = min(t_ppg_max, t_vid_max)

    print(
        "\n--- Overlap range ---\n"
        f"GT range:    [{t_ppg_min:.3f}, {t_ppg_max:.3f}] s\n"
        f"Video range: [{t_vid_min:.3f}, {t_vid_max:.3f}] s\n"
        f"Overlap:     [{overlap_start:.3f}, {overlap_end:.3f}] s"
    )

    # 4) Build sliding windows inside the overlap
    t_start_min = max(T_START_MIN, overlap_start)
    t_start_max = overlap_end - WIN_LEN_SEC

    if t_start_max <= t_start_min:
        raise RuntimeError(
            f"Not enough duration for a {WIN_LEN_SEC:.1f} s window. "
            f"t_start_min={t_start_min:.3f}, t_start_max={t_start_max:.3f}"
        )

    start_times = np.arange(t_start_min, t_start_max + 1e-6, STRIDE_SEC)
    print("\nSliding window configuration:")
    print(f"  Window length: {WIN_LEN_SEC:.1f} s")
    print(f"  Stride:        {STRIDE_SEC:.1f} s")
    print(f"  t_start range: [{t_start_min:.1f}, {t_start_max:.1f}] s")
    print(f"  Number of windows: {len(start_times)}")

    # 5) Loop over windows and compute lag
    lags_sec = []
    lags_frames = []
    valid_windows = []

    method_label = RPPG_METHOD.upper()

    for idx, t_start in enumerate(start_times):
        t_end = t_start + WIN_LEN_SEC
        print(f"\n--- Window {idx} [{t_start:.1f}, {t_end:.1f}] s ---")

        # Extract RGB via the generic extractor (FrameSource + ROI)
        try:
            t_frame_win, rgb_ts_win = extract_rgb_timeseries(
                source=source,
                roi_extractor=roi_extractor,
                t_start=t_start,
                t_end=t_end,
            )
        except RuntimeError as e:
            print(f"Skipping window due to error: {e}")
            continue

        n_win_frames = len(t_frame_win)
        print(f"Frames in window: {n_win_frames}")
        if n_win_frames < 20:
            print("Window has too few frames; skipping.")
            continue

        # Frame-based dt and fs
        dt_frames = np.diff(t_frame_win)
        dt_mean = float(np.mean(dt_frames))
        fs_est = 1.0 / dt_mean
        print(f"Estimated frame rate in window: {fs_est:.3f} Hz")

        # Interpolate GT PPG at frame times in this window
        ppg_win = np.interp(t_frame_win, t_ppg_s, ppg_wave)

        # Band-pass filter PPG and rPPG
        ppg_filt = bandpass_zero_phase(ppg_win, fs=fs_est)

        if method_label == "CHROM":
            rppg_raw = rppg_chrom(rgb_ts_win)
        elif method_label == "POS":
            rppg_raw = rppg_pos(rgb_ts_win)
        else:
            raise ValueError(f"Unknown RPPG_METHOD '{RPPG_METHOD}', use 'CHROM' or 'POS'.")

        rppg_filt = bandpass_zero_phase(rppg_raw, fs=fs_est)

        # Normalize
        ppg_n = _normalize(ppg_filt)
        rppg_n = _normalize(rppg_filt)

        # Estimate lag
        lag_sec = estimate_phase_lag_sign_invariant(
            s_rppg=rppg_n,
            s_ppg=ppg_n,
            dt=dt_mean,
            max_lag_seconds=2.0,
        )
        lag_ms = lag_sec * 1000.0
        lag_fr = lag_sec * fs_est

        lags_sec.append(lag_sec)
        lags_frames.append(lag_fr)
        valid_windows.append((t_start, t_end))

        print(
            f"{method_label} lag (filtered): {lag_ms:.2f} ms "
            f"(≈ {lag_fr:.2f} frames; positive means {method_label} lags PPG)"
        )

    lags_sec = np.array(lags_sec, dtype=np.float64)
    lags_frames = np.array(lags_frames, dtype=np.float64)

    print("\n=== Summary of phase lags across sliding windows ===")
    print(f"Valid windows: {len(valid_windows)}")

    if len(lags_sec) == 0:
        print("No valid windows; nothing to plot.")
        return

    lags_ms = lags_sec * 1000.0
    mean_ms = float(lags_ms.mean())
    std_ms = float(lags_ms.std())
    mean_fr = float(lags_frames.mean())
    std_fr = float(lags_frames.std())

    print(f"\n{method_label} lag (ms): mean = {mean_ms:.2f}, std = {std_ms:.2f}")
    print(f"{method_label} lag (frames): mean = {mean_fr:.2f}, std = {std_fr:.2f}")

    print(f"\nFirst few windows:")
    for i, (w, lag_m, lag_f) in enumerate(zip(valid_windows, lags_ms, lags_frames)):
        if i >= 10:
            break
        print(
            f"  Window {i} [{w[0]:.1f}, {w[1]:.1f}] s: "
            f"{lag_m:.2f} ms, {lag_f:.2f} frames"
        )

    # 6) Bar plots: lag(ms) vs window index, lag(frames) vs window index
    num_win = len(valid_windows)
    x_idx = np.arange(num_win)

    # (a) lag(ms) vs window index
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.bar(x_idx, lags_ms, width=0.6, label=f"{method_label} lag (ms)")
    ax1.axhline(mean_ms, color="gray", linestyle="--",
                label=f"mean = {mean_ms:.1f} ms")
    ax1.set_title(f"{SEQ_ID} – {method_label} lag (ms) per window (ROI={ROI_MODE})")
    ax1.set_xlabel("Window index")
    ax1.set_ylabel("Lag (ms)")
    ax1.set_xticks(x_idx)
    ax1.grid(True, axis="y", linestyle=":")
    ax1.legend(loc="best")
    plt.tight_layout()
    plt.show()

    # (b) lag(frames) vs window index
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.bar(x_idx, lags_frames, width=0.6, label=f"{method_label} lag (frames)")
    ax2.axhline(mean_fr, color="gray", linestyle="--",
                label=f"mean = {mean_fr:.2f} frames")
    ax2.set_title(f"{SEQ_ID} – {method_label} lag (frames) per window (ROI={ROI_MODE})")
    ax2.set_xlabel("Window index")
    ax2.set_ylabel("Lag (frames)")
    ax2.set_xticks(x_idx)
    ax2.grid(True, axis="y", linestyle=":")
    ax2.legend(loc="best")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
