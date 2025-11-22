# PURE_step9_inspect_single_window_consistent_plot_and_save.py
"""
STEP 9 (Consistent + Plot + Save)
---------------------------------
Visualize pointwise amplitude & phase differences between GT and rPPG (POS/CHROM)
for a single sliding window.

This version is updated to use the modular architecture:
- PUREFrameSource (dataset abstraction)
- RoiExtractor (configurable ROI strategies)
- extract_rgb_timeseries (dataset-agnostic, ROI-agnostic)
"""

import sys
from pathlib import Path
# Add project root (Project_rPPG) to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# ------------------ Data I/O and alignment (PURE-specific) ------------------

from Main.Data_Read_Engine import (
    DEFAULT_SEQ_ID,
    load_pure_json,
    extract_streams_from_pure_json,
    pure_align_ppg_to_frame_times,
    load_pure_image_files_and_timestamps,
)

# ------------------ New architecture: dataset + ROI + extractor -------------

from Main.Signal_Processing_Engine.pure_dataset import PUREFrameSource
from Main.Signal_Processing_Engine.roi_central import CentralRoiExtractor
from Main.Signal_Processing_Engine.roi_face_opencv import OpenCVFaceBoxRoi
from Main.Signal_Processing_Engine.roi_face_mediapipe import MediaPipeFaceRegionsRoi
from Main.Signal_Processing_Engine.rgb_extractor import extract_rgb_timeseries

# ------------------ rPPG methods and filtering ------------------------------

from Main.rPPG_Algorithm_Cell.rppg_methods import rppg_pos, rppg_chrom
from Main.rPPG_Algorithm_Cell.rppg_filtering import bandpass_zero_phase


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
WIN_LEN_SEC = 8.0
STRIDE_SEC = 1.0
T_START_MIN = 5.0
WINDOW_INDEX = 0          # choose window index (0, 8, 15, 25, ...)
RPPG_METHOD = "CHROM"     # "POS" or "CHROM"
OUTPUT_DIR = "Figures/step9"

# ROI configuration (same idea as STEP 8)
ROI_MODE = "central"      # "central", "face_opencv", "face_mediapipe"
ROI_FRAC = 0.5            # only used for central ROI


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def estimate_phase_lag_sign_invariant(s_rppg, s_ppg, dt, max_lag_seconds=2.0):
    """Identical lag computation as Step 8 (sign-invariant)."""
    s1 = np.asarray(s_rppg, dtype=np.float32)
    s2 = np.asarray(s_ppg, dtype=np.float32)
    assert s1.shape == s2.shape, "Signals must have same length"
    s1 -= np.mean(s1)
    s2 -= np.mean(s2)
    s1_std, s2_std = np.std(s1), np.std(s2)
    if s1_std == 0 or s2_std == 0:
        return 0.0
    s1 /= s1_std
    s2 /= s2_std
    T = len(s1)
    max_lag_samples = int(min(max_lag_seconds / dt, T - 1))
    corr = np.correlate(s1, s2, mode="full")
    lags = np.arange(-T + 1, T)
    mask = (lags >= -max_lag_samples) & (lags <= max_lag_samples)
    corr_win = corr[mask]
    lags_win = lags[mask]
    abs_corr_win = np.abs(corr_win)
    best_idx = np.argmax(abs_corr_win)
    best_lag_samples = lags_win[best_idx]
    return float(best_lag_samples * dt)


def _normalize(sig: np.ndarray) -> np.ndarray:
    sig = np.asarray(sig, dtype=np.float32)
    sig -= np.mean(sig)
    std = np.std(sig)
    return sig if std == 0 else sig / std


def extract_pure_rgb_timeseries_modular(
    seq_id: str,
    t_start: float,
    t_end: float,
    roi_mode: str = "central",
    roi_frac: float = 0.5,
):
    """
    PURE RGB extraction using the modular architecture:
    - PUREFrameSource (dataset abstraction)
    - RoiExtractor (selectable strategy)
    - extract_rgb_timeseries (generic extractor)

    Returns
    -------
    t_frame_s_win : np.ndarray
        Frame timestamps, shape (T,).
    rgb_ts : np.ndarray
        Mean RGB per frame, shape (T, 3), float32, RGB order.
    """
    source = PUREFrameSource(seq_id)

    if roi_mode == "central":
        roi = CentralRoiExtractor(frac=roi_frac)
    elif roi_mode == "face_opencv":
        roi = OpenCVFaceBoxRoi()
    elif roi_mode == "face_mediapipe":
        roi = MediaPipeFaceRegionsRoi()
    else:
        raise ValueError(f"Unknown roi_mode: {roi_mode}")

    t_frame_s_win, rgb_ts = extract_rgb_timeseries(
        source=source,
        roi_extractor=roi,
        t_start=t_start,
        t_end=t_end,
    )
    return t_frame_s_win, rgb_ts


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    seq_id = DEFAULT_SEQ_ID
    method_label = RPPG_METHOD.upper()
    print(f"=== STEP 9 (Plot + Save) — {method_label} vs GT PPG, ROI={ROI_MODE} ===")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Load JSON + PPG
    data = load_pure_json(seq_id)
    t_ppg_ns, wave, t_vid_ns_json, hr_dev = extract_streams_from_pure_json(data)

    # 2) Load frame timestamps from images to define sliding windows
    image_files, t_img_ns = load_pure_image_files_and_timestamps(seq_id)
    t_frame_s_all = (t_img_ns - t_img_ns[0]) * 1e-9

    # 3) Select window based on global time range
    t_start_max = t_frame_s_all[-1] - WIN_LEN_SEC
    start_times = np.arange(T_START_MIN, t_start_max + 1e-6, STRIDE_SEC)

    if WINDOW_INDEX < 0 or WINDOW_INDEX >= len(start_times):
        raise IndexError(
            f"WINDOW_INDEX {WINDOW_INDEX} is out of range (0..{len(start_times)-1})"
        )

    t_start = float(start_times[WINDOW_INDEX])
    t_end = t_start + WIN_LEN_SEC
    print(f"Window {WINDOW_INDEX}: {t_start:.1f}–{t_end:.1f} s")

    # 4) Extract RGB frames using new architecture
    t_frame_s_win, rgb_ts = extract_pure_rgb_timeseries_modular(
        seq_id=seq_id,
        t_start=t_start,
        t_end=t_end,
        roi_mode=ROI_MODE,
        roi_frac=ROI_FRAC,
    )

    dt_mean = float(np.mean(np.diff(t_frame_s_win)))
    fps_est = 1.0 / dt_mean
    fs = fps_est
    print(f"Frames in window: {len(t_frame_s_win)}, fps ≈ {fps_est:.3f} Hz")

    # 5) Align GT PPG to frame times
    t_ppg_s = (t_ppg_ns - t_vid_ns_json[0]) * 1e-9
    ppg_gt = pure_align_ppg_to_frame_times(
        t_ppg_s=t_ppg_s,
        wave=wave,
        t_vid_s=t_frame_s_win,
    )

    # 6) Compute rPPG
    if method_label == "POS":
        rppg_raw = rppg_pos(rgb_ts)
    else:
        rppg_raw = rppg_chrom(rgb_ts)

    # 7) Filter and normalize
    ppg_filt = _normalize(bandpass_zero_phase(ppg_gt, fs))
    rppg_filt = _normalize(bandpass_zero_phase(rppg_raw, fs))

    # 8) Lag estimation
    lag_sec = estimate_phase_lag_sign_invariant(rppg_filt, ppg_filt, dt_mean)
    lag_ms = lag_sec * 1000.0
    lag_frames = lag_sec * fps_est
    print(f"Estimated lag ({method_label}): {lag_ms:.2f} ms ≈ {lag_frames:.2f} frames")

    # 9) Phase visualization (Hilbert)
    phase_gt = np.unwrap(np.angle(hilbert(ppg_filt)))
    phase_rppg = np.unwrap(np.angle(hilbert(rppg_filt)))
    phase_diff_deg = np.degrees(np.angle(np.exp(1j * (phase_rppg - phase_gt))))

    # 10) Plot and save
    t = t_frame_s_win
    fig, ax = plt.subplots(figsize=(12, 6))

    gt_line, = ax.plot(
        t, ppg_filt,
        color="tab:blue",
        marker="o",
        markersize=4,
        label="GT PPG",
    )
    rppg_line, = ax.plot(
        t, rppg_filt,
        color="tab:red",
        marker="x",
        markersize=4,
        label=f"{method_label} rPPG",
    )

    # Amplitude connectors (sparse)
    step = max(1, len(t) // 80)
    for i in range(0, len(t), step):
        ax.plot(
            [t[i], t[i]],
            [ppg_filt[i], rppg_filt[i]],
            color="gray",
            alpha=0.5,
            linewidth=0.8,
        )

    # Phase difference overlay
    sc = ax.scatter(
        t,
        (ppg_filt + rppg_filt) / 2,
        c=phase_diff_deg,
        cmap="coolwarm",
        s=35,
        edgecolors="k",
        label="Phase diff (°)",
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Phase difference (degrees)")

    ax.set_title(
        f"{method_label} vs GT PPG — Pointwise amplitude & phase differences\n"
        f"Window [{t_start:.1f}, {t_end:.1f}] s | "
        f"Lag ≈ {lag_ms:.1f} ms ({lag_frames:.2f} frames), ROI={ROI_MODE}"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized amplitude")
    ax.grid(True)

    legend_text = f"Mean lag ≈ {lag_ms:.1f} ms  (≈ {lag_frames:.2f} frames)"
    ax.legend(handles=[gt_line, rppg_line, sc], title=legend_text, loc="best")

    plt.tight_layout()

    # Save and display
    out_name = (
        f"seq{seq_id}_{method_label}_ROI-{ROI_MODE}_"
        f"win{WINDOW_INDEX:02d}_{lag_ms:.0f}ms.png"
    )
    out_path = os.path.join(OUTPUT_DIR, out_name)
    plt.savefig(out_path, dpi=200)
    print(f"Figure saved to: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
