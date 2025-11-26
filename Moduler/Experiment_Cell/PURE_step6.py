# PURE_step8_sliding_lag_8s_1s.py
"""
STEP 8: Sliding-window constant-lag analysis with 8 s windows and 1 s stride.

This version is updated to use the new modular architecture:

- PUREFrameSource (dataset abstraction)
- RoiExtractor implementations (central, face-based, etc.)
- extract_rgb_timeseries (dataset-agnostic, ROI-agnostic)

Pipeline:
1) Load PPG and frame timestamps once.
2) Derive the valid time range from the frame timestamps.
3) Generate sliding windows [t_start, t_start + 8] with 1 s stride.
4) For each window:
   - extract RGB from frames via FrameSource + RoiExtractor,
   - compute rPPG using the selected method (POS or CHROM),
   - interpolate GT PPG at the frame times,
   - bandpass filter both rPPG and PPG,
   - estimate lag with sign-invariant cross-correlation.
5) Summarize lag (mean, std) and visualize.

No time shifting of signals; this step only analyses lag characteristics.
"""
import sys
from pathlib import Path
# Add project root (Project_rPPG) to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

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
# Configuration: rPPG method + ROI strategy
# ----------------------------------------------------------------------

# rPPG method: "POS" or "CHROM"
RPPG_METHOD = "CHROM"
# RPPG_METHOD = "POS"

# ROI strategy: "central", "face_opencv", "face_mediapipe"
ROI_MODE = "central"
ROI_FRAC = 0.5   # fraction for central ROI only


# ----------------------------------------------------------------------
# Helper: sign-invariant phase lag
# ----------------------------------------------------------------------

def estimate_phase_lag_sign_invariant(
    s_rppg: np.ndarray,
    s_ppg: np.ndarray,
    dt: float,
    max_lag_seconds: float = 2.0,
) -> float:
    """
    Estimate phase lag between rPPG and GT PPG using cross-correlation
    with sign invariance (maximize |corr|).

    Parameters
    ----------
    s_rppg : np.ndarray
        rPPG signal, shape (T,).
    s_ppg : np.ndarray
        GT PPG signal, shape (T,).
    dt : float
        Sampling interval in seconds.
    max_lag_seconds : float
        Maximum lag considered in either direction.

    Returns
    -------
    lag_sec : float
        Estimated lag in seconds (positive means rPPG lags PPG).
    """
    s1 = np.asarray(s_rppg, dtype=np.float32)
    s2 = np.asarray(s_ppg, dtype=np.float32)
    assert s1.shape == s2.shape, "Signals must have the same length."

    # Mean removal and unit-variance normalization
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

    # Restrict lag search around zero
    mask = (lags >= -max_lag_samples) & (lags <= max_lag_samples)
    corr_win = corr[mask]
    lags_win = lags[mask]

    # Sign-invariant: maximize absolute correlation
    abs_corr_win = np.abs(corr_win)
    best_idx = np.argmax(abs_corr_win)
    best_lag_samples = lags_win[best_idx]
    lag_sec = float(best_lag_samples * dt)

    return lag_sec


def _normalize(sig: np.ndarray) -> np.ndarray:
    """
    Normalize a signal to zero mean and unit variance (if std > 0).
    """
    sig = np.asarray(sig, dtype=np.float32)
    sig = sig - np.mean(sig)
    std = np.std(sig)
    if std == 0:
        return sig
    return sig / std


# ----------------------------------------------------------------------
# Helper: use new architecture to extract RGB in a window
# ----------------------------------------------------------------------

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

    Parameters
    ----------
    seq_id : str
        PURE sequence ID.
    t_start, t_end : float
        Window in seconds (relative to first frame).
    roi_mode : {"central", "face_opencv", "face_mediapipe"}
        Which ROI strategy to use.
    roi_frac : float
        Fraction for CentralRoiExtractor (ignored for face-based ROIs).

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

    print(
        f"=== STEP 8: Sliding-window lag analysis (8 s, 1 s stride) "
        f"for {seq_id} [{method_label}], ROI={ROI_MODE} ==="
    )

    # Window configuration
    win_len_sec = 8.0   # 8-second window
    stride_sec = 1.0    # 1-second stride (hop)

    # 1) Load JSON PPG once
    data = load_pure_json(seq_id)
    t_ppg_ns, wave, t_vid_ns_json, hr_dev = extract_streams_from_pure_json(data)
    t0_ns = t_vid_ns_json[0]
    t_ppg_s = (t_ppg_ns - t0_ns) * 1e-9

    # 2) Load frame timestamps from PURE images (for global time range)
    image_files, t_img_ns = load_pure_image_files_and_timestamps(seq_id)
    t_frame_s_all = (t_img_ns - t_img_ns[0]) * 1e-9

    t_min = float(t_frame_s_all[0])
    t_max = float(t_frame_s_all[-1])

    # 3) Define starting times for sliding windows
    t_start_min = 5.0
    t_start_max = t_max - win_len_sec

    if t_start_max <= t_start_min:
        raise RuntimeError(
            f"Not enough duration for an {win_len_sec} s window. "
            f"t_min={t_min:.2f}, t_max={t_max:.2f}"
        )

    start_times = np.arange(t_start_min, t_start_max + 1e-6, stride_sec)

    print("\nSliding window configuration:")
    print(f"  Window length: {win_len_sec:.1f} s")
    print(f"  Stride:        {stride_sec:.1f} s")
    print(f"  t_start range: [{t_start_min:.1f}, {t_start_max:.1f}] s")
    print(f"  Number of windows: {len(start_times)}")

    lags_sec = []
    valid_windows = []
    fps_list = []

    # 4) Loop over sliding windows
    for idx, t_start in enumerate(start_times):
        t_end = t_start + win_len_sec
        print(f"\n--- Window {idx} [{t_start:.1f}, {t_end:.1f}] s ---")

        # Extract RGB using new architecture (FrameSource + ROI)
        try:
            t_frame_s_win, rgb_ts = extract_pure_rgb_timeseries_modular(
                seq_id=seq_id,
                t_start=t_start,
                t_end=t_end,
                roi_mode=ROI_MODE,
                roi_frac=ROI_FRAC,
            )
        except RuntimeError as e:
            print(f"Skipping window due to error: {e}")
            continue

        print(f"Frames in window: {len(t_frame_s_win)}")
        print(f"RGB time series shape: {rgb_ts.shape}")

        if len(t_frame_s_win) < 20:
            print("Window has too few frames; skipping.")
            continue

        # Time step and frame rate
        dt_frames = np.diff(t_frame_s_win)
        dt_mean = float(np.mean(dt_frames))
        fps_est = 1.0 / dt_mean
        print(f"Estimated frame rate in window: {fps_est:.3f} Hz")

        # 5) Interpolate GT PPG at frame times
        ppg_gt_at_frames = pure_align_ppg_to_frame_times(
            t_ppg_s=t_ppg_s,
            wave=wave,
            t_vid_s=t_frame_s_win,
        )

        # 6) Bandpass filter GT PPG
        fs = fps_est
        ppg_gt_filt = bandpass_zero_phase(ppg_gt_at_frames, fs=fs)

        # 7) Compute rPPG using selected method (POS or CHROM)
        if method_label == "POS":
            rppg_raw = rppg_pos(rgb_ts)
        elif method_label == "CHROM":
            rppg_raw = rppg_chrom(rgb_ts)
        else:
            raise ValueError(
                f"Unknown RPPG_METHOD '{RPPG_METHOD}', use 'POS' or 'CHROM'."
            )

        rppg_filt = bandpass_zero_phase(rppg_raw, fs=fs)

        # 8) Normalize
        ppg_gt_n = _normalize(ppg_gt_filt)
        rppg_n = _normalize(rppg_filt)

        # 9) Estimate lag (sign-invariant)
        lag_sec = estimate_phase_lag_sign_invariant(
            s_rppg=rppg_n,
            s_ppg=ppg_gt_n,
            dt=dt_mean,
            max_lag_seconds=2.0,
        )

        lags_sec.append(lag_sec)
        valid_windows.append((t_start, t_end))
        fps_list.append(fps_est)

        print(
            f"{method_label} lag (filtered): {lag_sec*1000:.2f} ms "
            f"(positive means {method_label} lags PPG)"
        )

    # Convert to arrays
    lags_sec = np.array(lags_sec, dtype=np.float32)
    fps_arr = np.array(fps_list, dtype=np.float32)

    print("\n=== Summary of filtered phase lags across sliding windows ===")
    print(f"Valid windows: {len(valid_windows)}")

    if len(lags_sec) > 0:
        print(f"\n{method_label} lags (ms) for first few windows:")
        for i, (w, lag_s) in enumerate(zip(valid_windows, lags_sec)):
            if i >= 10:
                break
            print(
                f"  Window {i} [{w[0]:.1f}, {w[1]:.1f}] s: "
                f"{lag_s * 1000:.2f} ms"
            )
        print(
            f"\n{method_label} mean lag: {lags_sec.mean() * 1000:.2f} ms, "
            f"std: {lags_sec.std() * 1000:.2f} ms"
        )

    # ------------------------------------------------------------------
    # Lag statistics and visualizations (single method)
    # ------------------------------------------------------------------
    if len(valid_windows) > 0:
        # Window centers in seconds
        win_centers = np.array(
            [(w[0] + w[1]) / 2.0 for w in valid_windows], dtype=np.float32
        )

        # Lag in ms and frames
        lags_ms = lags_sec * 1000.0
        lags_frames = lags_sec * fps_arr

        mean_ms = float(lags_ms.mean())
        std_ms = float(lags_ms.std())
        mean_fr = float(lags_frames.mean())
        std_fr = float(lags_frames.std())

        print(
            f"\n{method_label} lag (frames): mean = {mean_fr:.2f}, "
            f"std = {std_fr:.2f}"
        )

        # 1) Line plot: lag(ms) vs window center + mean/std
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            win_centers,
            lags_ms,
            marker="o",
            linestyle="-",
            label=f"{method_label} lag (ms)",
        )

        ax.axhline(
            mean_ms,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            label=f"mean = {mean_ms:.1f} ms",
        )
        ax.axhline(
            mean_ms + std_ms,
            color="gray",
            linestyle=":",
            linewidth=1.0,
            label=f"mean ± std ({std_ms:.1f} ms)",
        )
        ax.axhline(
            mean_ms - std_ms,
            color="gray",
            linestyle=":",
            linewidth=1.0,
        )

        ax.set_title(
            f"Sliding-window phase lag ({method_label}, seq {seq_id}, "
            f"win={win_len_sec:.1f}s, stride={stride_sec:.1f}s, ROI={ROI_MODE})"
        )
        ax.set_xlabel("Window center time (s)")
        ax.set_ylabel("Lag (ms)")
        ax.grid(True)
        ax.legend(loc="best")
        plt.tight_layout()
        plt.show()

        # 2) Boxplots: lag (ms) and lag (frames)
        fig_box, axes_box = plt.subplots(1, 2, figsize=(12, 5))

        axes_box[0].boxplot(
            [lags_ms],
            labels=[method_label],
            showmeans=True,
        )
        axes_box[0].set_title(f"{method_label} lag distribution (ms)")
        axes_box[0].set_ylabel("Lag (ms)")
        axes_box[0].grid(True, axis="y", linestyle=":")

        axes_box[1].boxplot(
            [lags_frames],
            labels=[method_label],
            showmeans=True,
        )
        axes_box[1].set_title(f"{method_label} lag distribution (frames)")
        axes_box[1].set_ylabel("Lag (frames)")
        axes_box[1].grid(True, axis="y", linestyle=":")

        plt.tight_layout()
        plt.show()

        # 3) Bar plots: lag(ms) vs window index, lag(frames) vs window index
        num_win = len(valid_windows)
        x_idx = np.arange(num_win)
        width = 0.6

        # (a) lag(ms) vs window
        fig_bar1, ax_bar1 = plt.subplots(figsize=(12, 5))
        ax_bar1.bar(
            x_idx,
            lags_ms,
            width=width,
            label=f"{method_label} lag (ms)",
        )

        ax_bar1.set_title(f"{method_label} lag (ms) per window")
        ax_bar1.set_xlabel("Window index")
        ax_bar1.set_ylabel("Lag (ms)")
        ax_bar1.set_xticks(x_idx)
        ax_bar1.grid(True, axis="y", linestyle=":")
        ax_bar1.legend(loc="best")
        plt.tight_layout()
        plt.show()

        # (b) lag(frames) vs window
        fig_bar2, ax_bar2 = plt.subplots(figsize=(12, 5))
        ax_bar2.bar(
            x_idx,
            lags_frames,
            width=width,
            label=f"{method_label} lag (frames)",
        )

        ax_bar2.set_title(f"{method_label} lag (frames) per window")
        ax_bar2.set_xlabel("Window index")
        ax_bar2.set_ylabel("Lag (frames)")
        ax_bar2.set_xticks(x_idx)
        ax_bar2.grid(True, axis="y", linestyle=":")
        ax_bar2.legend(loc="best")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
