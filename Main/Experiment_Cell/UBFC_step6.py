# UBFC_step10_full_dataset_constant_lag.py
"""
UBFC STEP 10: Full-Dataset Constant-Lag Evaluation
--------------------------------------------------
Aggregate phase-lag results across all UBFC sequences (vid_*) for POS/CHROM.

For each sequence:
    - Load GT PPG + timestamps (with fixes)
    - Build UBFCFrameSource (video frames + timestamps)
    - Determine overlap between GT and video
    - Perform sliding-window (8 s, 1 s stride) lag analysis
    - Record mean and std lag (ms, frames)

Outputs:
    - Summary CSV: Results/ubfc_lag_summary_all.csv
    - Boxplots + violin plots of per-sequence mean lag (ms) for POS/CHROM
"""

import sys
from pathlib import Path
# Add project root (Project_rPPG) to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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



# ------------------- Parameters -------------------
UBFC_ROOT = Path(r"D:\Data\UBFC\Dataset_3")
RESULT_DIR = Path("Results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

WIN_LEN_SEC = 8.0
STRIDE_SEC = 1.0
T_START_MIN = 0.0

# Evaluate one or both:
ALL_METHOD = ["CHROM"]  # or ["POS", "CHROM"]

# ROI config
ROI_MODE = "central"   # "central", "face_opencv", "face_mediapipe"
ROI_FRAC = 0.5         # only used for central ROI


# ------------------- Utilities -------------------
def _normalize(sig: np.ndarray) -> np.ndarray:
    sig = np.asarray(sig, np.float32)
    sig -= np.mean(sig)
    std = np.std(sig)
    return sig if std == 0 else sig / std


def estimate_phase_lag_sign_invariant(
    s_rppg: np.ndarray,
    s_ppg: np.ndarray,
    dt: float,
    max_lag_seconds: float = 2.0,
) -> float:
    """
    Sign-invariant lag estimator (same as in other steps).
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

    best_idx = int(np.argmax(np.abs(corr_win)))
    best_lag_samples = int(lags_win[best_idx])

    return float(best_lag_samples * dt)


def make_roi(roi_mode: str, roi_frac: float):
    if roi_mode == "central":
        return CentralRoiExtractor(frac=roi_frac)
    elif roi_mode == "face_opencv":
        return OpenCVFaceBoxRoi()
    elif roi_mode == "face_mediapipe":
        return MediaPipeFaceRegionsRoi()
    else:
        raise ValueError(f"Unknown ROI_MODE: {roi_mode}")


def analyze_sequence(seq_dir: Path, method: str):
    """
    Return arrays of lags (ms, frames) for one UBFC sequence and one method.

    Parameters
    ----------
    seq_dir : Path
        Folder like .../Dataset_3/vid_1 containing GT + video.
    method : {"POS", "CHROM"}
    """
    txt_files = list(seq_dir.glob("*.txt"))
    avi_files = list(seq_dir.glob("*.avi"))

    if len(txt_files) != 1:
        raise RuntimeError(f"[{seq_dir.name}] Expected exactly 1 .txt file, found: {txt_files}")

    if len(avi_files) != 1:
        raise RuntimeError(f"[{seq_dir.name}] Expected exactly 1 .avi file, found: {avi_files}")

    gt_file = txt_files[0]
    vid_file = avi_files[0]

    if not gt_file.exists() or not vid_file.exists():
        print(f"[{seq_dir} Missing GT or video; skipping.")
        return None, None

    # Load GT
    try:
        t_ppg_s, ppg_wave, _ = load_ubfc_gt_and_fix(gt_file)
    except Exception as e:
        print(f"[{seq_id}] GT load error: {e}")
        return None, None

    # Build FrameSource + ROI
    try:
        source = UBFCFrameSource(vid_file)
    except Exception as e:
        print(f"[{seq_id}] video load error: {e}")
        return None, None

    roi_extractor = make_roi(ROI_MODE, ROI_FRAC)
    t_frame_s_all = source.t_frame_s

    # Overlap
    t_ppg_min, t_ppg_max = float(t_ppg_s[0]), float(t_ppg_s[-1])
    t_vid_min, t_vid_max = float(t_frame_s_all[0]), float(t_frame_s_all[-1])

    overlap_start = max(t_ppg_min, t_vid_min)
    overlap_end = min(t_ppg_max, t_vid_max)

    t_start_min = max(T_START_MIN, overlap_start)
    t_start_max = overlap_end - WIN_LEN_SEC

    if t_start_max <= t_start_min:
        # Not enough duration
        return None, None

    start_times = np.arange(t_start_min, t_start_max + 1e-6, STRIDE_SEC)

    lags_ms = []
    lags_frames = []

    method_label = method.upper()

    for t_start in start_times:
        t_end = t_start + WIN_LEN_SEC

        # Extract RGB
        try:
            t_frame_win, rgb_ts_win = extract_rgb_timeseries(
                source=source,
                roi_extractor=roi_extractor,
                t_start=t_start,
                t_end=t_end,
            )
        except RuntimeError:
            continue

        if len(t_frame_win) < 20:
            continue

        dt_frames = np.diff(t_frame_win)
        dt_mean = float(np.mean(dt_frames))
        fps_est = 1.0 / dt_mean
        fs = fps_est

        # Align GT at frame times
        ppg_win = ubfc_align_ppg_to_frame_times(
            t_ppg_s=t_ppg_s,
            ppg_wave=ppg_wave,
            t_frame_s=t_frame_win,
        )

        # Filter
        ppg_filt = bandpass_zero_phase(ppg_win, fs=fs)

        if method_label == "POS":
            rppg_raw = rppg_pos(rgb_ts_win)
        else:
            rppg_raw = rppg_chrom(rgb_ts_win)

        rppg_filt = bandpass_zero_phase(rppg_raw, fs=fs)

        # Normalize
        ppg_n = _normalize(ppg_filt)
        rppg_n = _normalize(rppg_filt)

        # Lag
        lag_sec = estimate_phase_lag_sign_invariant(
            s_rppg=rppg_n,
            s_ppg=ppg_n,
            dt=dt_mean,
            max_lag_seconds=2.0,
        )
        lags_ms.append(lag_sec * 1000.0)
        lags_frames.append(lag_sec * fps_est)

    if len(lags_ms) == 0:
        return None, None

    return np.array(lags_ms, dtype=np.float32), np.array(lags_frames, dtype=np.float32)


# ------------------- Main -------------------
def main():
    # Find all UBFC sequences (folders like vid_1, vid_2, ...)
    seq_dirs = sorted(
        d for d in UBFC_ROOT.iterdir()
        if d.is_dir() and d.name.lower().startswith("vid")
    )

    print(f"Found {len(seq_dirs)} UBFC sequences in {UBFC_ROOT}.\n")

    summary = []

    for method in ALL_METHOD:
        print(f"=== Evaluating {method} across all UBFC sequences (ROI={ROI_MODE}) ===")
        for seq_dir in tqdm(seq_dirs):
            seq_id = seq_dir.name
            lags_ms, lags_fr = analyze_sequence(seq_dir, method)
            if lags_ms is None or len(lags_ms) == 0:
                continue

            summary.append({
                "seq": seq_id,
                "method": method.upper(),
                "n_win": int(len(lags_ms)),
                "lag_mean_ms": float(np.mean(lags_ms)),
                "lag_std_ms": float(np.std(lags_ms)),
                "lag_mean_frames": float(np.mean(lags_fr)),
                "lag_std_frames": float(np.std(lags_fr)),
            })

    if len(summary) == 0:
        print("No valid sequences/windows found. Exiting.")
        return

    # Save CSV
    csv_path = RESULT_DIR / "ubfc_lag_summary_all.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nSummary saved to {csv_path}")

    # Aggregate for visualization
    pos_vals = [s["lag_mean_ms"] for s in summary if s["method"] == "POS"]
    chrom_vals = [s["lag_mean_ms"] for s in summary if s["method"] == "CHROM"]

    methods_present = []
    data_for_plots = []

    if len(pos_vals) > 0:
        methods_present.append("POS")
        data_for_plots.append(pos_vals)
    if len(chrom_vals) > 0:
        methods_present.append("CHROM")
        data_for_plots.append(chrom_vals)

    if len(data_for_plots) == 0:
        print("No lag values to plot.")
        return

    # Boxplots
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data_for_plots, labels=methods_present, showmeans=True)
    ax.set_ylabel("Sequence-wise mean lag (ms)")
    ax.set_title("UBFC: Mean lag distribution per sequence")
    ax.grid(True, axis="y", linestyle=":")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "ubfc_boxplot_mean_lag_ms.png", dpi=200)
    plt.show()

    # Violin plots
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.violinplot(data_for_plots, showmeans=True, showmedians=False)
    ax2.set_xticks(np.arange(1, len(methods_present) + 1))
    ax2.set_xticklabels(methods_present)
    ax2.set_ylabel("Sequence-wise mean lag (ms)")
    ax2.set_title("UBFC: Distribution of lags across dataset")
    ax2.grid(True, axis="y", linestyle=":")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "ubfc_violin_lag_ms.png", dpi=200)
    plt.show()

    # Print overall
    print("\n=== Global UBFC Results ===")
    if len(pos_vals) > 0:
        print(
            f"POS:   mean = {np.mean(pos_vals):.2f} ms, "
            f"std = {np.std(pos_vals):.2f} ms, n={len(pos_vals)}"
        )
    if len(chrom_vals) > 0:
        print(
            f"CHROM: mean = {np.mean(chrom_vals):.2f} ms, "
            f"std = {np.std(chrom_vals):.2f} ms, n={len(chrom_vals)}"
        )


if __name__ == "__main__":
    main()
