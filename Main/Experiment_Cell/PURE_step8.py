"""
STEP 10: Full-Dataset Constant-Lag Evaluation
---------------------------------------------
Aggregate phase-lag results across all PURE sequences for POS and CHROM.

For each sequence:
    - Load JSON and frame timestamps
    - Perform sliding-window (8 s window, 1 s stride) lag analysis
    - Record mean and std lag (ms, frames)

Outputs:
    - Summary CSV: Results/lag_summary_all.csv
    - Boxplots, violin plots for POS/CHROM

This version uses the modular architecture:
- Main.Data_Read_Engine for PURE I/O and alignment
- Main.Signal_Processing_Engine for dataset/ROI/rPPG/filtering
"""
import sys
from pathlib import Path
# Add project root (Project_rPPG) to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os, glob, csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ------------------- PURE data I/O & alignment -------------------

from Main.Data_Read_Engine import (
    load_pure_json,
    extract_streams_from_pure_json,
    pure_align_ppg_to_frame_times,
    load_pure_image_files_and_timestamps,
)

# ------------------- New architecture: dataset + ROI + extractor -------------------

from Main.Signal_Processing_Engine.pure_dataset import PUREFrameSource
from Main.Signal_Processing_Engine.roi_central import CentralRoiExtractor
from Main.Signal_Processing_Engine.roi_face_opencv import OpenCVFaceBoxRoi
from Main.Signal_Processing_Engine.roi_face_mediapipe import MediaPipeFaceRegionsRoi
from Main.Signal_Processing_Engine.rgb_extractor import extract_rgb_timeseries

# ------------------- rPPG methods & filtering -------------------

from Main.rPPG_Algorithm_Cell.rppg_methods import rppg_pos, rppg_chrom
from Main.rPPG_Algorithm_Cell.rppg_filtering import bandpass_zero_phase



# ------------------- Parameters -------------------
WIN_LEN_SEC = 8.0
STRIDE_SEC = 1.0
T_START_MIN = 5.0
RESULT_DIR = "Results"
ALL_METHOD = ["CHROM"]  # or ["POS", "CHROM"]

# ROI configuration (same for all sequences)
ROI_MODE = "central"     # "central", "face_opencv", "face_mediapipe"
ROI_FRAC = 0.5           # only used for central ROI

os.makedirs(RESULT_DIR, exist_ok=True)


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
    Same sign-invariant lag estimator as Steps 8 & 9.
    """
    s1, s2 = np.asarray(s_rppg, np.float32), np.asarray(s_ppg, np.float32)
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
    corr_win, lags_win = corr[mask], lags[mask]

    best_idx = np.argmax(np.abs(corr_win))
    return float(lags_win[best_idx] * dt)


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


def analyze_sequence(seq_id: str, method: str):
    """
    Return arrays of lags (ms, frames) for one sequence and one method.

    Parameters
    ----------
    seq_id : str
        PURE sequence ID (e.g. "01-01").
    method : {"POS", "CHROM"}

    Returns
    -------
    lags_ms : np.ndarray
    lags_frames : np.ndarray
    """
    try:
        data = load_pure_json(seq_id)
        t_ppg_ns, wave, t_vid_ns_json, _ = extract_streams_from_pure_pjson(data)
    except NameError:
        # in case the function is named extract_streams_from_pure_json
        data = load_pure_json(seq_id)
        t_ppg_ns, wave, t_vid_ns_json, _ = extract_streams_from_pure_json(data)
    except Exception as e:
        print(f"[{seq_id}] JSON load error: {e}")
        return None, None

    try:
        image_files, t_img_ns = load_pure_image_files_and_timestamps(seq_id)
    except Exception as e:
        print(f"[{seq_id}] image I/O error: {e}")
        return None, None

    t_frame_s_all = (t_img_ns - t_img_ns[0]) * 1e-9

    # Sliding windows
    t_start_max = t_frame_s_all[-1] - WIN_LEN_SEC
    if t_start_max <= T_START_MIN:
        # not enough duration
        return None, None

    start_times = np.arange(T_START_MIN, t_start_max + 1e-6, STRIDE_SEC)

    lags_ms, lags_frames = [], []

    for t_start in start_times:
        t_end = t_start + WIN_LEN_SEC

        # Extract RGB via modular architecture
        try:
            t_frame_s_win, rgb_ts = extract_pure_rgb_timeseries_modular(
                seq_id=seq_id,
                t_start=t_start,
                t_end=t_end,
                roi_mode=ROI_MODE,
                roi_frac=ROI_FRAC,
            )
        except Exception:
            continue

        if len(t_frame_s_win) < 20:
            continue

        dt_mean = float(np.mean(np.diff(t_frame_s_win)))
        fps_est = 1.0 / dt_mean
        fs = fps_est

        # Align GT PPG to frame times
        t_ppg_s = (t_ppg_ns - t_vid_ns_json[0]) * 1e-9
        ppg_gt = pure_align_ppg_to_frame_times(
            t_ppg_s=t_ppg_s,
            wave=wave,
            t_vid_s=t_frame_s_win,
        )

        # rPPG method choice
        method_label = method.upper()
        if method_label == "POS":
            rppg_raw = rppg_pos(rgb_ts)
        else:
            rppg_raw = rppg_chrom(rgb_ts)

        ppg_filt = _normalize(bandpass_zero_phase(ppg_gt, fs))
        rppg_filt = _normalize(bandpass_zero_phase(rppg_raw, fs))

        lag_sec = estimate_phase_lag_sign_invariant(rppg_filt, ppg_filt, dt_mean)
        lags_ms.append(lag_sec * 1000.0)
        lags_frames.append(lag_sec * fps_est)

    if len(lags_ms) == 0:
        return None, None

    return np.array(lags_ms, dtype=np.float32), np.array(lags_frames, dtype=np.float32)


# ------------------- Main -------------------
def main():
    # You kept a hard-coded json_dir; we only use it to infer seq_ids.
    json_dir = r"D:\Data\MAIN_PURE\PURE\ALL\ALL"
    seq_paths = glob.glob(os.path.join(json_dir, "*.json"))
    seq_ids = sorted(
        [os.path.splitext(os.path.basename(p))[0] for p in seq_paths]
    )

    summary = []
    print(f"Found {len(seq_ids)} sequences.\n")

    for method in ALL_METHOD:
        print(f"=== Evaluating {method} across all sequences (ROI={ROI_MODE}) ===")
        for seq_id in tqdm(seq_ids):
            lags_ms, lags_fr = analyze_sequence(seq_id, method)
            if lags_ms is None or len(lags_ms) == 0:
                continue
            summary.append({
                "seq": seq_id,
                "method": method,
                "n_win": len(lags_ms),
                "lag_mean_ms": float(np.mean(lags_ms)),
                "lag_std_ms": float(np.std(lags_ms)),
                "lag_mean_frames": float(np.mean(lags_fr)),
                "lag_std_frames": float(np.std(lags_fr)),
            })

    if len(summary) == 0:
        print("No valid sequences/windows found. Exiting.")
        return

    # Save CSV
    csv_path = os.path.join(RESULT_DIR, "lag_summary_all.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nSummary saved to {csv_path}")

    # Aggregate for visualization
    pos_vals = [s["lag_mean_ms"] for s in summary if s["method"].upper() == "POS"]
    chrom_vals = [s["lag_mean_ms"] for s in summary if s["method"].upper() == "CHROM"]

    # Only plot methods that were actually evaluated
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
    ax.set_ylabel("Mean Lag (ms)")
    ax.set_title("Sequence-wise Mean Lag Distribution")
    ax.grid(True, axis="y", linestyle=":")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "boxplot_mean_lag_ms.png"), dpi=200)
    plt.show()

    # Violin plots for variability
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.violinplot(data_for_plots, showmeans=True, showmedians=False)
    ax2.set_xticks(np.arange(1, len(methods_present) + 1))
    ax2.set_xticklabels(methods_present)
    ax2.set_ylabel("Mean Lag (ms)")
    ax2.set_title("Distribution of Lags across PURE Dataset")
    ax2.grid(True, axis="y", linestyle=":")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "violin_lag_ms.png"), dpi=200)
    plt.show()

    # Print overall
    print("\n=== Global Results ===")
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
