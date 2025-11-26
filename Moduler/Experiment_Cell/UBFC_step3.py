"""
UBFC STEP 3 (modular):
Align GT PPG to video frame times for one recording.

We:
- load GT PPG + timestamps and fix anomalies,
- load video via UBFCFrameSource and derive frame times,
- compute overlap between GT and video,
- interpolate GT PPG onto overlapping frame times,
- report trimming at start/end,
- visualize a local window with raw GT vs frame-aligned GT.
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


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
UBFC_ROOT = Path(r"D:\Data\UBFC\Dataset_3")
SEQ_ID = "vid_1"

SEQ_DIR = UBFC_ROOT / SEQ_ID
GT_FILE = SEQ_DIR / "ground_truth_1.txt"
VID_FILE = SEQ_DIR / f"{SEQ_ID}.avi"


def main():
    print(f"=== UBFC STEP 3: Align GT PPG to video frame times for {SEQ_ID} ===")

    # 1) Load GT PPG + timestamps and clean anomalies
    t_ppg_s, ppg_raw, corrected_idx = load_ubfc_gt_and_fix(GT_FILE)
    print(f"Loaded GT: {len(ppg_raw)} PPG samples")

    print("\n--- GT timestamp anomaly report ---")
    if len(corrected_idx) == 0:
        print("No non-increasing timestamps detected – GT is strictly increasing.")
    else:
        print(
            f"{len(corrected_idx)} non-increasing timestamps corrected at indices: "
            f"{corrected_idx[:10]}"
        )
        if len(corrected_idx) > 10:
            print("...(further indices omitted)")

    dt_ppg = np.diff(t_ppg_s)
    print(
        f"GT PPG duration: {t_ppg_s[0]:.3f} s → {t_ppg_s[-1]:.3f} s "
        f"(Δ={t_ppg_s[-1] - t_ppg_s[0]:.3f} s)"
    )
    print(
        f"GT dt stats: min={dt_ppg.min():.6f} s, "
        f"max={dt_ppg.max():.6f} s, mean={dt_ppg.mean():.6f} s"
    )

    # 2) Load video via UBFCFrameSource and derive frame times
    source = UBFCFrameSource(VID_FILE)
    frame_count = source.frames_bgr.shape[0]
    fps = source.fps
    t_frame_s = source.t_frame_s
    duration_video = t_frame_s[-1] - t_frame_s[0]

    print(
        f"\nVideo info: {frame_count} frames @ {fps:.3f} fps → "
        f"{duration_video:.3f} s (0.000 s → {t_frame_s[-1]:.3f} s)"
    )

    # 3) Compute overlap and trimming
    t_ppg_start = t_ppg_s[0]
    t_ppg_end = t_ppg_s[-1]

    mask_overlap = (t_frame_s >= t_ppg_start) & (t_frame_s <= t_ppg_end)
    t_frame_overlap = t_frame_s[mask_overlap]

    n_frames_overlap = mask_overlap.sum()
    n_trim_start = int(np.sum(t_frame_s < t_ppg_start))
    n_trim_end = int(np.sum(t_frame_s > t_ppg_end))

    print("\n--- Overlap between GT and video ---")
    print(
        f"GT range:    [{t_ppg_start:.3f}, {t_ppg_end:.3f}] s "
        f"(duration {t_ppg_end - t_ppg_start:.3f} s)"
    )
    print(
        f"Video range: [0.000, {t_frame_s[-1]:.3f}] s "
        f"(duration {duration_video:.3f} s)"
    )
    print(
        f"Overlap frames: {n_frames_overlap} / {frame_count} "
        f"(trim start={n_trim_start}, trim end={n_trim_end})"
    )

    if n_frames_overlap == 0:
        raise RuntimeError("No temporal overlap between GT and video.")

    # 4) Interpolate GT PPG onto overlapping frame times
    ppg_on_frames = ubfc_align_ppg_to_frame_times(
        t_ppg_s=t_ppg_s,
        ppg_wave=ppg_raw,
        t_frame_s=t_frame_overlap,
    )

    print(
        f"\nInterpolated GT PPG onto {len(ppg_on_frames)} frame timestamps "
        f"within overlap."
    )

    # 5) Small table for the first few overlapping frames
    print("\nFirst 10 overlapping frames (frame_idx, t_frame, nearest t_ppg, Δt):")
    overlap_indices = np.where(mask_overlap)[0]
    for k in range(min(10, len(overlap_indices))):
        fi = overlap_indices[k]
        tf = t_frame_s[fi]
        j = int(np.argmin(np.abs(t_ppg_s - tf)))
        tg = t_ppg_s[j]
        dt = tf - tg
        print(
            f"  frame {fi:4d}: t_frame={tf:8.4f} s, "
            f"nearest t_ppg={tg:8.4f} s, Δt={dt*1000:8.3f} ms"
        )

    # 6) Visualization: local window where raw GT vs frame-aligned GT can be seen
    t_mid = 0.5 * (t_frame_overlap[0] + t_frame_overlap[-1])
    win_len = 6.0
    t_win_start = max(t_frame_overlap[0], t_mid - win_len / 2.0)
    t_win_end = min(t_frame_overlap[-1], t_mid + win_len / 2.0)

    mask_frame_win = (t_frame_overlap >= t_win_start) & (t_frame_overlap <= t_win_end)
    t_frame_win = t_frame_overlap[mask_frame_win]
    ppg_frame_win = ppg_on_frames[mask_frame_win]

    mask_ppg_win = (t_ppg_s >= t_win_start) & (t_ppg_s <= t_win_end)
    t_ppg_win = t_ppg_s[mask_ppg_win]
    ppg_win = ppg_raw[mask_ppg_win]

    print(
        f"\nVisualization window: [{t_win_start:.3f}, {t_win_end:.3f}] s "
        f"(≈{t_win_end - t_win_start:.3f} s)"
    )
    print(
        f"  GT samples in window: {len(t_ppg_win)}, "
        f"frame-aligned samples in window: {len(t_frame_win)}"
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        t_frame_win,
        ppg_frame_win,
        label="GT PPG interpolated at frame times",
        linewidth=1.5,
    )
    ax.scatter(
        t_ppg_win,
        ppg_win,
        s=12,
        marker="o",
        alpha=0.7,
        label="Original GT PPG samples",
    )

    ax.set_title(
        f"UBFC {SEQ_ID} – GT PPG vs frame-aligned GT PPG\n"
        "Local window in the overlap region"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("PPG amplitude (a.u.)")
    ax.grid(True)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.show()

    print("\nUBFC Step 3 completed – GT PPG aligned to frame times and visualized.")


if __name__ == "__main__":
    main()
