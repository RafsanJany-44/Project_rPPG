"""
UBFC STEP 2 (modular):
Validate and visualize GT–Video time alignment.

Tasks:
1. Load UBFC GT (PPG + timestamps) and fix non-increasing timestamps.
2. Load video via UBFCFrameSource and compute frame timestamps.
3. Compare durations of GT vs video.
4. Compute overlap range and number of frames inside overlap.
5. Plot intuitive timeline visualization showing overlap.
"""
import sys
from pathlib import Path
# Add project root (Project_rPPG) to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from Main.Data_Read_Engine.ubfc_alignment import load_ubfc_gt_and_fix
from Main.Signal_Processing_Engine.ubfc_dataset import UBFCFrameSource


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SEQ_ID = "vid_1"
UBFC_ROOT = Path(r"D:\Data\UBFC\Dataset_3")
SEQ_DIR = UBFC_ROOT / SEQ_ID
GT_FILE = SEQ_DIR / "ground_truth_1.txt"
VIDEO_FILE = SEQ_DIR / f"{SEQ_ID}.avi"


def main():
    print(f"=== UBFC STEP 2: Align and visualize GT/Video timing for {SEQ_ID} ===")

    # --- Load GT (with timestamp fix) ---
    t_ppg_s, ppg_wave, corrected_idx = load_ubfc_gt_and_fix(GT_FILE)
    print(f"Loaded GT: {len(ppg_wave)} samples")

    if corrected_idx:
        print(f"{len(corrected_idx)} non-increasing timestamps corrected at indices {corrected_idx[:10]}")
        if len(corrected_idx) > 10:
            print("...(further indices omitted)")
    else:
        print("No duplicated/non-increasing timestamps found.")

    # --- Video info via UBFCFrameSource ---
    source = UBFCFrameSource(VIDEO_FILE)
    n_frames = source.frames_bgr.shape[0]
    fps = source.fps
    t_frame_s = source.t_frame_s

    dur_vid = t_frame_s[-1] - t_frame_s[0]
    dur_ppg = t_ppg_s[-1] - t_ppg_s[0]
    delta_t = dur_vid - dur_ppg
    delta_frames = delta_t * fps

    print(f"Video info: {n_frames} frames @ {fps:.3f} fps → {dur_vid:.3f} s")
    print(f"GT PPG duration: {dur_ppg:.3f} s")
    print(f"Duration difference (video − GT): {delta_t:.3f} s "
          f"≈ {delta_frames:.2f} frames ({(delta_t/dur_vid)*100:.2f} % of duration)")

    # --- Build overlap ---
    t_ppg_min, t_ppg_max = t_ppg_s[0], t_ppg_s[-1]
    t_vid_min, t_vid_max = t_frame_s[0], t_frame_s[-1]

    t_min = max(t_ppg_min, t_vid_min)
    t_max = min(t_ppg_max, t_vid_max)
    mask = (t_frame_s >= t_min) & (t_frame_s <= t_max)
    t_frame_valid = t_frame_s[mask]

    print(f"Frames within overlap: {len(t_frame_valid)} "
          f"({n_frames - len(t_frame_valid)} trimmed outside overlap)")

    # --- Plot timeline visualization ---
    fig, ax = plt.subplots(figsize=(10, 2.8))

    ax.hlines(1.0, t_frame_s[0], t_frame_s[-1], color="royalblue", lw=6, label="Video frames")
    ax.hlines(0.5, t_ppg_s[0], t_ppg_s[-1], color="orange", lw=6, label="GT PPG timestamps")
    ax.axvspan(t_min, t_max, color="lightgreen", alpha=0.3, label="Overlap region")

    ax.text(t_frame_s[-1] + 0.1, 1.0, f"{dur_vid:.3f}s", va="center", color="royalblue")
    ax.text(t_ppg_s[-1] + 0.1, 0.5, f"{dur_ppg:.3f}s", va="center", color="orange")

    ax.set_title(f"{SEQ_ID}: GT–Video Duration Alignment")
    ax.set_xlabel("Time (s)")
    ax.set_yticks([0.5, 1.0])
    ax.set_yticklabels(["GT PPG", "Video"])
    ax.set_xlim(0.0, max(dur_vid, dur_ppg) + 0.5)
    ax.legend(loc="upper right")
    ax.grid(True, axis="x", linestyle=":")

    plt.tight_layout()
    plt.show()

    # Optional zoomed tail view
    zoom_start = max(t_ppg_s[-1] - 1.0, 0)
    fig2, ax2 = plt.subplots(figsize=(8, 2.8))
    ax2.plot(t_frame_s, np.ones_like(t_frame_s), "|", color="royalblue", markersize=12, label="Video frames")
    ax2.plot(t_ppg_s, np.zeros_like(t_ppg_s), "|", color="orange", markersize=12, label="PPG timestamps")
    ax2.set_xlim(zoom_start, max(dur_vid, dur_ppg))
    ax2.set_yticks([])
    ax2.set_title(f"Zoomed tail region (last 1 s) — {SEQ_ID}")
    ax2.set_xlabel("Time (s)")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    print("\nUBFC Step 2 completed – timestamps cleaned, durations visualized, overlap confirmed.")


if __name__ == "__main__":
    main()
