"""
UBFC STEP 1 (modular):
Check GT and video timing for a single UBFC recording.

We:
- load UBFC ground_truth_*.txt,
- parse PPG + timestamps and fix any non-increasing timestamps,
- verify 1:1 alignment,
- load the corresponding .avi video via UBFCFrameSource,
- compare GT duration vs video duration,
- compute dt statistics and derive average sampling rate.
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

from Main.Data_Read_Engine import load_ubfc_ground_truth  
from Main.Data_Read_Engine.ubfc_alignment import fix_timestamps_inplace
from Main.Signal_Processing_Engine.ubfc_dataset import UBFCFrameSource


# ----------------------------------------------------------------------
# User configuration
# ----------------------------------------------------------------------
SEQ_ID = "vid_1"
UBFC_ROOT = Path(r"D:\Data\UBFC\Dataset_3")
SEQ_DIR = UBFC_ROOT / SEQ_ID
GT_FILE = SEQ_DIR / "ground_truth_1.txt"
VIDEO_FILE = SEQ_DIR / f"{SEQ_ID}.avi"


def main():
    print(f"=== UBFC STEP 1: Check GT and Video timing for {SEQ_ID} ===")

    # --- Load GT ---
    t_ppg_s_raw, ppg_wave = load_ubfc_ground_truth(str(GT_FILE))
    n_ppg = len(ppg_wave)
    n_t = len(t_ppg_s_raw)
    print(f"GT loaded successfully: {n_ppg} PPG samples, {n_t} timestamps")

    if n_ppg != n_t:
        raise RuntimeError(
            f"PPG and timestamp lengths differ: {n_ppg} vs {n_t}. "
            f"GT file may be malformed."
        )

    # Fix non-increasing timestamps
    t_ppg_s, corrected_idx = fix_timestamps_inplace(t_ppg_s_raw, eps=1e-6)
    if corrected_idx:
        print(f"Corrected {len(corrected_idx)} non-increasing timestamps at indices: "
              f"{corrected_idx[:10]}")
        if len(corrected_idx) > 10:
            print("...(further indices omitted)")
    else:
        print("GT timestamps already strictly increasing.")

    # --- Check timestamp spacing ---
    dt_ppg = np.diff(t_ppg_s)
    if np.any(dt_ppg <= 0):
        print("\nWARNING: Non-positive time differences remain in GT timestamps.")
        print(f"dt_min = {dt_ppg.min():.6f} s, dt_max = {dt_ppg.max():.6f} s")
    else:
        dt_mean = float(np.mean(dt_ppg))
        dt_std = float(np.std(dt_ppg))
        dt_min = float(np.min(dt_ppg))
        dt_max = float(np.max(dt_ppg))

        fs_mean = 1.0 / dt_mean

        print("\nGT timestamp Δt statistics:")
        print(f"  dt_mean = {dt_mean*1000:.3f} ms")
        print(f"  dt_std  = {dt_std*1000:.6f} ms")
        print(f"  dt_min  = {dt_min*1000:.3f} ms")
        print(f"  dt_max  = {dt_max*1000:.3f} ms")
        print(f"  → Approx PPG sampling rate ≈ {fs_mean:.3f} Hz")

        if dt_std < 1e-6:
            print("  All timestamps are essentially evenly spaced (dt_std ≈ 0).")

    # --- Video info via UBFCFrameSource ---
    source = UBFCFrameSource(VIDEO_FILE)
    n_frames = source.frames_bgr.shape[0]
    fps = source.fps
    t_frame_s = source.t_frame_s

    vid_dur = t_frame_s[-1] - t_frame_s[0]
    gt_dur = t_ppg_s[-1] - t_ppg_s[0]

    print(f"\nVideo info: {n_frames} frames @ {fps:.3f} fps → {vid_dur:.3f} s")
    print(f"GT PPG duration: {gt_dur:.3f} s")

    dur_diff = vid_dur - gt_dur
    print(f"Duration difference (video − GT): {dur_diff:.3f} s")

    print("\nUBFC Step 1 completed – GT and video timing validated.")


if __name__ == "__main__":
    main()
