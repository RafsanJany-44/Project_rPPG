# step2_align_ppg_to_frames.py
"""
STEP 2: Align PPG to video frame times and visualize.

Goals:
- Use the verified JSON timing to interpolate PPG at video frame times.
- Plot raw PPG (native sampling) and resampled PPG at frame times.
- Inspect whether a visually consistent phase relation appears.
"""

import sys
from pathlib import Path
# Add project root (Project_rPPG) to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Main.Data_Read_Engine  import DEFAULT_SEQ_ID
from Main.Data_Read_Engine import load_pure_json, extract_streams_from_pure_json
from Main.Data_Read_Engine import pure_align_ppg_to_frame_times, pure_select_time_window


def main():
    seq_id = DEFAULT_SEQ_ID
    print(f"=== STEP 2: Align PPG to frame times for sequence {seq_id} ===")

    # 1) Load JSON and extract streams (already validated in step 1)
    data = load_pure_json(seq_id)
    t_ppg_ns, wave, t_vid_ns, hr_dev = extract_streams_from_pure_json(data)

    # 2) Convert timestamps to seconds relative to the first video frame
    t0_ns = t_vid_ns[0]
    t_ppg_s = (t_ppg_ns - t0_ns) * 1e-9
    t_vid_s = (t_vid_ns - t0_ns) * 1e-9

    # 3) Interpolate PPG waveform at video frame times
    wave_interp = pure_align_ppg_to_frame_times(t_ppg_s, wave, t_vid_s)

    print("PPG aligned to frame times using linear interpolation.")
    print(f"Total PPG samples (raw): {len(t_ppg_s)}")
    print(f"Total frames (aligned points): {len(t_vid_s)}")

    # 4) Define a short time window for detailed inspection (first 6 seconds)
    t_start_win = 5.0   # start at 5 s to avoid any initial transient
    t_end_win = 11.0    # 6-second window

    (
        t_ppg_win,
        wave_win,
    ) = pure_select_time_window(t_ppg_s, wave, t_start=t_start_win, t_end=t_end_win)

    (
        t_vid_win,
        wave_interp_win,
    ) = pure_select_time_window(t_vid_s, wave_interp, t_start=t_start_win, t_end=t_end_win)

    print(
        f"Selected window: {t_start_win:.1f} s to {t_end_win:.1f} s "
        f"({t_end_win - t_start_win:.1f} s duration)"
    )
    print(f"PPG samples in window: {len(t_ppg_win)}")
    print(f"Frames in window:      {len(t_vid_win)}")

    # 5) Prepare a small DataFrame for possible export / inspection
    df_window = pd.DataFrame({
        "t_frame_s": t_vid_win,
        "PPG_interp_at_frame": wave_interp_win,
    })

    print("\nFirst few aligned frame samples in the selected window:")
    print(df_window.head(10))

    # 6) Visualization
    #    Figure layout:
    #      - Top subplot: full-length signals
    #      - Bottom subplot: zoomed 6-second window

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    # 6a) Full-length view
    ax_full = axes[0]
    ax_full.plot(t_ppg_s, wave, label="Raw PPG (native sampling)", alpha=0.7)
    ax_full.plot(
        t_vid_s,
        wave_interp,
        "o",
        markersize=2,
        label="PPG resampled at frame times",
        alpha=0.6,
    )
    ax_full.set_title(f"Sequence {seq_id}: PPG and PPG@frames (full duration)")
    ax_full.set_xlabel("Time (s)")
    ax_full.set_ylabel("Amplitude (a.u.)")
    ax_full.legend(loc="best")
    ax_full.grid(True)

    # 6b) Zoomed-in window
    ax_zoom = axes[1]
    ax_zoom.plot(
        t_ppg_win,
        wave_win,
        label="Raw PPG (native sampling)",
        alpha=0.7,
    )
    ax_zoom.plot(
        t_vid_win,
        wave_interp_win,
        "o-",
        markersize=4,
        label="PPG@frame times (interpolated)",
        alpha=0.8,
    )
    ax_zoom.set_title(
        f"Sequence {seq_id}: Zoomed window [{t_start_win:.1f}, {t_end_win:.1f}] s"
    )
    ax_zoom.set_xlabel("Time (s)")
    ax_zoom.set_ylabel("Amplitude (a.u.)")
    ax_zoom.legend(loc="best")
    ax_zoom.grid(True)

    plt.tight_layout()
    plt.show()

    # 7) Optionally save the windowed alignment to CSV for offline inspection
    out_csv_name = f"{seq_id}_aligned_window_{int(t_start_win)}_{int(t_end_win)}s.csv"
    df_window.to_csv(out_csv_name, index=False)
    print(f"\nWindowed alignment data saved to: {out_csv_name}")


if __name__ == "__main__":
    main()
