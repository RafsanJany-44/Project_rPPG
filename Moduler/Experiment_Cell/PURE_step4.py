# step4_ppg_vs_frame_timing_diagnostics.py
"""
STEP 4: Timing diagnostics between PPG timestamps and image (frame) timestamps.

Goals:
- Validate the temporal relationship between PPG samples (/FullPackage)
  and video frames (/Image).
- Summarize durations, offsets, and sampling intervals.
- For each frame timestamp, find the nearest PPG sample and analyze
  the time difference (delta).
- Visualize:
    * PPG waveform with frame times overlaid (zoomed window).
    * Nearest-sample delta as a function of time.
    * Histogram of the nearest-sample delta.
"""

import sys
from pathlib import Path
# Add project root (Project_rPPG) to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



import numpy as np
import matplotlib.pyplot as plt

from Main.Data_Read_Engine import DEFAULT_SEQ_ID
from Main.Data_Read_Engine import load_pure_json, extract_streams_from_pure_json


import numpy as np
import matplotlib.pyplot as plt

from Main.Data_Read_Engine import DEFAULT_SEQ_ID
from Main.Data_Read_Engine import load_pure_json, extract_streams_from_pure_json
from Main.Data_Read_Engine import load_pure_image_files_and_timestamps 


def compute_nearest_ppg_deltas(t_ppg_s: np.ndarray, t_vid_s: np.ndarray) -> np.ndarray:
    """
    For each video frame time, compute the time difference to the nearest
    PPG sample.

    Parameters
    ----------
    t_ppg_s : np.ndarray
        Time stamps of PPG samples in seconds (monotonically increasing).
    t_vid_s : np.ndarray
        Time stamps of video frames in seconds.

    Returns
    -------
    dt_nearest : np.ndarray
        For each frame time t_vid_s[i], the signed time difference:
            t_ppg_nearest - t_vid_s[i]
        where t_ppg_nearest is the closest PPG sample time.
    """
    dt_nearest = np.zeros_like(t_vid_s)

    # We use searchsorted to find insertion positions in the PPG time axis.
    # For each frame time, we compare the PPG sample just before and just after.
    indices = np.searchsorted(t_ppg_s, t_vid_s)

    for i, idx in enumerate(indices):
        # Candidates: idx-1 and idx (if they exist)
        candidates = []
        if idx > 0:
            candidates.append(idx - 1)
        if idx < len(t_ppg_s):
            candidates.append(idx)

        # Select the candidate with minimal absolute time difference
        best_idx = candidates[0]
        best_dt = t_ppg_s[best_idx] - t_vid_s[i]
        for c in candidates[1:]:
            dt_c = t_ppg_s[c] - t_vid_s[i]
            if abs(dt_c) < abs(best_dt):
                best_idx = c
                best_dt = dt_c

        dt_nearest[i] = best_dt

    return dt_nearest


def main():
    '''
    # for timestaps from json file for the image
    seq_id = DEFAULT_SEQ_ID
    print(f"=== STEP 4: PPG vs frame timing diagnostics for sequence {seq_id} ===")

    # 1) Load JSON and extract time streams
    data = load_pure_json(seq_id)
    t_ppg_ns, wave, t_vid_ns, hr_dev = extract_streams_from_json(data)

    # 2) Convert timestamps to seconds, relative to the first video frame
    t0_ns = t_vid_ns[0]
    t_ppg_s = (t_ppg_ns - t0_ns) * 1e-9
    t_vid_s = (t_vid_ns - t0_ns) * 1e-9
    '''

    # for timestaps from the image file name -> both generate the same result its just for checkig
    seq_id = DEFAULT_SEQ_ID
    print(f"=== STEP 4: PPG vs frame timing diagnostics for sequence {seq_id} ===")

    # 1) Load JSON and extract PPG timestamps and values
    data = load_pure_json(seq_id)
    t_ppg_ns, wave, t_vid_ns_json, hr_dev = extract_streams_from_pure_json(data)

    # 2) Load image files and timestamps from filenames
    image_files, t_img_ns = load_pure_image_files_and_timestamps(seq_id)

    print(f"JSON /Image entries:       {len(t_vid_ns_json)}")
    print(f"Image*.png frame files:    {len(image_files)}")

    # 3) Verify that JSON /Image timestamps and filename timestamps match
    same_shape = t_vid_ns_json.shape == t_img_ns.shape
    same_values = np.array_equal(t_vid_ns_json, t_img_ns)

    print("\n--- JSON vs filename timestamp check (inside Step 4) ---")
    print(f"Same shape:  {same_shape}")
    print(f"Same values: {same_values}")

    # 4) Decide which timestamps to use as frame times
    #    Here we explicitly use the timestamps parsed from filenames (t_img_ns)
    #    as the reference frame times.
    t_frame_ns = t_img_ns

    # 5) Convert timestamps to seconds, relative to the first frame
    t0_ns = t_frame_ns[0]
    t_ppg_s = (t_ppg_ns    - t0_ns) * 1e-9
    t_vid_s = (t_frame_ns  - t0_ns) * 1e-9


    # 3) Basic duration and offset diagnostics
    duration_ppg = t_ppg_s[-1] - t_ppg_s[0]
    duration_vid = t_vid_s[-1] - t_vid_s[0]
    start_offset = t_ppg_s[0] - t_vid_s[0]
    end_offset = t_ppg_s[-1] - t_vid_s[-1]

    dt_ppg = np.diff(t_ppg_s)
    dt_vid = np.diff(t_vid_s)

    fs_ppg_mean = 1.0 / np.mean(dt_ppg)
    fs_ppg_std = np.std(1.0 / dt_ppg)
    fps_vid_mean = 1.0 / np.mean(dt_vid)
    fps_vid_std = np.std(1.0 / dt_vid)

    print("\n--- Duration and offset summary ---")
    print(f"PPG duration:   {duration_ppg:.3f} s")
    print(f"Video duration: {duration_vid:.3f} s")
    print(f"Start offset (PPG_start - VID_start): {start_offset*1000:.3f} ms")
    print(f"End offset   (PPG_end   - VID_end):   {end_offset*1000:.3f} ms")

    print("\n--- Sampling statistics (recomputed) ---")
    print(f"PPG fs (mean):    {fs_ppg_mean:.3f} Hz")
    print(f"PPG fs (std):     {fs_ppg_std:.3f} Hz")
    print(f"Video fps (mean): {fps_vid_mean:.3f} Hz")
    print(f"Video fps (std):  {fps_vid_std:.3f} Hz")

    # 4) For each frame, compute the time difference to nearest PPG sample
    dt_nearest = compute_nearest_ppg_deltas(t_ppg_s, t_vid_s)  # seconds
    dt_nearest_ms = dt_nearest * 1000.0

    mean_abs_dt = np.mean(np.abs(dt_nearest_ms))
    max_abs_dt = np.max(np.abs(dt_nearest_ms))
    min_dt = np.min(dt_nearest_ms)
    max_dt = np.max(dt_nearest_ms)

    print("\n--- Nearest-sample delta statistics ---")
    print(f"Mean(|delta|): {mean_abs_dt:.3f} ms")
    print(f"Max(|delta|):  {max_abs_dt:.3f} ms")
    print(f"Min(delta):    {min_dt:.3f} ms")
    print(f"Max(delta):    {max_dt:.3f} ms")

    # 5) Choose a zoom window for visualization
    t_start_win = 5.0
    t_end_win = 11.0

    mask_ppg = (t_ppg_s >= t_start_win) & (t_ppg_s <= t_end_win)
    mask_vid = (t_vid_s >= t_start_win) & (t_vid_s <= t_end_win)

    t_ppg_win = t_ppg_s[mask_ppg]
    wave_win = wave[mask_ppg]
    t_vid_win = t_vid_s[mask_vid]
    dt_nearest_win_ms = dt_nearest_ms[mask_vid]

    print(
        f"\nZoom window: {t_start_win:.1f} s to {t_end_win:.1f} s "
        f"({t_end_win - t_start_win:.1f} s duration)"
    )
    print(f"PPG samples in window:   {len(t_ppg_win)}")
    print(f"Frames in window:        {len(t_vid_win)}")

    # 6) Visualization
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # 6a) PPG waveform with frame times overlaid (zoomed window)
    ax0 = axes[0]
    ax0.plot(t_ppg_win, wave_win, label="PPG waveform (native)", alpha=0.8)
    for tv in t_vid_win:
        ax0.axvline(tv, color="k", linestyle=":", linewidth=0.7, alpha=0.4)
    ax0.set_title(
        f"Sequence {seq_id}: PPG waveform with frame timestamps "
        f"({t_start_win:.1f}-{t_end_win:.1f} s)"
    )
    ax0.set_xlabel("Time (s)")
    ax0.set_ylabel("PPG amplitude (a.u.)")
    ax0.grid(True)
    ax0.legend(loc="best")

    # 6b) Nearest PPG delta vs frame time (full duration)
    ax1 = axes[1]
    ax1.plot(t_vid_s, dt_nearest_ms, ".", markersize=3, alpha=0.7)
    ax1.set_title("Nearest PPG sample time difference vs frame time")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Delta to nearest PPG (ms)")
    ax1.grid(True)

    # 6c) Histogram of nearest-sample deltas
    ax2 = axes[2]
    ax2.hist(dt_nearest_ms, bins=40, edgecolor="black", alpha=0.7)
    ax2.set_title("Histogram of nearest PPG sample time differences")
    ax2.set_xlabel("Delta to nearest PPG (ms)")
    ax2.set_ylabel("Count")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
