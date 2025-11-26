# step1_check_json_timing.py
"""
STEP 1: Verify JSON structure and timing for one PURE sequence.

Goals:
- Start with an easy sample (for example, a steady sequence such as 01-01).
- Verify low-level execution of data loading and timestamp handling.
- Check sampling rates and basic timing statistics.
"""
import sys
from pathlib import Path
# Add project root (Project_rPPG) to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import numpy as np
from Main.Data_Read_Engine import DEFAULT_SEQ_ID
from Main.Data_Read_Engine import load_pure_json, extract_streams_from_pure_json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Ru with configurable parameters.")
    parser.add_argument(
        "--seq",
        type=str,
        default="01-01",           # ← default if not provided
        help="Sequence ID (01-01). Defaults to 01-01."
    )

    args = parser.parse_args()


    # Override config value in memory
    DEFAULT_SEQ_ID = args.seq

    seq_id = DEFAULT_SEQ_ID
    print(f"=== STEP 1: JSON timing check for sequence {seq_id} ===")

    # 1) Load JSON
    data = load_pure_json(seq_id)
    print("JSON loaded successfully.")

    # 2) Extract streams from JSON
    t_ppg_ns, wave, t_vid_ns, hr_dev = extract_streams_from_pure_json(data)
    print(f"PPG samples:   {len(t_ppg_ns)}")
    print(f"Video frames:  {len(t_vid_ns)}")

    # 3) Convert timestamps to seconds relative to the first video frame
    t0_ns = t_vid_ns[0]

    t_ppg_s = (t_ppg_ns - t0_ns) * 1e-9
    t_vid_s = (t_vid_ns - t0_ns) * 1e-9

    # 4) Compute sampling statistics
    #    PPG sampling interval and corresponding frequency
    dt_ppg = np.diff(t_ppg_s)
    #    Video sampling interval and corresponding frequency
    dt_vid = np.diff(t_vid_s)

    fs_ppg_mean = 1.0 / np.mean(dt_ppg)
    fs_ppg_std = np.std(1.0 / dt_ppg)

    fps_vid_mean = 1.0 / np.mean(dt_vid)
    fps_vid_std = np.std(1.0 / dt_vid)

    print("\n--- Sampling statistics ---")
    print(f"PPG fs (mean):    {fs_ppg_mean:.3f} Hz")
    print(f"PPG fs (std):     {fs_ppg_std:.3f} Hz")
    print(f"Video fps (mean): {fps_vid_mean:.3f} Hz")
    print(f"Video fps (std):  {fps_vid_std:.3f} Hz")
    

    # 5) Show first few entries for manual inspection
    N_SHOW = 20

    print("\n--- First PPG samples (time_s, waveform, HR_dev_BPM) ---")
    for i in range(min(N_SHOW, len(t_ppg_s))):
        print(
            f"PPG[{i:3d}]  t = {t_ppg_s[i]:8.4f} s  "
            f"wave = {wave[i]:4.0f}  HR = {hr_dev[i]:3.0f} BPM"
        )

    print("\n--- First video frames (time_s) ---")
    for i in range(min(N_SHOW, len(t_vid_s))):
        print(f"VID[{i:3d}]  t = {t_vid_s[i]:8.4f} s")

    


if __name__ == "__main__":
    main()
