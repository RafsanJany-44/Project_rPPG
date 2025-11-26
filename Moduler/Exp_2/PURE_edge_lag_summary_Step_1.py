# Main/Exp_2/PURE_edge_lag_summary_Step_1.py


import sys
from pathlib import Path
# Add project root (Project_rPPG) to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



import os
import glob
import numpy as np

from Main.Exp_2.PURE_edge_lag_single_seq_Step_1 import compute_edge_lags_for_sequence


# We point to the folder that contains all PURE JSON files, e.g. "01-01.json"
PURE_JSON_ROOT = r"D:\Data\MAIN_PURE\PURE\ALL\ALL"


def find_pure_sequences(json_root: str):
    """
    We find PURE sequence IDs from the JSON files in json_root.

    We assume files like:
        01-01.json, 01-02.json, ...
    and we return the list:
        ["01-01", "01-02", ...]
    """
    pattern = os.path.join(json_root, "*.json")
    json_files = glob.glob(pattern)

    seq_ids = []
    for p in json_files:
        base = os.path.basename(p)
        seq_id, ext = os.path.splitext(base)
        if seq_id:  # simple sanity check
            seq_ids.append(seq_id)

    return sorted(seq_ids)


def main():
    print(f"PURE JSON root: {PURE_JSON_ROOT}")
    seq_ids = find_pure_sequences(PURE_JSON_ROOT)
    print(f"Found {len(seq_ids)} sequences: {seq_ids}\n")

    results = []

    for seq_id in seq_ids:
        print(f"\n----- Processing {seq_id} -----")
        try:
            info = compute_edge_lags_for_sequence(seq_id)
            results.append(info)
        except Exception as e:
            print(f"[{seq_id}] Error: {e}")

    if not results:
        print("\nNo successful sequences. We need to check paths or data.")
        return

    print("\n================ Edge lag + decision summary (PURE, CHROM) ================")
    print("seq_id    delta_fr   remain_fr   decision")
    for r in results:
        print(
            f"{r['seq_id']:8s}"
            f"{r['delta_lag_frames']:10.3f}"
            f"{r['remaining_frames']:12.3f}"
            f"{r['decision']:>12s}"
        )

    deltas = np.array([r["delta_lag_frames"] for r in results], dtype=np.float32)
    remains = np.array([r["remaining_frames"] for r in results], dtype=np.float32)

    print("\nDelta lag (frames) across sequences:")
    print(f"  mean = {float(deltas.mean()):.3f}")
    print(f"  std  = {float(deltas.std()):.3f}")
    print(f"  min  = {float(deltas.min()):.3f}")
    print(f"  max  = {float(deltas.max()):.3f}")

    print("\nRemaining frames (video tail) across sequences:")
    print(f"  mean = {float(remains.mean()):.3f}")
    print(f"  std  = {float(remains.std()):.3f}")
    print(f"  min  = {float(remains.min()):.3f}")
    print(f"  max  = {float(remains.max()):.3f}")

    n_trim = sum(r["decision"] == "TRIM" for r in results)
    n_fps = sum(r["decision"] == "FPS_ADJUST" for r in results)
    n_unc = sum(r["decision"] == "UNCERTAIN" for r in results)

    print("\nDecision counts:")
    print(f"  TRIM       : {n_trim}")
    print(f"  FPS_ADJUST : {n_fps}")
    print(f"  UNCERTAIN  : {n_unc}")


if __name__ == "__main__":
    main()


"""
Label meanings for the decisions:

TRIM:
    Lag difference between start and end windows is small.
    Alignment is essentially OK. We can safely trim remaining frames at the tail.

FPS_ADJUST:
    Lag difference is large and matches the number of leftover frames at the tail.
    This suggests a small global FPS drift. We can correct FPS so that
    video and PPG line up from start to end without trimming frames.

UNCERTAIN:
    Lag is large but cannot be explained by a simple FPS drift or small tail.
    These sequences need further inspection.
"""
