# Main/Exp_2/Step_1_UBFC_edge_lag_summary.py
import sys
from pathlib import Path
# Add project root (Project_rPPG) to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from pathlib import Path
import numpy as np

from Main.Exp_2.Step_1_UBFC_edge_lag_single_seq import (
    UBFC_ROOT,
    compute_edge_lags_for_sequence,
)


def find_ubfc_sequences(root: Path):
    """
    We find UBFC sequence folders under the root.
    We assume folders like 'vid_1', 'vid_2', ... each contain one .avi and one .txt.
    """
    seq_dirs = []
    for p in root.iterdir():
        if p.is_dir() and p.name.lower().startswith("vid_"):
            seq_dirs.append(p.name)
    return sorted(seq_dirs)


def main():
    print(f"UBFC root: {UBFC_ROOT}")
    seq_ids = find_ubfc_sequences(UBFC_ROOT)
    print(f"Found {len(seq_ids)} sequence folders: {seq_ids}\n")

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

    print("\n================ Edge lag + decision summary (CHROM) ================")
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



'''
| Label          | Meaning                               | Action                              |
| -------------- | ------------------------------------- | ----------------------------------- |
| TRIM           | Lag small → alignment OK              | Just trim remaining frames          |
| FPS_ADJUST     | Lag large AND matches leftover frames | Adjust FPS (drift fix)              |
| UNCERTAIN      | Lag large but leftover frames ≈ 0     | Hard case → needs further study     |
''' 