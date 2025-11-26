# Main/Experiment_Cell/UBFC_edge_lag_summary.py

from pathlib import Path
import numpy as np

from .Do_Not_Run_UBFC_edge_lag_single_seq import (
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

    print("\n================ Edge lag summary (CHROM) ================")
    print("seq_id    lag_start_fr   lag_end_fr   delta_fr   fps_start   fps_end")
    for r in results:
        print(
            f"{r['seq_id']:8s}"
            f"{r['lag_start_frames']:13.3f}"
            f"{r['lag_end_frames']:12.3f}"
            f"{r['delta_lag_frames']:10.3f}"
            f"{r['fps_start']:11.3f}"
            f"{r['fps_end']:10.3f}"
        )

    deltas = np.array([r["delta_lag_frames"] for r in results], dtype=np.float32)
    print("\nDelta lag (frames) across sequences:")
    print(f"  mean = {float(deltas.mean()):.3f}")
    print(f"  std  = {float(deltas.std()):.3f}")
    print(f"  min  = {float(deltas.min()):.3f}")
    print(f"  max  = {float(deltas.max()):.3f}")

    small_thr = 2.0  # threshold for "small" drift in frames
    n_small = int((deltas <= small_thr).sum())
    n_large = int((deltas > small_thr).sum())

    print(f"\nNumber of sequences with delta_lag_frames ≤ {small_thr:.1f}: {n_small}")
    print(f"Number of sequences with delta_lag_frames  > {small_thr:.1f}: {n_large}")


if __name__ == "__main__":
    main()
