# Main/VIZ/UBFC_sequence_viz_plotly_window.py

from pathlib import Path
import argparse

import sys
from pathlib import Path
# Add project root (Project_rPPG) to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



from Main.VIZ.UBFC_seq_viz_1 import (
    prepare_full_signals,
    UBFC_ROOT,
    WIN_LEN,
    PADDING,
)


from Main.VIZ.Plot_Me import make_window_plot_1, make_window_plot_2


def main():
    parser = argparse.ArgumentParser(
        description=(
            "UBFC visualization for a user-defined window INDEX.\n"
            "Window timing is computed as:\n"
            "  t_start = overlap_start + padding + win_idx * WIN_LEN\n"
            "  t_end   = t_start + WIN_LEN"
        )
    )
    parser.add_argument(
        "--seq",
        type=str,
        required=False,
        default="vid_15",
        help="Sequence ID, for example vid_1, vid_15, vid_20.",
    )
    parser.add_argument(
        "--win_idx",
        type=int,
        required=False,
        default= 5,
        help=(
            "Window index (0-based). "
            "win_idx=0 → first window; "
            "win_idx=1 → next window; etc."
        ),
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(UBFC_ROOT),
        help="UBFC root folder.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="Figures/UBFC_VIZ_plotly_Window",
        help="Output directory for HTML figures.",
    )
    parser.add_argument(
        "--local_win",
        type=float,
        default=4.0,
        help="Local window length (seconds) for per-frame lag estimation.",
    )
    # Optional overrides, but by default we use global WIN_LEN and PADDING
    parser.add_argument(
        "--win_len",
        type=float,
        default=WIN_LEN,
        help="Window length in seconds (default = global WIN_LEN).",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=PADDING,
        help="Padding from overlap start in seconds (default = global PADDING).",
    )

    args = parser.parse_args()

    seq_id = args.seq
    win_idx = int(args.win_idx)
    root = Path(args.root)
    out_dir = Path(args.out) / seq_id

    win_len = float(args.win_len)
    padding = float(args.padding)

    print(f"\n=== UBFC Plotly window visualization (by index) for {seq_id} ===")
    print(f"UBFC root: {root}")
    print(f"WIN_LEN = {win_len:.3f} s, PADDING = {padding:.3f} s")
    print(f"Requested window index: {win_idx}")

    # 1) Prepare full overlap signals (GT + CHROM)
    info = prepare_full_signals(seq_id, root)

    overlap_start = info["overlap_start"]
    overlap_end = info["overlap_end"]

    # 2) Compute time range for this window index (Option C)
    base_start = overlap_start + padding
    t_start = base_start + win_idx * win_len
    t_end = t_start + win_len

    print(
        f"Global overlap: [{overlap_start:.3f}, {overlap_end:.3f}] s\n"
        f"Computed window (index {win_idx}): [{t_start:.3f}, {t_end:.3f}] s"
    )

    # 3) Check that the window is inside the overlap
    if t_start < overlap_start or t_end > overlap_end:
        raise RuntimeError(
            "Requested window index goes outside GT–video overlap.\n"
            f"Overlap is [{overlap_start:.3f}, {overlap_end:.3f}] s.\n"
            f"Computed window is [{t_start:.3f}, {t_end:.3f}] s."
        )

    # 4) Build detailed visualizations for this window
    label = f"win{win_idx}"

    make_window_plot_1(
        info=info,
        t_start=t_start,
        t_end=t_end,
        label=label,
        out_dir=out_dir,
        local_win_seconds=float(args.local_win),
    )

    make_window_plot_2(
        info=info,
        t_start=t_start,
        t_end=t_end,
        label=label,
        out_dir=out_dir,
        local_win_seconds=float(args.local_win),
    )

    print("\nInteractive window-index visualization complete.")


if __name__ == "__main__":
    main()
