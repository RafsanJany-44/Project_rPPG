# Main/VIZ/UBFC_sequence_viz_plotly_full.py

from pathlib import Path
import argparse

from UBFC_Start_End_Seq import (
    prepare_full_signals,
    UBFC_ROOT,
)
from plot_me import make_window_plot_with_pairs


def main():
    parser = argparse.ArgumentParser(
        description="UBFC full-length visualization (GT vs CHROM + local lag) over the entire overlap."
    )
    parser.add_argument(
        "--seq",
        type=str,
        required=True,
        help="Sequence ID, for example vid_1, vid_15, vid_20.",
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
        default="Figures/UBFC_VIZ_plotly_Full",
        help="Output directory for HTML figures.",
    )
    parser.add_argument(
        "--local_win",
        type=float,
        default=4.0,
        help="Local window length (seconds) for per-frame lag estimation.",
    )

    args = parser.parse_args()

    seq_id = args.seq
    root = Path(args.root)
    out_dir = Path(args.out) / seq_id

    print(f"\n=== UBFC full-length Plotly visualization for {seq_id} ===")
    print(f"UBFC root: {root}")

    # 1) Load full overlap signals (GT + CHROM) using your existing function
    info = prepare_full_signals(seq_id, root)

    overlap_start = info["overlap_start"]
    overlap_end = info["overlap_end"]

    print(
        f"Full GT–video overlap: [{overlap_start:.3f}, {overlap_end:.3f}] s "
        f"({overlap_end - overlap_start:.3f} s total)"
    )

    t_start = overlap_start
    t_end = overlap_end

    # 2) Make "full-length" plots using the same mechanisms as window plots
    #    They will just use the entire overlap instead of an 8 s slice.


    make_window_plot_with_pairs(
        info=info,
        t_start=t_start,
        t_end=t_end,
        label="full",
        out_dir=out_dir,
        local_win_seconds=float(args.local_win),
    )


    print("\nFull-length interactive visualization complete.")


if __name__ == "__main__":
    main()
