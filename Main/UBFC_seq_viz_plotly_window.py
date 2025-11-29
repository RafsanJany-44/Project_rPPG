# Main/VIZ/UBFC_sequence_viz_plotly_window.py

from pathlib import Path
import argparse

from UBFC_Start_Seq import (
    prepare_full_signals,
    UBFC_ROOT,
    WIN_LEN,
    PADDING,      
)

from plot_me import make_window_plot_with_pairs


def main():
    parser = argparse.ArgumentParser(
        description=(
            "UBFC visualization for a user-defined window INDEX.\n"
            "Window timing is computed as (same as Step-1 analysis):\n"
            "  base_start = overlap_start + padding\n"
            "  t_start(k) = base_start + k * stride\n"
            "  t_end(k)   = t_start(k) + win_len"
        )
    )
    parser.add_argument(
        "--seq",
        type=str,
        required=True,
        help="Sequence ID, for example vid_1, vid_14, vid_25.",
    )
    parser.add_argument(
        "--win_idx",
        type=int,
        required=True,
        help="Window index (0-based) from the per-window lag summary.",
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
    parser.add_argument(
        "--win_len",
        type=float,
        default=WIN_LEN,
        help="Window length in seconds (default = global WIN_LEN = 8.0).",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=PADDING,
        help="Padding from overlap start in seconds (default = global PADDING = 1.0).",
    )
    parser.add_argument(
        "--stride",
        type=float,
        default=1,
        help="Stride between windows in seconds (default = global WIN_STRIDE = 1.0).",
    )

    args = parser.parse_args()

    seq_id = args.seq
    win_idx = int(args.win_idx)
    root = Path(args.root)
    out_dir = Path(args.out) / seq_id

    win_len = float(args.win_len)
    padding = float(args.padding)
    stride = float(args.stride)

    print(f"\n=== UBFC Plotly window visualization (by index) for {seq_id} ===")
    print(f"UBFC root : {root}")
    print(f"WIN_LEN   = {win_len:.3f} s")
    print(f"PADDING   = {padding:.3f} s")
    print(f"STRIDE    = {stride:.3f} s")
    print(f"Requested window index: {win_idx}")

    # 1) Prepare full overlap signals (GT + CHROM)
    info = prepare_full_signals(seq_id, root)

    overlap_start = info["overlap_start"]
    overlap_end = info["overlap_end"]

    # 2) Compute time range for this window index (using same policy as analysis)
    base_start = overlap_start + padding
    t_start = base_start + win_idx * stride
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

    # 4) Build detailed visualization (GT vs CHROM + pair CSV + local lag curve)
    label = f"win{win_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)

    make_window_plot_with_pairs(
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
