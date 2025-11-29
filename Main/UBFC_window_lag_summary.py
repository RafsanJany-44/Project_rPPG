# Main/VIZ/UBFC_window_lag_summary.py

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from UBFC_Start_Seq import (
    prepare_full_signals,
    UBFC_ROOT,
)

from plot_me import estimate_global_lag


def compute_window_lags(
    info: dict,
    win_len: float = 8.0,
    stride: float = 1.0,
    pad_start: float = 1.0,
    pad_end: float = 1.0,
    max_lag_seconds: float = 2.0,
    min_frames: int = 20,
):
    """
    Slide an 8 s window with 1 s stride across the GT–video overlap region,
    and compute one global lag for each window.

    Returns a list of dictionaries, one per window, with:
      - win_idx
      - t_start, t_end, t_center
      - n_frames
      - lag_sec, lag_frames
    """
    t_full = info["t_frame_full"]
    ppg_full = info["ppg_full"]
    rppg_full = info["rppg_full"]
    overlap_start = info["overlap_start"]
    overlap_end = info["overlap_end"]

    if len(t_full) < min_frames:
        raise RuntimeError("Too few frames in full overlap region.")

    # Base timing
    base_start = overlap_start + pad_start
    last_end_allowed = overlap_end - pad_end

    win_stats = []
    win_idx = 0
    t_start = base_start

    while t_start + win_len <= last_end_allowed + 1e-9:
        t_end = t_start + win_len

        # Select frames in this window
        mask = (t_full >= t_start) & (t_full <= t_end)
        t_win = t_full[mask]
        n_win = len(t_win)

        if n_win >= min_frames:
            ppg_win = ppg_full[mask]
            rppg_win = rppg_full[mask]

            dt_win = float(np.mean(np.diff(t_win)))
            lag_sec, lag_frames, _, _ = estimate_global_lag(
                s_rppg=rppg_win,
                s_ppg=ppg_win,
                dt=dt_win,
                max_lag_seconds=max_lag_seconds,
            )
        else:
            # Not enough frames → mark lag as NaN
            lag_sec = np.nan
            lag_frames = np.nan

        win_stats.append(
            {
                "win_idx": win_idx,
                "t_start": t_start,
                "t_end": t_end,
                "t_center": 0.5 * (t_start + t_end),
                "n_frames": n_win,
                "lag_sec": float(lag_sec),
                "lag_frames": float(lag_frames),
            }
        )

        win_idx += 1
        t_start += stride

    return win_stats


def build_window_lag_figure(df: pd.DataFrame, seq_id: str):
    """
    Build a Plotly figure with:
      - Row 1: bar plot of lag_frames vs window index
      - Row 2: line plot of lag_frames vs window center time
    """
    # We will ignore windows with NaN lag in plots
    df_plot = df.copy()
    valid_mask = ~np.isnan(df_plot["lag_frames"])
    df_valid = df_plot[valid_mask]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.1,
        subplot_titles=(
            f"{seq_id} – per-window lag (frames) vs window index",
            f"{seq_id} – per-window lag (frames) vs time",
        ),
        row_heights=[0.5, 0.5],
    )

    # Row 1: bar plot vs window index
    fig.add_trace(
        go.Bar(
            x=df_valid["win_idx"],
            y=df_valid["lag_frames"],
            name="Lag (frames)",
        ),
        row=1,
        col=1,
    )

    # Row 1: zero reference line
    if len(df_valid) > 0:
        fig.add_trace(
            go.Scatter(
                x=[df_valid["win_idx"].min(), df_valid["win_idx"].max()],
                y=[0.0, 0.0],
                mode="lines",
                name="Zero lag",
                line=dict(dash="dash", width=1),
            ),
            row=1,
            col=1,
        )

    # Row 2: line plot vs time (center of window)
    fig.add_trace(
        go.Scatter(
            x=df_valid["t_center"],
            y=df_valid["lag_frames"],
            mode="lines+markers",
            name="Lag (frames)",
        ),
        row=2,
        col=1,
    )

    # Row 2: zero reference line
    if len(df_valid) > 0:
        fig.add_trace(
            go.Scatter(
                x=[df_valid["t_center"].min(), df_valid["t_center"].max()],
                y=[0.0, 0.0],
                mode="lines",
                name="Zero lag (time)",
                line=dict(dash="dash", width=1),
            ),
            row=2,
            col=1,
        )

    fig.update_xaxes(title_text="Window index", row=1, col=1)
    fig.update_yaxes(title_text="Lag (frames)", row=1, col=1)

    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Lag (frames)", row=2, col=1)

    fig.update_layout(
        title=f"{seq_id} – per-window global lag summary",
        height=800,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )

    return fig


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Per-window phase lag summary for a UBFC sequence.\n"
            "Uses 8 s windows with 1 s stride (by default) inside the GT–video overlap."
        )
    )
    parser.add_argument(
        "--seq",
        type=str,
        required=True,
        help="Sequence ID, for example vid_1, vid_15, vid_25.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(UBFC_ROOT),
        help="UBFC root folder (default: UBFC_ROOT from UBFC_sequence_viz_plotly).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="Figures/UBFC_Window_Lag_Summary",
        help="Output directory for CSV and HTML.",
    )
    parser.add_argument(
        "--win_len",
        type=float,
        default=8.0,
        help="Window length in seconds (default 8.0).",
    )
    parser.add_argument(
        "--stride",
        type=float,
        default=1.0,
        help="Stride in seconds between window starts (default 1.0).",
    )
    parser.add_argument(
        "--pad_start",
        type=float,
        default=1.0,
        help="Padding from overlap start in seconds (default 1.0).",
    )
    parser.add_argument(
        "--pad_end",
        type=float,
        default=1.0,
        help="Padding from overlap end in seconds (default 1.0).",
    )
    parser.add_argument(
        "--max_lag",
        type=float,
        default=2.0,
        help="Maximum lag search range in seconds (default 2.0).",
    )

    args = parser.parse_args()

    seq_id = args.seq
    root = Path(args.root)
    out_dir = Path(args.out) / seq_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== UBFC per-window lag summary for {seq_id} ===")
    print(f"UBFC root: {root}")

    # 1) Load full overlap signals (GT + CHROM)
    info = prepare_full_signals(seq_id, root)

    overlap_start = info["overlap_start"]
    overlap_end = info["overlap_end"]
    print(
        f"GT–video overlap: [{overlap_start:.3f}, {overlap_end:.3f}] s "
        f"({overlap_end - overlap_start:.3f} s total)"
    )

    # 2) Compute per-window lags
    win_stats = compute_window_lags(
        info=info,
        win_len=float(args.win_len),
        stride=float(args.stride),
        pad_start=float(args.pad_start),
        pad_end=float(args.pad_end),
        max_lag_seconds=float(args.max_lag),
    )

    if not win_stats:
        print("No valid windows were found.")
        return

    # 3) Save table as CSV
    df = pd.DataFrame(win_stats)
    csv_path = out_dir / f"{seq_id}_window_lag_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved window lag summary table to: {csv_path}")

    # Print first few rows for quick inspection
    print("\nFirst 10 windows:")
    print(df.head(10).to_string(index=False))

    # 4) Build and save Plotly figure
    fig = build_window_lag_figure(df, seq_id=seq_id)
    html_path = out_dir / f"{seq_id}_window_lag_summary.html"
    fig.write_html(html_path)
    print(f"Saved interactive window lag summary figure to: {html_path}")

    try:
        fig.show()
    except Exception:
        pass

    # 5) Simple textual summary
    valid_mask = ~np.isnan(df["lag_frames"])
    if valid_mask.any():
        mean_lag = float(df.loc[valid_mask, "lag_frames"].mean())
        std_lag = float(df.loc[valid_mask, "lag_frames"].std())
        print(
            f"\nLag statistics over valid windows:\n"
            f"  mean lag_frames = {mean_lag:.3f}\n"
            f"  std  lag_frames = {std_lag:.3f}\n"
            f"  windows used    = {valid_mask.sum()} of {len(df)}"
        )


if __name__ == "__main__":
    main()