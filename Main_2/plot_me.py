from pathlib import Path
import argparse

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lag_calculation import _normalize, estimate_global_lag, estimate_local_lag_curve




def make_window_plot_1(
    info,
    t_start: float,
    t_end: float,
    label: str,
    out_dir: Path,
    local_win_seconds: float = 4.0,
):
    """
    We build an interactive Plotly figure for one window:
    - Top: GT vs CHROM with:
        * CHROM points colored by local lag (frames),
        * explicit connection lines between each CHROM point and the
          GT point it is best aligned with (based on local lag).
    - Bottom: local lag (frames) vs time.
    """
    seq_id = info["seq_id"]
    t_full = info["t_frame_full"]
    ppg_full = info["ppg_full"]
    rppg_full = info["rppg_full"]

    # We select the window
    mask = (t_full >= t_start) & (t_full <= t_end)
    t_win = t_full[mask]
    ppg_win = ppg_full[mask]
    rppg_win = rppg_full[mask]

    if len(t_win) < 20:
        print(f"[{seq_id}] {label} window has too few frames ({len(t_win)}).")
        return

    dt = float(np.mean(np.diff(t_win)))

    # 1) Local lag curve for this window (in samples & frames)
    lag_samples_curve, lag_frames_curve = estimate_local_lag_curve(
        s_rppg=rppg_win,
        s_ppg=ppg_win,
        dt=dt,
        local_win_seconds=local_win_seconds,
        max_lag_seconds=2.0,
    )

    # 2) One global lag for reference
    lag_sec_global, lag_frames_global, _, _ = estimate_global_lag(
        s_rppg=rppg_win,
        s_ppg=ppg_win,
        dt=dt,
        max_lag_seconds=2.0,
    )

    # 3) Normalized signals
    ppg_n = _normalize(ppg_win)
    rppg_n = _normalize(rppg_win)

    # 4) Build GT–CHROM pair mapping for each point
    # Positive lag_samples means CHROM lags GT:
    #   rPPG(t) is best aligned with GT(t - lag_samples * dt)
    t_pair_gt = t_win - lag_samples_curve * dt

    # We interpolate normalized GT at the paired times
    ppg_n_pair = np.interp(t_pair_gt, t_win, ppg_n)

    # 5) Create line segments that connect each CHROM point to its paired GT point
    # We use a single Scatter with NaNs to break segments
    line_x = []
    line_y = []
    for tr, yr, tg, yg in zip(t_win, rppg_n, t_pair_gt, ppg_n_pair):
        line_x.extend([tr, tg, None])
        line_y.extend([yr, yg, None])

    out_dir.mkdir(parents=True, exist_ok=True)

    # 6) Build Plotly figure with two rows
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=(
            f"{seq_id} – {label} window signals with GT–CHROM pairs",
            f"{seq_id} – {label} window local lag (frames)",
        ),
    )

    # Row 1: GT PPG (normalized) as a line
    fig.add_trace(
        go.Scatter(
            x=t_win,
            y=ppg_n,
            mode="lines",
            name="GT PPG (normalized)",
            line=dict(width=1.5),
        ),
        row=1,
        col=1,
    )

    # Row 1: CHROM rPPG (normalized) as line+markers, colored by local lag (frames)
    fig.add_trace(
        go.Scatter(
            x=t_win,
            y=rppg_n,
            mode="lines+markers",
            name="CHROM rPPG (normalized)",
            marker=dict(
                size=6,
                color=lag_frames_curve,
                colorscale="RdBu",
                colorbar=dict(
                    title="Local lag (frames)",
                ),
            ),
            line=dict(width=1.0),
        ),
        row=1,
        col=1,
    )

    # Row 1: connection lines between CHROM(t) and GT(t - lag)
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            name="GT–CHROM pair connections",
            line=dict(width=0.7, color="gray"),
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Row 2: local lag curve
    fig.add_trace(
        go.Scatter(
            x=t_win,
            y=lag_frames_curve,
            mode="lines+markers",
            name="Local lag (frames)",
        ),
        row=2,
        col=1,
    )

    # Zero-lag reference line
    fig.add_hline(
        y=0.0,
        line=dict(color="black", dash="dash", width=1),
        row=2,
        col=1,
    )

    # Global lag reference line
    fig.add_hline(
        y=lag_frames_global,
        line=dict(color="green", dash="dot", width=1),
        row=2,
        col=1,
    )

    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude (normalized)", row=1, col=1)
    fig.update_yaxes(title_text="Lag (frames)", row=2, col=1)

    fig.update_layout(
        title=(
            f"{seq_id} – {label} window "
            f"[{t_start:.2f}, {t_end:.2f}] s "
            f"(global lag ≈ {lag_frames_global:.2f} frames)"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
        ),
        template="plotly_white",
    )

    html_path = out_dir / f"{seq_id}_{label}_window_lag_heat_map.html"
    fig.write_html(html_path)
    print(f"Saved interactive {label} window visualization to: {html_path}")

    fig.show()

    print(
        f"[{seq_id}] {label} window: "
        f"global lag_frames≈{lag_frames_global:.2f}, "
        f"frames_in_window={len(t_win)}"
    )




def build_pair_table(
    t_win: np.ndarray,
    ppg_n: np.ndarray,
    rppg_n: np.ndarray,
    lag_samples_curve: np.ndarray,
    dt: float,
) -> pd.DataFrame:
    """
    We build a frame-wise GT–CHROM pairing table.

    For each frame i:
      - t_rppg[i] is the time of the CHROM sample.
      - lag_samples[i] is the local lag in samples.
      - t_gt_pair[i] = t_rppg[i] - lag_samples[i] * dt is the GT time
        that is best aligned with this CHROM sample.
      - ppg_n_pair[i] is GT amplitude at t_gt_pair[i] (via interpolation).
      - lag_frames[i] is the local lag expressed in frames.
    """
    t_win = np.asarray(t_win, dtype=np.float64)
    ppg_n = np.asarray(ppg_n, dtype=np.float64)
    rppg_n = np.asarray(rppg_n, dtype=np.float64)
    lag_samples_curve = np.asarray(lag_samples_curve, dtype=np.float64)

    # We derive FPS from dt
    fps = 1.0 / float(dt)

    # GT time for each CHROM sample based on local lag
    t_gt_pair = t_win - lag_samples_curve * dt

    # GT amplitude at the paired times
    ppg_n_pair = np.interp(t_gt_pair, t_win, ppg_n)

    # Lag in frames
    lag_frames = lag_samples_curve * fps

    df = pd.DataFrame(
        {
            "frame_idx": np.arange(len(t_win), dtype=int),
            "t_rppg": t_win,
            "rppg_n": rppg_n,
            "t_gt_pair": t_gt_pair,
            "ppg_n_pair": ppg_n_pair,
            "lag_samples": lag_samples_curve,
            "lag_frames": lag_frames,
        }
    )

    return df



def make_window_plot_2(
    info,
    t_start: float,
    t_end: float,
    label: str,
    out_dir: Path,
    local_win_seconds: float = 4.0,
):
    """
    We build an interactive Plotly figure for one window:
      - Top: GT (PPG) vs CHROM with GT–CHROM connection lines
      - Bottom: local lag curve (frames)

    We also save a frame-wise GT–CHROM pairing table to CSV.
    """
    seq_id = info["seq_id"]
    t_full = info["t_frame_full"]
    ppg_full = info["ppg_full"]
    rppg_full = info["rppg_full"]

    # 1) We select the window
    mask = (t_full >= t_start) & (t_full <= t_end)
    t_win = t_full[mask]
    ppg_win = ppg_full[mask]
    rppg_win = rppg_full[mask]

    if len(t_win) < 20:
        print(f"[{seq_id}] {label} window has too few frames ({len(t_win)}).")
        return

    dt = float(np.mean(np.diff(t_win)))

    # 2) Local lag curve (sample-wise)
    lag_samples_curve, lag_frames_curve = estimate_local_lag_curve(
        s_rppg=rppg_win,
        s_ppg=ppg_win,
        dt=dt,
        local_win_seconds=local_win_seconds,
        max_lag_seconds=2.0,
    )

    # 3) One global lag for reference
    lag_sec_global, lag_frames_global, _, _ = estimate_global_lag(
        s_rppg=rppg_win,
        s_ppg=ppg_win,
        dt=dt,
        max_lag_seconds=2.0,
    )

    # 4) Normalized signals
    ppg_n = _normalize(ppg_win)
    rppg_n = _normalize(rppg_win)

    # 5) We build and save the GT–CHROM pair table
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_df = build_pair_table(
        t_win=t_win,
        ppg_n=ppg_n,
        rppg_n=rppg_n,
        lag_samples_curve=lag_samples_curve,
        dt=dt,
    )
    csv_path = out_dir / f"{seq_id}_{label}_window_pairs.csv"
    pair_df.to_csv(csv_path, index=False)
    print(f"Saved GT–CHROM pair table to: {csv_path}")
    print("First 20 rows of the pair table:")
    print(pair_df.head(20).to_string(index=False))

    # 6) Connection lines between CHROM points and their paired GT points
    t_pair_gt = pair_df["t_gt_pair"].values
    ppg_n_pair = pair_df["ppg_n_pair"].values

    line_x = []
    line_y = []
    for tr, yr, tg, yg in zip(t_win, rppg_n, t_pair_gt, ppg_n_pair):
        line_x.extend([tr, tg, None])
        line_y.extend([yr, yg, None])

    # 7) Build the Plotly figure with two rows
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,        # <-- changed from shared_x to shared_xaxes
        row_heights=[0.65, 0.35],
        vertical_spacing=0.08,
        subplot_titles=(
            f"{seq_id} – {label} window: GT vs CHROM",
            "Local lag curve (frames)",
        ),
    )

    # Top row: GT and CHROM signals
    fig.add_trace(
        go.Scatter(
            x=t_win,
            y=ppg_n,
            mode="lines",
            name="GT PPG (normalized)",
            line=dict(width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=t_win,
            y=rppg_n,
            mode="lines",
            name="CHROM (normalized)",
            line=dict(width=2),
        ),
        row=1,
        col=1,
    )

    # Connection lines
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            name="GT–CHROM pairs",
            line=dict(width=1),
            opacity=0.4,
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Bottom row: local lag curve
    fig.add_trace(
        go.Scatter(
            x=t_win,
            y=lag_frames_curve,
            mode="lines",
            name="Local lag (frames)",
            line=dict(width=2),
        ),
        row=2,
        col=1,
    )

    # Horizontal zero line in lag subplot
    fig.add_trace(
        go.Scatter(
            x=[t_win[0], t_win[-1]],
            y=[0.0, 0.0],
            mode="lines",
            name="Zero lag",
            line=dict(width=1, dash="dash"),
        ),
        row=2,
        col=1,
    )

    # Global lag as a horizontal line
    fig.add_trace(
        go.Scatter(
            x=[t_win[0], t_win[-1]],
            y=[lag_frames_global, lag_frames_global],
            mode="lines",
            name=f"Global lag ≈ {lag_frames_global:.2f} frames",
            line=dict(width=1, dash="dot"),
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude (normalized)", row=1, col=1)
    fig.update_yaxes(title_text="Lag (frames)", row=2, col=1)

    fig.update_layout(
        title=f"{seq_id} – {label} window "
              f"(global lag ≈ {lag_frames_global:.2f} frames)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
        ),
        height=700,
    )

    html_path = out_dir / f"{seq_id}_{label}_window.html"
    fig.write_html(html_path)
    print(f"Saved interactive plot to: {html_path}")

    try:
        fig.show()
    except Exception:
        pass



def make_window_plot_with_pairs(
    info,
    t_start: float,
    t_end: float,
    label: str,
    out_dir: Path,
    local_win_seconds: float = 4.0,
):
    """
    Combined behavior of make_window_plot_1 and make_window_plot_2:

    - Uses the plotting style of make_window_plot_1:
        * Top: GT vs CHROM with:
            - CHROM points colored by local lag (frames),
            - explicit connection lines between each CHROM point and the
              GT point it is best aligned with.
        * Bottom: local lag (frames) vs. time, with zero- and global-lag
          reference lines.

    - Additionally:
        * Builds and saves a GT–CHROM pair table to CSV using build_pair_table.
    """
    seq_id = info["seq_id"]
    t_full = info["t_frame_full"]
    ppg_full = info["ppg_full"]
    rppg_full = info["rppg_full"]

    # 1) Select the window
    mask = (t_full >= t_start) & (t_full <= t_end)
    t_win = t_full[mask]
    ppg_win = ppg_full[mask]
    rppg_win = rppg_full[mask]

    if len(t_win) < 20:
        print(f"[{seq_id}] {label} window has too few frames ({len(t_win)}).")
        return

    dt = float(np.mean(np.diff(t_win)))

    # 2) Local lag curve (samples & frames)
    lag_samples_curve, lag_frames_curve = estimate_local_lag_curve(
        s_rppg=rppg_win,
        s_ppg=ppg_win,
        dt=dt,
        local_win_seconds=local_win_seconds,
        max_lag_seconds=2.0,
    )

    # 3) One global lag for reference
    lag_sec_global, lag_frames_global, _, _ = estimate_global_lag(
        s_rppg=rppg_win,
        s_ppg=ppg_win,
        dt=dt,
        max_lag_seconds=2.0,
    )

    # 4) Normalized signals (for both plot & table)
    ppg_n = _normalize(ppg_win)
    rppg_n = _normalize(rppg_win)

    # 5) Build and save GT–CHROM pair table (like make_window_plot_2)
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_df = build_pair_table(
        t_win=t_win,
        ppg_n=ppg_n,
        rppg_n=rppg_n,
        lag_samples_curve=lag_samples_curve,
        dt=dt,
    )
    csv_path = out_dir / f"{seq_id}_{label}_window_pairs.csv"
    pair_df.to_csv(csv_path, index=False)
    print(f"Saved GT–CHROM pair table to: {csv_path}")
    print("First 20 rows of the pair table:")
    print(pair_df.head(20).to_string(index=False))

    # 6) Use the pair table to build connection lines (plotting style of make_window_plot_1)
    #    We rely on the table's columns so that plot == CSV mapping exactly.
    t_pair_gt = pair_df["t_gt_pair"].values
    ppg_n_pair = pair_df["ppg_n_pair"].values

    line_x = []
    line_y = []
    for tr, yr, tg, yg in zip(t_win, rppg_n, t_pair_gt, ppg_n_pair):
        line_x.extend([tr, tg, None])
        line_y.extend([yr, yg, None])

    # 7) Build Plotly figure with two rows (layout from make_window_plot_1)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=(
            f"{seq_id}  {label} window signals with GT-CHROM pairs",
            f"{seq_id}  {label} window local lag (frames)",
        ),
    )

    # Row 1: GT PPG (normalized) as a line
    fig.add_trace(
        go.Scatter(
            x=t_win,
            y=ppg_n,
            mode="lines",
            name="GT PPG (normalized)",
            line=dict(width=1.5),
        ),
        row=1,
        col=1,
    )

    # Row 1: CHROM rPPG (normalized) as line+markers, colored by local lag (frames)
    fig.add_trace(
        go.Scatter(
            x=t_win,
            y=rppg_n,
            mode="lines+markers",
            name="CHROM rPPG (normalized)",
            marker=dict(
                size=6,
                color=lag_frames_curve,
                colorscale="RdBu",
                colorbar=dict(
                    title="Local lag (frames)",
                ),
            ),
            line=dict(width=1.0),
        ),
        row=1,
        col=1,
    )

    # Row 1: connection lines between CHROM(t) and GT(t - lag)
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            name="GT-CHROM pair connections",
            line=dict(width=0.7, color="rgba(200,0,0)"),
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Row 2: local lag curve
    fig.add_trace(
        go.Scatter(
            x=t_win,
            y=lag_frames_curve,
            mode="lines+markers",
            name="Local lag (frames)",
        ),
        row=2,
        col=1,
    )

    # Zero-lag reference line (using add_hline like in make_window_plot_1)
    fig.add_hline(
        y=0.0,
        line=dict(color="black", dash="dash", width=1),
        row=2,
        col=1,
    )

    # Global lag reference line
    fig.add_hline(
        y=lag_frames_global,
        line=dict(color="green", dash="dot", width=1),
        row=2,
        col=1,
    )

    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude (normalized)", row=1, col=1)
    fig.update_yaxes(title_text="Lag (frames)", row=2, col=1)

    fig.update_layout(
        title=(
            f"{seq_id} – {label} window "
            f"[{t_start:.2f}, {t_end:.2f}] s "
            f"(global lag ≈ {lag_frames_global:.2f} frames)"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
        ),
        template="plotly_white",
        height=700,
    )

    html_path = out_dir / f"{seq_id}_{label}_window_with_pairs.html"
    fig.write_html(html_path)
    print(f"Saved interactive {label} window visualization to: {html_path}")

    try:
        fig.show()
    except Exception:
        pass

    print(
        f"[{seq_id}] {label} window: "
        f"global lag_frames≈{lag_frames_global:.2f}, "
        f"frames_in_window={len(t_win)}"
    )

