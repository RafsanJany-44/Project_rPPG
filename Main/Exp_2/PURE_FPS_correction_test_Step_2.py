# Main/Exp_2/PURE_FPS_correction_test_Step_2.py


import sys
from pathlib import Path
# Add project root (Project_rPPG) to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import numpy as np

from Main.Data_Read_Engine import (
    load_pure_json,
    extract_streams_from_pure_json,
    pure_align_ppg_to_frame_times,
)

from Main.Signal_Processing_Engine.pure_dataset import PUREFrameSource
from Main.Signal_Processing_Engine.roi_central import CentralRoiExtractor
from Main.Signal_Processing_Engine.rgb_extractor import extract_rgb_timeseries
from Main.rPPG_Algorithm_Cell import rppg_chrom, bandpass_zero_phase


SEQ_ID = "01-01"   # we can change this later

WIN_LEN = 8.0
PADDING = 1.0
ROI_FRAC = 0.5


def compute_lag_in_window(
    source: PUREFrameSource,
    t_ppg_s: np.ndarray,
    ppg_wave: np.ndarray,
    t_start: float,
    t_end: float,
    fps_override: float | None = None,
):
    """
    We compute CHROM–GT lag for a single window.

    If fps_override is not None, we temporarily rebuild the frame timestamps
    using this FPS before we extract the RGB time series.
    """
    roi = CentralRoiExtractor(frac=ROI_FRAC)

    # We keep the original frame timestamps so we can restore them later
    original_t = source.t_frame_s.copy()

    # If we override FPS, we rebuild the full timeline with that FPS
    if fps_override is not None:
        n = len(original_t)
        source.t_frame_s = np.arange(n, dtype=np.float64) / float(fps_override)

    try:
        # 1) Extract RGB from frames in [t_start, t_end] using current timestamps
        t_frame_win, rgb_ts = extract_rgb_timeseries(
            source=source,
            roi_extractor=roi,
            t_start=t_start,
            t_end=t_end,
        )

        if len(t_frame_win) < 20:
            raise RuntimeError(
                f"Too few frames in [{t_start:.3f}, {t_end:.3f}] "
                f"(got {len(t_frame_win)})"
            )

        # 2) Frame dt and FPS
        dt = float(np.mean(np.diff(t_frame_win)))
        fps_est = 1.0 / dt

        # 3) Align GT PPG to these frame times
        ppg_at_frames = pure_align_ppg_to_frame_times(
            t_ppg_s=t_ppg_s,
            wave=ppg_wave,
            t_vid_s=t_frame_win,
        )

        # 4) Filter both signals
        ppg_f = bandpass_zero_phase(ppg_at_frames, fs=fps_est)
        rppg_f = bandpass_zero_phase(rppg_chrom(rgb_ts), fs=fps_est)

        # 5) Compute lag by cross-correlation (sign-invariant)
        s1 = (rppg_f - np.mean(rppg_f)) / (np.std(rppg_f) + 1e-8)
        s2 = (ppg_f - np.mean(ppg_f)) / (np.std(ppg_f) + 1e-8)

        T = len(s1)
        corr = np.correlate(s1, s2, mode="full")
        lags = np.arange(-T + 1, T)

        max_lag_samples = int(min(2.0 / dt, T - 1))
        mask = (lags >= -max_lag_samples) & (lags <= max_lag_samples)

        best = int(np.argmax(np.abs(corr[mask])))
        lag_samples = int(lags[mask][best])
        lag_sec = lag_samples * dt
        lag_frames = lag_sec * fps_est

        return lag_sec, lag_frames, fps_est

    finally:
        # We always restore the original timestamps after the computation
        source.t_frame_s = original_t


def main():
    print(f"\n=== FPS Correction Test for PURE {SEQ_ID} ===")

    # --- load JSON GT and video timestamps ---
    data = load_pure_json(SEQ_ID)
    t_ppg_ns, wave, t_vid_ns_json, hr_dev = extract_streams_from_pure_json(data)

    # We use video start (from JSON) as time origin for PPG
    t0_ns = t_vid_ns_json[0]
    t_ppg_s = (t_ppg_ns - t0_ns) * 1e-9

    # --- build frame source from PURE images ---
    source = PUREFrameSource(SEQ_ID)
    t_all = source.t_frame_s   # already relative to first frame in seconds
    n_frames = len(t_all)

    # "Original" FPS: estimated from image timestamps
    if len(t_all) < 2:
        raise RuntimeError(f"Not enough frames for FPS estimation in {SEQ_ID}")
    dt_all = np.diff(t_all)
    fps_original = 1.0 / float(np.mean(dt_all))

    # --- overlap between PPG and frames ---
    overlap_start = max(float(t_ppg_s[0]), float(t_all[0]))
    overlap_end = min(float(t_ppg_s[-1]), float(t_all[-1]))

    if overlap_end - overlap_start < 2 * WIN_LEN + 2 * PADDING:
        raise RuntimeError("Not enough overlap for two 8 s windows with padding.")

    t_start_start = overlap_start + PADDING
    t_end_start = t_start_start + WIN_LEN

    t_start_end = overlap_end - PADDING - WIN_LEN
    t_end_end = t_start_end + WIN_LEN

    # --- compute "corrected" FPS from GT duration ---
    ppg_duration = float(t_ppg_s[-1] - t_ppg_s[0])
    if ppg_duration <= 0:
        raise RuntimeError(f"Non-positive GT duration for {SEQ_ID}: {ppg_duration}")

    fps_corrected = n_frames / ppg_duration

    print(f"Original FPS (from images): {fps_original:.3f}")
    print(f"Corrected FPS (frames/GT):  {fps_corrected:.3f}\n")

    print("Computing lags with original FPS timeline...")
    lag_s_s, lag_f_s, _ = compute_lag_in_window(
        source, t_ppg_s, wave, t_start_start, t_end_start, fps_override=None
    )
    lag_s_e, lag_f_e, _ = compute_lag_in_window(
        source, t_ppg_s, wave, t_start_end, t_end_end, fps_override=None
    )

    print("Computing lags with corrected FPS timeline...")
    lag_s_s_c, lag_f_s_c, _ = compute_lag_in_window(
        source, t_ppg_s, wave, t_start_start, t_end_start,
        fps_override=fps_corrected,
    )
    lag_s_e_c, lag_f_e_c, _ = compute_lag_in_window(
        source, t_ppg_s, wave, t_start_end, t_end_end,
        fps_override=fps_corrected,
    )

    # --- print summary ---
    print("\n=== Lag Comparison (PURE) ===")
    print(f"Start lag original   : {lag_f_s:.2f} frames")
    print(f"End lag original     : {lag_f_e:.2f} frames")
    print(f"Δlag original        : {abs(lag_f_e - lag_f_s):.2f} frames\n")

    print(f"Start lag corrected  : {lag_f_s_c:.2f} frames")
    print(f"End  lag corrected   : {lag_f_e_c:.2f} frames")
    print(f"Δlag corrected       : {abs(lag_f_e_c - lag_f_s_c):.2f} frames\n")

    if abs(lag_f_e_c - lag_f_s_c) < abs(lag_f_e - lag_f_s):
        print("→ FPS correction improves alignment")
    else:
        print("→ FPS correction does NOT improve alignment")


if __name__ == "__main__":
    main()
