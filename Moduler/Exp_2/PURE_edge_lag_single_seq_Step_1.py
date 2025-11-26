# Main/Exp_2/PURE_edge_lag_single_seq.py


import sys
from pathlib import Path
# Add project root (Project_rPPG) to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



from typing import Dict
import numpy as np

from Main.Data_Read_Engine import (
    DEFAULT_SEQ_ID,
    load_pure_json,
    extract_streams_from_pure_json,
    pure_align_ppg_to_frame_times,
)
from Main.Signal_Processing_Engine.pure_dataset import PUREFrameSource
from Main.Signal_Processing_Engine.roi_central import CentralRoiExtractor
from Main.Signal_Processing_Engine.rgb_extractor import extract_rgb_timeseries
from Main.rPPG_Algorithm_Cell import rppg_chrom, bandpass_zero_phase


SEQ_ID = DEFAULT_SEQ_ID  # "01-01" etc.

WIN_LEN_SEC = 8.0
T_START_PADDING = 1.0   # we skip the first 1 s of overlap
T_END_PADDING = 1.0     # we skip the last 1 s of overlap

ROI_FRAC = 0.5          # central 50% x 50% ROI

# "Small drift" threshold in frames (similar interpretation as UBFC)
SMALL_DRIFT_THR_FRAMES = 2.0


def _normalize(sig: np.ndarray) -> np.ndarray:
    """We normalize a 1D signal to zero mean and unit variance (if std > 0)."""
    sig = np.asarray(sig, dtype=np.float32)
    sig = sig - np.mean(sig)
    std = np.std(sig)
    if std == 0:
        return sig
    return sig / std


def estimate_phase_lag_sign_invariant(
    s_rppg: np.ndarray,
    s_ppg: np.ndarray,
    dt: float,
    max_lag_seconds: float = 2.0,
) -> float:
    """
    We estimate phase lag using cross-correlation and allow for sign flips.
    We return lag in seconds, where a positive value means rPPG lags PPG.
    """
    s1 = _normalize(s_rppg)
    s2 = _normalize(s_ppg)

    T = len(s1)
    if T < 2:
        return 0.0

    corr = np.correlate(s1, s2, mode="full")
    lags = np.arange(-T + 1, T)

    max_lag_samples = int(max_lag_seconds / dt)
    max_lag_samples = min(max_lag_samples, T - 1)

    mask = (lags >= -max_lag_samples) & (lags <= max_lag_samples)
    corr_win = corr[mask]
    lags_win = lags[mask]

    best_idx = int(np.argmax(np.abs(corr_win)))
    best_lag_samples = int(lags_win[best_idx])

    return float(best_lag_samples * dt)


def compute_edge_lags_for_sequence(seq_id: str = SEQ_ID) -> Dict[str, float]:
    """
    We compute CHROM–GT phase lag for PURE:
    - one 8 s window near the start of the overlap,
    - one 8 s window near the end of the overlap.

    We return lag at start/end (in frames), their difference,
    the remaining frames at the tail, and our decision:
    TRIM / FPS_ADJUST / UNCERTAIN.
    """
    print(f"\n=== PURE edge lag diagnostic for {seq_id} (CHROM) ===")

    # 1) We load JSON and extract PPG and video timestamps
    data = load_pure_json(seq_id)
    t_ppg_ns, wave, t_vid_ns_json, hr_dev = extract_streams_from_pure_json(data)

    # We use the first video timestamp as common time origin
    t0_ns = t_vid_ns_json[0]
    t_ppg_s = (t_ppg_ns - t0_ns) * 1e-9
    t_vid_s_json = (t_vid_ns_json - t0_ns) * 1e-9

    print(f"GT samples: {len(wave)}")

    # 2) We build our frame source from the PURE images
    # 2) We build our frame source from the PURE images
    source = PUREFrameSource(seq_id)
    t_frame_all = source.t_frame_s          # seconds, relative to first frame

    # We estimate nominal FPS from the full timeline
    if len(t_frame_all) < 2:
        raise RuntimeError(f"Not enough frames for FPS estimation in {seq_id}")

    dt_all = np.diff(t_frame_all)
    fps_nominal = 1.0 / float(np.mean(dt_all))

    print(f"Frames from images: {len(t_frame_all)}, nominal FPS≈{fps_nominal:.3f}")
    print(f"Image time range: {t_frame_all[0]:.3f} → {t_frame_all[-1]:.3f} s")


    # 3) We compute the overlap in time between PPG and frames (both in seconds)
    t_ppg_min, t_ppg_max = float(t_ppg_s[0]), float(t_ppg_s[-1])
    t_vid_min, t_vid_max = float(t_frame_all[0]), float(t_frame_all[-1])

    overlap_start = max(t_ppg_min, t_vid_min)
    overlap_end = min(t_ppg_max, t_vid_max)

    print(
        f"GT range (JSON) : [{t_ppg_min:.3f}, {t_ppg_max:.3f}] s\n"
        f"Frame range     : [{t_vid_min:.3f}, {t_vid_max:.3f}] s\n"
        f"Overlap         : [{overlap_start:.3f}, {overlap_end:.3f}] s"
    )

    if overlap_end - overlap_start < 2 * WIN_LEN_SEC:
        raise RuntimeError("Not enough overlap for two 8 s windows.")

    # 4) We define one start window and one end window inside the overlap
    t_start_start = overlap_start + T_START_PADDING
    t_end_start = t_start_start + WIN_LEN_SEC

    t_start_end = overlap_end - T_END_PADDING - WIN_LEN_SEC
    t_end_end = t_start_end + WIN_LEN_SEC

    print(
        f"\nStart window: [{t_start_start:.3f}, {t_end_start:.3f}] s\n"
        f"End window:   [{t_start_end:.3f}, {t_end_end:.3f}] s"
    )

    roi_extractor = CentralRoiExtractor(frac=ROI_FRAC)

    def _compute_lag_for_window(t_start: float, t_end: float):
        """We compute CHROM–GT lag for a single time window."""
        t_frame_win, rgb_ts = extract_rgb_timeseries(
            source=source,
            roi_extractor=roi_extractor,
            t_start=t_start,
            t_end=t_end,
        )

        if len(t_frame_win) < 20:
            raise RuntimeError(
                f"Too few frames in window [{t_start:.3f}, {t_end:.3f}] "
                f"(got {len(t_frame_win)})"
            )

        dt_frames = np.diff(t_frame_win)
        dt_mean = float(np.mean(dt_frames))
        fps_est = 1.0 / dt_mean

        # We align GT PPG to these frame times (uses PURE-specific alignment)
        ppg_at_frames = pure_align_ppg_to_frame_times(
            t_ppg_s=t_ppg_s,
            wave=wave,
            t_vid_s=t_frame_win,
        )

        # We bandpass-filter both signals in the cardiac band
        fs = fps_est
        ppg_filt = bandpass_zero_phase(ppg_at_frames, fs=fs)
        rppg_raw = rppg_chrom(rgb_ts)
        rppg_filt = bandpass_zero_phase(rppg_raw, fs=fs)

        # We estimate the lag between the filtered signals
        lag_sec = estimate_phase_lag_sign_invariant(
            s_rppg=rppg_filt,
            s_ppg=ppg_filt,
            dt=dt_mean,
            max_lag_seconds=2.0,
        )

        lag_frames = lag_sec * fps_est
        return lag_sec, lag_frames, fps_est

    # 5) We compute lag for the start and end windows
    lag_start_sec, lag_start_frames, fps_start = _compute_lag_for_window(
        t_start_start, t_end_start
    )
    lag_end_sec, lag_end_frames, fps_end = _compute_lag_for_window(
        t_start_end, t_end_end
    )

    delta_lag_frames = abs(lag_end_frames - lag_start_frames)

    # 6) We compute remaining frames at the tail and make a simple decision
    #    (frames after GT ends, expressed as frames)
    extra_tail_time = max(0.0, t_vid_max - t_ppg_max)
    remaining_frames = extra_tail_time * fps_nominal

    if delta_lag_frames <= SMALL_DRIFT_THR_FRAMES:
        decision = "TRIM"
    elif abs(delta_lag_frames - remaining_frames) <= SMALL_DRIFT_THR_FRAMES:
        decision = "FPS_ADJUST"
    else:
        decision = "UNCERTAIN"

    print("\n=== Edge lag results (PURE) ===")
    print(f"seq_id            : {seq_id}")
    print(f"lag_start_sec     : {lag_start_sec:.6f} s")
    print(f"lag_start_frames  : {lag_start_frames:.3f} frames")
    print(f"lag_end_sec       : {lag_end_sec:.6f} s")
    print(f"lag_end_frames    : {lag_end_frames:.3f} frames")
    print(f"delta_lag_frames  : {delta_lag_frames:.3f} frames")
    print(f"fps_start_est     : {fps_start:.3f} Hz")
    print(f"fps_end_est       : {fps_end:.3f} Hz")
    print(f"extra_tail_time   : {extra_tail_time:.3f} s")
    print(f"remaining_frames  : {remaining_frames:.3f} frames")
    print(f"decision          : {decision}")

    return {
        "seq_id": seq_id,
        "lag_start_frames": float(lag_start_frames),
        "lag_end_frames": float(lag_end_frames),
        "delta_lag_frames": float(delta_lag_frames),
        "fps_start": float(fps_start),
        "fps_end": float(fps_end),
        "extra_tail_time": float(extra_tail_time),
        "remaining_frames": float(remaining_frames),
        "decision": decision,
    }


def main():
    _ = compute_edge_lags_for_sequence(SEQ_ID)


if __name__ == "__main__":
    main()
