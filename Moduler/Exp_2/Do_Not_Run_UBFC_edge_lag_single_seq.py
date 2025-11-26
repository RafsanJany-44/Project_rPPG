# Main/Experiment_Cell/UBFC_edge_lag_single_seq.py

from pathlib import Path
import numpy as np

from Main.Data_Read_Engine.ubfc_alignment import (
    load_ubfc_gt_and_fix,
    ubfc_align_ppg_to_frame_times,
)
from Main.Signal_Processing_Engine.ubfc_dataset import UBFCFrameSource
from Main.Signal_Processing_Engine.roi_central import CentralRoiExtractor
from Main.Signal_Processing_Engine.rgb_extractor import extract_rgb_timeseries
from Main.Signal_Processing_Engine.roi_central import CentralRoiExtractor
from Main.Signal_Processing_Engine.roi_face_opencv import OpenCVFaceBoxRoi
from Main.Signal_Processing_Engine.roi_face_mediapipe import MediaPipeFaceRegionsRoi


from Main.rPPG_Algorithm_Cell import rppg_chrom, rppg_pos
from Main.rPPG_Algorithm_Cell import bandpass_zero_phase


#UBFC_ROOT = Path("/media/data/rPPG/rPPG_Data/UBFC_rPPG")
UBFC_ROOT = Path(r"D:\Data\UBFC\Dataset_3")
SEQ_ID = "vid_1"   # we can change this later

WIN_LEN_SEC = 8.0
T_START_PADDING = 1.0   # we skip the first 1 s of overlap
T_END_PADDING = 1.0     # we skip the last 1 s of overlap

ROI_FRAC = 0.5          # central 50% x 50% ROI


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


def auto_detect_gt_and_video(seq_dir: Path):
    """
    We detect the single .txt (GT) and .avi (video) file in a UBFC sequence folder.
    """
    txt_files = list(seq_dir.glob("*.txt"))
    avi_files = list(seq_dir.glob("*.avi"))

    if len(txt_files) != 1:
        raise RuntimeError(
            f"[{seq_dir.name}] Expected exactly 1 .txt, found {len(txt_files)}: {txt_files}"
        )
    if len(avi_files) != 1:
        raise RuntimeError(
            f"[{seq_dir.name}] Expected exactly 1 .avi, found {len(avi_files)}: {avi_files}"
        )

    return txt_files[0], avi_files[0]


def compute_edge_lags_for_sequence(seq_id: str):
    """
    We compute CHROM–GT phase lag for:
    - one 8 s window near the start of the overlap,
    - one 8 s window near the end of the overlap.

    We return lag at start/end (in frames) and their difference.
    """
    seq_dir = UBFC_ROOT / seq_id
    if not seq_dir.is_dir():
        raise RuntimeError(f"Sequence folder not found: {seq_dir}")

    print(f"\n=== UBFC edge lag diagnostic for {seq_id} ===")

    # 1) We detect GT and video files
    gt_file, vid_file = auto_detect_gt_and_video(seq_dir)
    print(f"GT file : {gt_file.name}")
    print(f"Video   : {vid_file.name}")

    # 2) We load GT PPG and timestamps and fix non-increasing timestamps
    t_ppg_s, ppg_wave, corrected_idx = load_ubfc_gt_and_fix(gt_file)
    print(f"GT samples: {len(ppg_wave)}")
    if corrected_idx:
        print(f"  Corrected {len(corrected_idx)} non-increasing timestamps")

    # 3) We build our frame source from the video
    source = UBFCFrameSource(vid_file)
    t_frame_all = source.t_frame_s
    fps_nominal = source.fps
    print(f"Video frames: {len(t_frame_all)}, nominal FPS={fps_nominal:.3f}")
    print(f"Video time range: {t_frame_all[0]:.3f} → {t_frame_all[-1]:.3f} s")

    # 4) We compute the overlap in time between GT and video
    t_ppg_min, t_ppg_max = float(t_ppg_s[0]), float(t_ppg_s[-1])
    t_vid_min, t_vid_max = float(t_frame_all[0]), float(t_frame_all[-1])

    overlap_start = max(t_ppg_min, t_vid_min)
    overlap_end = min(t_ppg_max, t_vid_max)

    print(
        f"GT range:    [{t_ppg_min:.3f}, {t_ppg_max:.3f}] s\n"
        f"Video range: [{t_vid_min:.3f}, {t_vid_max:.3f}] s\n"
        f"Overlap:     [{overlap_start:.3f}, {overlap_end:.3f}] s"
    )

    if overlap_end - overlap_start < 2 * WIN_LEN_SEC:
        raise RuntimeError("Not enough overlap for two 8 s windows.")

    # 5) We define one start window and one end window inside the overlap
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
        """
        We compute CHROM–GT lag for a single time window.
        """
        # We extract RGB time series from frames in [t_start, t_end]
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

        # We align GT PPG to these frame times
        ppg_at_frames = ubfc_align_ppg_to_frame_times(
            t_ppg_s=t_ppg_s,
            ppg_wave=ppg_wave,
            t_frame_s=t_frame_win,
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

    # 6) We compute lag for the start and end windows
    lag_start_sec, lag_start_frames, fps_start = _compute_lag_for_window(
        t_start_start, t_end_start
    )
    lag_end_sec, lag_end_frames, fps_end = _compute_lag_for_window(
        t_start_end, t_end_end
    )

    delta_lag_frames = abs(lag_end_frames - lag_start_frames)

    print("\n=== Edge lag results ===")
    print(f"seq_id            : {seq_id}")
    print(f"lag_start_sec     : {lag_start_sec:.6f} s")
    print(f"lag_start_frames  : {lag_start_frames:.3f} frames")
    print(f"lag_end_sec       : {lag_end_sec:.6f} s")
    print(f"lag_end_frames    : {lag_end_frames:.3f} frames")
    print(f"delta_lag_frames  : {delta_lag_frames:.3f} frames")
    print(f"fps_start_est     : {fps_start:.3f} Hz")
    print(f"fps_end_est       : {fps_end:.3f} Hz")

    return {
        "seq_id": seq_id,
        "lag_start_frames": lag_start_frames,
        "lag_end_frames": lag_end_frames,
        "delta_lag_frames": delta_lag_frames,
        "fps_start": fps_start,
        "fps_end": fps_end,
    }


def main():
    _ = compute_edge_lags_for_sequence(SEQ_ID)


if __name__ == "__main__":
    main()
