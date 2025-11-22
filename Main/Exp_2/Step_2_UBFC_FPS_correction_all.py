# Main/Exp_2/Step_2_UBFC_FPS_correction_all

from pathlib import Path
import numpy as np

import sys
from pathlib import Path
# Add project root (Project_rPPG) to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from Main.Data_Read_Engine.ubfc_alignment import (
    load_ubfc_gt_and_fix,
    ubfc_align_ppg_to_frame_times,
)
from Main.Signal_Processing_Engine.ubfc_dataset import UBFCFrameSource
from Main.Signal_Processing_Engine.roi_central import CentralRoiExtractor
from Main.Signal_Processing_Engine.rgb_extractor import extract_rgb_timeseries
from Main.rPPG_Algorithm_Cell import rppg_chrom, bandpass_zero_phase


#UBFC_ROOT = Path(r"D:\Data\UBFC\Dataset_3")
UBFC_ROOT = Path("/media/data/rPPG/rPPG_Data/UBFC_rPPG")

WIN_LEN = 8.0
PADDING = 1.0
ROI_FRAC = 0.5


def list_ubfc_sequences(root: Path):
    """
    We list UBFC sequence folders like vid_1, vid_2, ...
    """
    seq_ids = []
    for p in root.iterdir():
        if p.is_dir() and p.name.lower().startswith("vid_"):
            seq_ids.append(p.name)
    return sorted(seq_ids)


def auto_detect_gt_and_video(seq_dir: Path):
    """
    We detect the single .txt (GT) and .avi (video) file inside a UBFC sequence folder.
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


def compute_lag_in_window(source: UBFCFrameSource,
                          t_ppg_s: np.ndarray,
                          ppg_wave: np.ndarray,
                          t_start: float,
                          t_end: float,
                          fps_override: float | None = None):
    """
    We compute CHROM–GT lag for a single window.

    If fps_override is not None, we temporarily rebuild the frame timestamps
    using this FPS before we extract the RGB time series.
    """
    roi = CentralRoiExtractor(frac=ROI_FRAC)

    # We keep the original frame timestamps so we can restore them
    original_t = source.t_frame_s.copy()

    # If we override FPS, we rebuild the full timeline with that FPS
    if fps_override is not None:
        n = len(original_t)
        source.t_frame_s = np.arange(n, dtype=np.float64) / float(fps_override)

    try:
        # 1) Extract RGB from frames in [t_start, t_end]
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

        # 3) Align GT to these frame times
        ppg_at_frames = ubfc_align_ppg_to_frame_times(
            t_ppg_s=t_ppg_s,
            ppg_wave=ppg_wave,
            t_frame_s=t_frame_win,
        )

        # 4) Filter both signals
        ppg_f = bandpass_zero_phase(ppg_at_frames, fs=fps_est)
        rppg_f = bandpass_zero_phase(rppg_chrom(rgb_ts), fs=fps_est)

        # 5) We compute lag by sign-invariant cross-correlation
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

        return lag_frames

    finally:
        # We restore the original timestamps
        source.t_frame_s = original_t


def analyze_sequence(seq_id: str):
    """
    We run the FPS correction test for a single sequence.
    We return a dict with original and corrected Δlag.
    """
    seq_dir = UBFC_ROOT / seq_id
    if not seq_dir.is_dir():
        raise RuntimeError(f"Sequence folder not found: {seq_dir}")

    gt_file, vid_file = auto_detect_gt_and_video(seq_dir)

    # GT
    t_ppg_s, ppg_wave, _ = load_ubfc_gt_and_fix(gt_file)

    # Video
    source = UBFCFrameSource(vid_file)
    t_all = source.t_frame_s
    fps_original = float(source.fps)
    n_frames = len(t_all)

    # Overlap
    overlap_start = max(float(t_ppg_s[0]), float(t_all[0]))
    overlap_end = min(float(t_ppg_s[-1]), float(t_all[-1]))

    if overlap_end - overlap_start < 2 * WIN_LEN + 2 * PADDING:
        raise RuntimeError("Not enough overlap for two 8 s windows.")

    t_start_start = overlap_start + PADDING
    t_end_start = t_start_start + WIN_LEN

    t_start_end = overlap_end - PADDING - WIN_LEN
    t_end_end = t_start_end + WIN_LEN

    # Corrected FPS from GT duration
    ppg_duration = float(t_ppg_s[-1] - t_ppg_s[0])
    if ppg_duration <= 0:
        raise RuntimeError(f"Non-positive GT duration for {seq_id}: {ppg_duration}")

    fps_corrected = n_frames / ppg_duration

    # Lags with original FPS
    lag_start_orig = compute_lag_in_window(
        source, t_ppg_s, ppg_wave, t_start_start, t_end_start, fps_override=None
    )
    lag_end_orig = compute_lag_in_window(
        source, t_ppg_s, ppg_wave, t_start_end, t_end_end, fps_override=None
    )
    delta_orig = abs(lag_end_orig - lag_start_orig)

    # Lags with corrected FPS
    lag_start_corr = compute_lag_in_window(
        source, t_ppg_s, ppg_wave, t_start_start, t_end_start,
        fps_override=fps_corrected,
    )
    lag_end_corr = compute_lag_in_window(
        source, t_ppg_s, ppg_wave, t_start_end, t_end_end,
        fps_override=fps_corrected,
    )
    delta_corr = abs(lag_end_corr - lag_start_corr)

    improved = delta_corr < delta_orig

    return {
        "seq_id": seq_id,
        "fps_original": fps_original,
        "fps_corrected": fps_corrected,
        "lag_start_orig": lag_start_orig,
        "lag_end_orig": lag_end_orig,
        "delta_orig": delta_orig,
        "lag_start_corr": lag_start_corr,
        "lag_end_corr": lag_end_corr,
        "delta_corr": delta_corr,
        "improved": improved,
    }


def main():
    print(f"UBFC root: {UBFC_ROOT}")
    seq_ids = list_ubfc_sequences(UBFC_ROOT)
    print(f"Found {len(seq_ids)} sequences: {seq_ids}\n")

    results = []

    for seq_id in seq_ids:
        print(f"\n=== Analyzing {seq_id} ===")
        try:
            info = analyze_sequence(seq_id)
            results.append(info)

            print(
                f"FPS orig={info['fps_original']:.3f}, "
                f"FPS corr={info['fps_corrected']:.3f}"
            )
            print(
                f"  Δlag orig={info['delta_orig']:.2f} frames, "
                f"Δlag corr={info['delta_corr']:.2f} frames, "
                f"improved={info['improved']}"
            )
        except Exception as e:
            print(f"[{seq_id}] ERROR: {e}")

    if not results:
        print("\nNo successful sequences.")
        return

    print("\n================ FPS correction effect summary ================")
    print("seq_id    Δorig   Δcorr   improved?")
    for r in results:
        print(
            f"{r['seq_id']:8s}"
            f"{r['delta_orig']:7.2f}"
            f"{r['delta_corr']:8.2f}"
            f"{str(r['improved']):>11s}"
        )

    improved_count = sum(r["improved"] for r in results)
    worse_count = sum((not r["improved"]) for r in results)

    print("\nTotal sequences analyzed :", len(results))
    print("Sequences where FPS helps:", improved_count)
    print("Sequences where FPS does not help:", worse_count)


if __name__ == "__main__":
    main()
