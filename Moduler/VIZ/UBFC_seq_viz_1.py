# Main/VIZ/UBFC_sequence_viz_plotly.py

from pathlib import Path
import argparse

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


import sys
from pathlib import Path
# Add project root (Project_rPPG) to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



from Main.VIZ.Plot_Me import make_window_plot_1, make_window_plot_2


from Main.Data_Read_Engine.ubfc_alignment import (
    load_ubfc_gt_and_fix,
    ubfc_align_ppg_to_frame_times,
)

from Main.Signal_Processing_Engine.ubfc_dataset import UBFCFrameSource
from Main.Signal_Processing_Engine.roi_central import CentralRoiExtractor
from Main.Signal_Processing_Engine.roi_central import CentralRoiExtractor
from Main.Signal_Processing_Engine.roi_face_opencv import OpenCVFaceBoxRoi
from Main.Signal_Processing_Engine.roi_face_mediapipe import MediaPipeFaceRegionsRoi
from Main.Signal_Processing_Engine.rgb_extractor import extract_rgb_timeseries
from Main.rPPG_Algorithm_Cell import rppg_chrom, bandpass_zero_phase


#UBFC_ROOT = Path(r"D:\Data\UBFC\Dataset_3")
UBFC_ROOT = Path("/media/data/rPPG/rPPG_Data/UBFC_rPPG")

WIN_LEN = 8.0
PADDING = 1.0
ROI_FRAC = 0.5
ROI_MODE = "opencv_face"  # "central", "opencv_face", or "mediapipe_face"



def build_roi_extractor():
    if ROI_MODE == "central":
        # We use a simple central rectangle
        return CentralRoiExtractor(frac=ROI_FRAC)
    elif ROI_MODE == "opencv_face":
        # We use OpenCV face detector, then a rectangular face ROI
        return OpenCVFaceBoxRoi()
    
    elif ROI_MODE == "mediapipe_face":
        # We use MediaPipe facial landmarks and define a region
        # (for example forehead + cheeks; adjust args based on your class)
        return MediaPipeFaceRegionsRoi(
            use_forehead=True,
            use_left_cheek=True,
            use_right_cheek=True,
        )
    else:
        raise ValueError(f"Unknown ROI_MODE: {ROI_MODE}")





def auto_detect_gt_and_video(seq_dir: Path):
    """We detect the single .txt and .avi files in a UBFC sequence folder."""
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





def prepare_full_signals(seq_id: str, root: Path):
    """
    We load GT and video, build overlap, and compute frame-synchronous
    GT and CHROM signals inside the overlap.
    """
    seq_dir = root / seq_id
    if not seq_dir.is_dir():
        raise RuntimeError(f"Sequence folder not found: {seq_dir}")

    gt_file, vid_file = auto_detect_gt_and_video(seq_dir)

    t_ppg_s, ppg_wave, corrected_idx = load_ubfc_gt_and_fix(gt_file)

    source = UBFCFrameSource(vid_file)
    t_all = source.t_frame_s
    fps_nominal = float(source.fps)

    overlap_start = float(max(t_ppg_s[0], t_all[0]))
    overlap_end = float(min(t_ppg_s[-1], t_all[-1]))

    if overlap_end <= overlap_start:
        raise RuntimeError("No overlap between GT and video.")

    #roi = CentralRoiExtractor(frac=ROI_FRAC)
    roi = build_roi_extractor()

    t_frame_full, rgb_full = extract_rgb_timeseries(
        source=source,
        roi_extractor=roi,
        t_start=overlap_start,
        t_end=overlap_end,
    )

    if len(t_frame_full) < 20:
        raise RuntimeError("Too few frames in overlap region.")

    dt_full = float(np.mean(np.diff(t_frame_full)))
    fps_est = 1.0 / dt_full

    ppg_full = ubfc_align_ppg_to_frame_times(
        t_ppg_s=t_ppg_s,
        ppg_wave=ppg_wave,
        t_frame_s=t_frame_full,
    )

    ppg_filt = bandpass_zero_phase(ppg_full, fs=fps_est)
    rppg_raw = rppg_chrom(rgb_full)
    rppg_filt = bandpass_zero_phase(rppg_raw, fs=fps_est)

    info = {
        "seq_id": seq_id,
        "t_ppg_s": t_ppg_s,
        "ppg_wave": ppg_wave,
        "t_frame_full": t_frame_full,
        "ppg_full": ppg_filt,
        "rppg_full": rppg_filt,
        "fps_nominal": fps_nominal,
        "fps_est": fps_est,
        "overlap_start": overlap_start,
        "overlap_end": overlap_end,
    }
    return info




def main():
    parser = argparse.ArgumentParser(
        description="Interactive UBFC visualization with local lag per frame."
    )
    parser.add_argument(
        "--seq",
        type=str,
        required=False,
        default="vid_15",
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
        default="Figures/UBFC_VIZ_plotly",
        help="Output directory for HTML figures.", 
    )
    parser.add_argument(
        "--win_len",
        type=float,
        default=WIN_LEN,
        help="Window length in seconds (default 8.0).",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=PADDING,
        help="Padding from start and end of overlap in seconds (default 1.0).",
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

    print(f"\n=== UBFC Plotly visualization for {seq_id} ===")
    print(f"UBFC root: {root}")

    info = prepare_full_signals(seq_id, root)

    overlap_start = info["overlap_start"]
    overlap_end = info["overlap_end"]
    win_len = float(args.win_len)
    pad = float(args.padding)

    t_start_start = overlap_start + pad
    t_end_start = t_start_start + win_len

    t_start_end = overlap_end - pad - win_len
    t_end_end = t_start_end + win_len

    print(
        f"Overlap: [{overlap_start:.3f}, {overlap_end:.3f}] s\n"
        f"Start window: [{t_start_start:.3f}, {t_end_start:.3f}] s\n"
        f"End window:   [{t_start_end:.3f}, {t_end_end:.3f}] s"
    )

    # We build start-window visualization
    make_window_plot_1(
        info,
        t_start=t_start_start,
        t_end=t_end_start,
        label="start",
        out_dir=out_dir,
        local_win_seconds=float(args.local_win),
    )

    # We build end-window visualization
    make_window_plot_1(
        info,
        t_start=t_start_end,
        t_end=t_end_end,
        label="end",
        out_dir=out_dir,
        local_win_seconds=float(args.local_win),
    )

    # We build start-window visualization
    make_window_plot_2(
        info,
        t_start=t_start_start,
        t_end=t_end_start,
        label="start",
        out_dir=out_dir,
        local_win_seconds=float(args.local_win),
    )

    # We build end-window visualization
    make_window_plot_2(
        info,
        t_start=t_start_end,
        t_end=t_end_end,
        label="end",
        out_dir=out_dir,
        local_win_seconds=float(args.local_win),
    )



    print("\nInteractive visualization complete.")


if __name__ == "__main__":
    main()
