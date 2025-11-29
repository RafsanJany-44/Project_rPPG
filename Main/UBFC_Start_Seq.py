# Main/VIZ/UBFC_sequence_viz_plotly.py

from pathlib import Path
import argparse

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plot_me import make_window_plot_1, make_window_plot_2, make_window_plot_with_pairs
from ubfc_alignment import (
    load_ubfc_gt_and_fix,
    ubfc_align_ppg_to_frame_times,
)

from ubfc_dataset import UBFCFrameSource
from roi_central import CentralRoiExtractor
from roi_face_opencv import OpenCVFaceBoxRoi
from roi_face_mediapipe import MediaPipeFaceRegionsRoi
from roi_face_mediapipe_advanced_full_face import MediaPipeFaceMeshRoi as MediaPipeFaceMeshRoi_FullFace
from roi_face_mediapipe_advanced_convexHull import MediaPipeFaceMeshRoi  as MediaPipeFaceMeshRoi_ConvexHull
from roi_face_mediapipe_advanced_f_c import MediaPipeFaceMeshRoi  as MediaPipeFaceMeshRoi_F_C
from rgb_extractor import extract_rgb_timeseries
from rPPG_Algorithm_Cell import rppg_chrom, bandpass_zero_phase




UBFC_ROOT = Path(r"D:\Data\UBFC\Dataset_3")
#UBFC_ROOT = Path("/media/data/rPPG/rPPG_Data/UBFC_rPPG")
WIN_LEN = 8.0
PADDING = 1.0
ROI_FRAC = 0.5
ROI_MODE = "mediapipe_f_c"  # "central", "opencv_face", "mediapipe_face", or "mediapipe_full", "mediapipe_convexHull", "mediapipe_f_c"



def build_roi_extractor():
    """
    Build the ROI extractor according to ROI_MODE.

    Modes
    -----
    central        : simple central rectangle.
    opencv_face    : face bounding box from OpenCV detector.
    mediapipe_face : earlier rectangular forehead+cheeks region.
    mediapipe_mesh : advanced free-form mesh-based ROI.
    """
    if ROI_MODE == "central":
        return CentralRoiExtractor(frac=0.5)

    if ROI_MODE == "opencv_face":
        return OpenCVFaceBoxRoi()

    if ROI_MODE == "mediapipe_face":
        return MediaPipeFaceRegionsRoi()

    if ROI_MODE == "mediapipe_full":
        # Use default shape configuration from roi_face_mediapipe_advanced.py
        return MediaPipeFaceMeshRoi_FullFace(
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    
    if ROI_MODE == "mediapipe_convexHull":
        return MediaPipeFaceMeshRoi_ConvexHull(
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    if ROI_MODE == "mediapipe_f_c":
        return MediaPipeFaceMeshRoi_F_C(
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

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
        #required=True,
        default= 'vid_25',
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

    make_window_plot_with_pairs(
        info,
        t_start=t_start_start,
        t_end=t_end_start,
        label="start",
        out_dir=out_dir,
        local_win_seconds=float(args.local_win),
    )

    make_window_plot_with_pairs(
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
