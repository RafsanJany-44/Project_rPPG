# Main/UBFC_show_roi_debug.py
#
# Debug script to visualize which part of the frame is used as ROI
# (Central / OpenCV face / MediaPipe face) and save a PNG.

from pathlib import Path
import argparse

import cv2
import numpy as np

from ubfc_dataset import UBFCFrameSource  
from roi_central import CentralRoiExtractor
from roi_face_opencv import OpenCVFaceBoxRoi
from roi_face_mediapipe import MediaPipeFaceRegionsRoi

# ----- CONFIG -----
UBFC_ROOT = Path(r"D:\Data\UBFC\Dataset_3")  # adjust if needed
ROI_MODE = "mediapipe_face"  # "central", "opencv_face", "mediapipe_face"


def auto_detect_video(seq_dir: Path) -> Path:
    """Detect the single .avi video file in UBFC sequence folder."""
    avi_files = list(seq_dir.glob("*.avi"))
    if len(avi_files) != 1:
        raise RuntimeError(
            f"[{seq_dir.name}] Expected exactly 1 .avi, found {len(avi_files)}: {avi_files}"
        )
    return avi_files[0]


def build_roi_extractor():
    """Return the ROI extractor according to ROI_MODE."""
    if ROI_MODE == "central":
        # central 50% x 50%
        return CentralRoiExtractor(frac=0.5)
    elif ROI_MODE == "opencv_face":
        return OpenCVFaceBoxRoi()
    elif ROI_MODE == "mediapipe_face":
        # simple face-mesh–based forehead+cheeks region
        return MediaPipeFaceRegionsRoi()
    else:
        raise ValueError(f"Unknown ROI_MODE: {ROI_MODE}")


def extract_frame_bgr(video_path: Path, frame_idx: int) -> np.ndarray:
    """Read a single BGR frame using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx < 0 or frame_idx >= total_frames:
        cap.release()
        raise RuntimeError(
            f"frame_idx={frame_idx} is out of range [0, {total_frames - 1}]"
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")

    return frame  # BGR


def make_side_by_side_debug(frame_bgr: np.ndarray, roi_bgr: np.ndarray) -> np.ndarray:
    """Create a side-by-side (original | ROI) debug image."""
    h, w = frame_bgr.shape[:2]

    # resize ROI to have same height as original frame
    roi_h, roi_w = roi_bgr.shape[:2]
    scale = h / float(roi_h)
    new_w = int(roi_w * scale)
    roi_resized = cv2.resize(roi_bgr, (new_w, h), interpolation=cv2.INTER_AREA)

    # stack images horizontally
    debug_img = np.hstack([frame_bgr, roi_resized])
    return debug_img


def main():
    parser = argparse.ArgumentParser(
        description="Visualize which part of the UBFC frame is used as ROI."
    )
    parser.add_argument(
        "--seq",
        type=str,
        required=True,
        help="Sequence ID, e.g., vid_25",
    )
    parser.add_argument(
        "--frame_idx",
        type=int,
        required=True,
        help="Zero-based frame index to visualize (0 .. N-1).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="Figures/UBFC_ROI_Debug",
        help="Output directory for debug PNGs (relative or absolute).",
    )

    args = parser.parse_args()

    seq_id = args.seq
    frame_idx = int(args.frame_idx)
    root = UBFC_ROOT
    out_root = Path(args.out)

    print("\n=======================================")
    print(" UBFC ROI DEBUG")
    print("  seq_id    :", seq_id)
    print("  frame_idx :", frame_idx)
    print("  UBFC root :", root)
    print("  ROI_MODE  :", ROI_MODE)
    print("=======================================")

    seq_dir = root / seq_id
    if not seq_dir.is_dir():
        raise RuntimeError(f"Sequence folder not found: {seq_dir}")

    video_path = auto_detect_video(seq_dir)
    print(f"Detected video: {video_path}")

    # 1) Read original frame
    frame_bgr = extract_frame_bgr(video_path, frame_idx)

    # 2) Build ROI extractor and extract ROI
    roi_extractor = build_roi_extractor()
    roi_bgr = roi_extractor.extract(frame_bgr)

    print(f"Original frame shape: {frame_bgr.shape}")
    print(f"ROI frame shape      : {roi_bgr.shape}")

    # 3) Build side-by-side image (original | ROI)
    debug_img = make_side_by_side_debug(frame_bgr, roi_bgr)

    # 4) Save to disk
    out_dir = out_root / seq_id
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{seq_id}_frame{frame_idx:05d}_roi_debug.png"
    cv2.imwrite(str(out_path), debug_img)

    print("=======================================")
    print(" ROI DEBUG SAVED AT:")
    print("   ", out_path.resolve())
    print("=======================================\n")


    try:
        cv2.imshow("ROI Debug (left = original, right = ROI)", debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        # In case of headless environment
        pass


if __name__ == "__main__":
    main()
