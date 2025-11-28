# Main/UBFC_show_roi_debug.py
#
# Debug script to visualize which part of the frame is used as ROI
# (Central / OpenCV face / MediaPipe face / MediaPipe mesh) and save:
# - side-by-side image (original | ROI),
# - optional mask / overlay (for mesh),
# - basic RGB statistics of the ROI,
# - a small CSV with sampled ROI pixels.

from pathlib import Path
import argparse

import cv2
import numpy as np

from ubfc_dataset import UBFCFrameSource
from roi_central import CentralRoiExtractor
from roi_face_opencv import OpenCVFaceBoxRoi
from roi_face_mediapipe import MediaPipeFaceRegionsRoi
from roi_face_mediapipe_advanced import MediaPipeFaceMeshRoi

# ----- CONFIG -----
UBFC_ROOT = Path(r"D:\Data\UBFC\Dataset_3")  # adjust if needed
# Options: "central", "opencv_face", "mediapipe_face", "mediapipe_mesh"
ROI_MODE = "mediapipe_mesh"


def auto_detect_video(seq_dir: Path) -> Path:
    """Detect the single .avi video file in UBFC sequence folder."""
    avi_files = list(seq_dir.glob("*.avi"))
    if len(avi_files) != 1:
        raise RuntimeError(
            f"[{seq_dir.name}] Expected exactly 1 .avi, found {len(avi_files)}: {avi_files}"
        )
    return avi_files[0]


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

    if ROI_MODE == "mediapipe_mesh":
        # Use default shape configuration from roi_face_mediapipe_advanced.py
        return MediaPipeFaceMeshRoi(
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

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

    roi_h, roi_w = roi_bgr.shape[:2]
    if roi_h <= 0 or roi_w <= 0:
        # Fallback in case ROI is empty
        return frame_bgr.copy()

    scale = h / float(roi_h)
    new_w = int(roi_w * scale)
    roi_resized = cv2.resize(roi_bgr, (new_w, h), interpolation=cv2.INTER_AREA)

    debug_img = np.hstack([frame_bgr, roi_resized])
    return debug_img


def compute_and_save_roi_rgb_stats(
    frame_bgr: np.ndarray,
    roi_bgr: np.ndarray,
    out_dir: Path,
    seq_id: str,
    frame_idx: int,
    mask: np.ndarray | None = None,
    max_sample_pixels: int = 1000,
):
    """
    Compute basic RGB statistics from ROI and save a small CSV with sampled pixels.

    If mask is None:
        We treat roi_bgr as a cropped region and flatten all pixels.

    If mask is not None:
        We take pixels from the original frame where mask == 1.
    """
    h, w = frame_bgr.shape[:2]

    if mask is not None:
        # Use mask on original frame to get ROI pixels
        roi_pixels = frame_bgr[mask > 0]  # shape (N, 3), BGR
    else:
        # Use cropped ROI image directly
        roi_pixels = roi_bgr.reshape(-1, 3)  # BGR

    if roi_pixels.size == 0:
        print("No ROI pixels found (roi_pixels is empty).")
        return

    # Convert to float for statistics
    roi_pixels_f = roi_pixels.astype(np.float32)

    mean_bgr = roi_pixels_f.mean(axis=0)
    std_bgr = roi_pixels_f.std(axis=0)

    n_pixels = roi_pixels.shape[0]

    print("\n--- ROI RGB STATISTICS (BGR order) ---")
    print(f"Number of ROI pixels: {n_pixels}")
    print(
        f"Mean  B,G,R: {mean_bgr[0]:.2f}, {mean_bgr[1]:.2f}, {mean_bgr[2]:.2f}"
    )
    print(
        f"Std   B,G,R: {std_bgr[0]:.2f}, {std_bgr[1]:.2f}, {std_bgr[2]:.2f}"
    )

    # Save summary to TXT
    summary_path = out_dir / f"{seq_id}_frame{frame_idx:05d}_roi_rgb_stats.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("ROI RGB statistics (BGR order)\n")
        f.write(f"seq_id        : {seq_id}\n")
        f.write(f"frame_idx     : {frame_idx}\n")
        f.write(f"image_shape   : {h} x {w}\n")
        f.write(f"roi_pixels    : {n_pixels}\n")
        f.write(
            f"mean_BGR      : {mean_bgr[0]:.6f}, {mean_bgr[1]:.6f}, {mean_bgr[2]:.6f}\n"
        )
        f.write(
            f"std_BGR       : {std_bgr[0]:.6f}, {std_bgr[1]:.6f}, {std_bgr[2]:.6f}\n"
        )

    # Sample up to max_sample_pixels pixels for CSV
    if n_pixels > max_sample_pixels:
        idx = np.random.choice(n_pixels, size=max_sample_pixels, replace=False)
        roi_sample = roi_pixels[idx]
    else:
        roi_sample = roi_pixels

    # Save sample as CSV (columns: B,G,R)
    csv_path = out_dir / f"{seq_id}_frame{frame_idx:05d}_roi_rgb_sample.csv"
    # Header: B,G,R
    header = "B,G,R"
    np.savetxt(csv_path, roi_sample, fmt="%d", delimiter=",", header=header, comments="")

    print("RGB statistics saved at:")
    print("  ", summary_path.resolve())
    print("RGB pixel sample saved at:")
    print("  ", csv_path.resolve())


def main():
    parser = argparse.ArgumentParser(
        description="Visualize which part of the UBFC frame is used as ROI and inspect RGB values."
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
        help="Output directory for debug PNGs and RGB stats.",
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

    # 2) Build ROI extractor
    roi_extractor = build_roi_extractor()

    # 3) Extract ROI (different behavior for mesh vs others)
    mask = None
    debug_shapes = []

    if isinstance(roi_extractor, MediaPipeFaceMeshRoi):
        roi_bgr, mask, debug_shapes = roi_extractor.extract_with_mask(frame_bgr)
    else:
        roi_bgr = roi_extractor.extract(frame_bgr)

    print(f"Original frame shape: {frame_bgr.shape}")
    print(f"ROI frame shape      : {roi_bgr.shape}")

    # 4) Build side-by-side image (original | ROI)
    debug_img = make_side_by_side_debug(frame_bgr, roi_bgr)

    # 5) Prepare output dir
    out_dir = out_root / seq_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 6) Save basic debug image
    png_path = out_dir / f"{seq_id}_frame{frame_idx:05d}_roi_debug.png"
    cv2.imwrite(str(png_path), debug_img)

    print("=======================================")
    print(" ROI DEBUG IMAGE SAVED AT:")
    print("   ", png_path.resolve())

    # 7) For mesh mode, also save mask and overlay
    if mask is not None:
        # mask visualization
        mask_vis = (mask * 255).astype(np.uint8)
        mask_vis = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)
        mask_path = out_dir / f"{seq_id}_frame{frame_idx:05d}_roi_mask.png"
        cv2.imwrite(str(mask_path), mask_vis)

        # overlay polygons on top of original frame
        overlay = frame_bgr.copy()
        for poly in debug_shapes:
            pts_int = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [pts_int], isClosed=True, color=(0, 255, 0), thickness=1)

        overlay_path = out_dir / f"{seq_id}_frame{frame_idx:05d}_roi_overlay.png"
        cv2.imwrite(str(overlay_path), overlay)

        print(" Additional mesh debug saved at:")
        print("   ", mask_path.resolve())
        print("   ", overlay_path.resolve())

    # 8) Compute and save RGB stats of the ROI
    compute_and_save_roi_rgb_stats(
        frame_bgr=frame_bgr,
        roi_bgr=roi_bgr,
        out_dir=out_dir,
        seq_id=seq_id,
        frame_idx=frame_idx,
        mask=mask,
        max_sample_pixels=1000,
    )

    print("=======================================\n")

    # 9) Optional interactive window
    try:
        cv2.imshow("ROI Debug (left = original, right = ROI)", debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        pass


if __name__ == "__main__":
    main()
