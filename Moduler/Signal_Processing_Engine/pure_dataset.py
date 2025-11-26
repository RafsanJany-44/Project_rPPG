# pure_dataset.py
"""
PURE dataset implementation of FrameSource.

This is the only file that knows about:
- PURE sequence IDs
- How to find image files and timestamps for PURE
"""

from typing import Tuple
import numpy as np
import cv2

from .dataset_base import FrameSource  # adjust import if using package
from Main.Data_Read_Engine import load_pure_image_files_and_timestamps


class PUREFrameSource(FrameSource):
    """
    FrameSource implementation for the PURE dataset.

    Assumes pure_image_io.load_image_files_and_timestamps(seq_id) returns:
        image_files: list[pathlib.Path]
        t_img_ns:    np.ndarray of shape (N,), ns timestamps
    """

    def __init__(self, seq_id: str):
        self.seq_id = seq_id

        image_files, t_img_ns = load_pure_image_files_and_timestamps(seq_id)
        if len(image_files) == 0:
            raise RuntimeError(f"No images found for PURE sequence '{seq_id}'")

        self.image_files = image_files
        self.t_img_ns = t_img_ns

        # Use first frame as time origin (seconds)
        t0_ns = self.t_img_ns[0]
        self.t_frame_s = (self.t_img_ns - t0_ns) * 1e-9

    def get_frames_in_window(
        self,
        t_start: float,
        t_end: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return frames (BGR) in the given time window [t_start, t_end].
        """

        mask = (self.t_frame_s >= t_start) & (self.t_frame_s <= t_end)
        t_frame_s_win = self.t_frame_s[mask]
        selected_files = [f for f, m in zip(self.image_files, mask) if m]

        if len(selected_files) == 0:
            raise RuntimeError(
                f"No frames found for PURE seq={self.seq_id} "
                f"in window [{t_start}, {t_end}] s"
            )

        frames = []
        for f in selected_files:
            img_bgr = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise RuntimeError(f"Failed to load image: {f}")
            frames.append(img_bgr)

        frames_bgr = np.stack(frames, axis=0)
        return t_frame_s_win, frames_bgr

