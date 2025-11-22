# Main/Signal_Processing_Engine/ubfc_dataset.py
"""
UBFC dataset implementation of FrameSource.

This knows how to:
- Open a UBFC .avi video,
- Read all frames,
- Build frame timestamps from FPS,
- Return frames + timestamps inside a time window.

Designed to match the FrameSource protocol used by rgb_extractor.
"""

from typing import Tuple, Union
from pathlib import Path

import numpy as np
import cv2

from dataset_base import FrameSource


class UBFCFrameSource(FrameSource):
    """
    FrameSource implementation for the UBFC dataset.

    Assumes videos are standard .avi files with valid FPS.
    Time origin is the first frame at t = 0.0 s.
    """

    def __init__(self, video_path: Union[str, Path]):
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"UBFC video not found: {video_path}")

        self.video_path = video_path

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open UBFC video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            cap.release()
            raise RuntimeError(f"Invalid FPS reported for UBFC video: {fps}")

        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise RuntimeError(f"No frames read from UBFC video: {video_path}")

        self.fps = fps
        self.frames_bgr = np.stack(frames, axis=0)  # (T, H, W, 3), BGR, uint8
        n_frames = self.frames_bgr.shape[0]

        # Time origin = first frame, timestamps in seconds
        self.t_frame_s = np.arange(n_frames, dtype=np.float64) / fps

    def get_frames_in_window(
        self,
        t_start: float,
        t_end: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return frames (BGR) in the given time window [t_start, t_end].

        Parameters
        ----------
        t_start, t_end : float
            Window in seconds, relative to first frame (0.0).

        Returns
        -------
        t_frame_s_win : np.ndarray
            Timestamps in seconds, shape (T,).
        frames_bgr : np.ndarray
            Frames in BGR format, shape (T, H, W, 3), uint8.
        """
        t = self.t_frame_s
        mask = (t >= t_start) & (t <= t_end)
        t_frame_s_win = t[mask]
        if t_frame_s_win.size == 0:
            raise RuntimeError(
                f"No frames in window [{t_start}, {t_end}] s "
                f"for UBFC video {self.video_path.name}"
            )

        frames_bgr_win = self.frames_bgr[mask]
        return t_frame_s_win, frames_bgr_win
