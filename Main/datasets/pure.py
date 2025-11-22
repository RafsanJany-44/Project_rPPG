# datasets/pure.py
from typing import Tuple, List
import numpy as np
import cv2
from pathlib import Path

from .base import FrameSource
from pure_image_io import load_image_files_and_timestamps

class PUREFrameSource(FrameSource):
    def __init__(self, seq_id: str):
        self.seq_id = seq_id
        self.image_files, self.t_img_ns = load_image_files_and_timestamps(seq_id)
        t0_ns = self.t_img_ns[0]
        self.t_frame_s = (self.t_img_ns - t0_ns) * 1e-9  # seconds

    def get_frames_in_window(
        self,
        t_start: float,
        t_end: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        mask = (self.t_frame_s >= t_start) & (self.t_frame_s <= t_end)
        t_frame_s_win = self.t_frame_s[mask]
        selected_files = [f for f, m in zip(self.image_files, mask) if m]

        if len(selected_files) == 0:
            raise RuntimeError(
                f"No frames found for seq={self.seq_id} in [{t_start}, {t_end}] s"
            )

        frames = []
        for f in selected_files:
            img_bgr = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise RuntimeError(f"Failed to load image: {f}")
            frames.append(img_bgr)

        return t_frame_s_win, np.stack(frames, axis=0)
