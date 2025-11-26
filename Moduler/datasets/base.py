# datasets/base.py
from typing import Protocol, List, Tuple
import numpy as np

class FrameSource(Protocol):
    """Generic interface for a time-indexed sequence of frames."""
    def get_frames_in_window(
        self,
        t_start: float,
        t_end: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            t_frame_s_win: (T,) seconds relative to some origin.
            frames: list/array of images (T, H, W, 3) or list of np.ndarray.
        """
        ...
