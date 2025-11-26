# dataset_base.py
"""
Dataset abstraction for time-indexed image sequences.

Defines a FrameSource protocol that any dataset implementation
(PURE, other datasets, video files, webcams, etc.) can follow.
"""

from typing import Protocol, Tuple
import numpy as np


class FrameSource(Protocol):
    """
    Generic interface for a time-indexed sequence of frames.

    Implementations must provide frames and their timestamps,
    and be able to return a time-windowed subset.
    """

    def get_frames_in_window(
        self,
        t_start: float,
        t_end: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return frames and timestamps within a time window.

        Parameters
        ----------
        t_start : float
            Start time in seconds (inclusive), relative to some origin.
        t_end : float
            End time in seconds (inclusive), relative to the same origin.

        Returns
        -------
        t_frame_s_win : np.ndarray
            Timestamps in seconds, shape (T,).
        frames_bgr : np.ndarray
            Frames in BGR format, shape (T, H, W, 3), dtype uint8.
        """
        ...
