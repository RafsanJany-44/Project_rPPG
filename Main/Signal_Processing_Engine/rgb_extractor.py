# signals/rgb_extractor.py
from typing import Tuple
import numpy as np
import cv2

from Main.datasets.base import FrameSource
from Main.ROI_Engine.base import RoiExtractor

def extract_rgb_timeseries(
    source: FrameSource,
    roi_extractor: RoiExtractor,
    t_start: float,
    t_end: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dataset-agnostic RGB time series extraction.

    Parameters
    ----------
    source : FrameSource
        Any dataset-specific frame source (PURE, another dataset, video, etc.).
    roi_extractor : RoiExtractor
        Strategy for selecting pixels (central, face, MediaPipe regions...).
    t_start, t_end : float
        Time window in seconds (relative to source's origin).

    Returns
    -------
    t_frame_s_win : (T,)
    rgb_ts : (T, 3) float32, RGB order.
    """
    t_frame_s_win, frames_bgr = source.get_frames_in_window(t_start, t_end)

    rgb_list = []
    for img_bgr in frames_bgr:
        roi_bgr = roi_extractor.extract(img_bgr)
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        mean_rgb = roi_rgb.reshape(-1, 3).mean(axis=0)
        rgb_list.append(mean_rgb)

    rgb_ts = np.vstack(rgb_list).astype(np.float32)
    return t_frame_s_win, rgb_ts
