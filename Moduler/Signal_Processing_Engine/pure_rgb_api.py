# pure_rgb_api.py
"""
PURE-specific convenience API for RGB extraction.

These functions mirror the old pure_image_signal.extract_rgb_timeseries
signature, but are built on top of the modular, SOLID-compliant design.
"""

from typing import Tuple
import numpy as np

from pure_dataset import PUREFrameSource
from roi_central import CentralRoiExtractor
from roi_face_opencv import OpenCVFaceBoxRoi
from roi_face_mediapipe import MediaPipeFaceRegionsRoi
from rgb_extractor import extract_rgb_timeseries as _extract_rgb


def extract_pure_rgb_central(
    seq_id: str,
    t_start: float,
    t_end: float,
    roi_frac: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    PURE RGB extraction using a central rectangular ROI (old behavior).
    """
    source = PUREFrameSource(seq_id)
    roi = CentralRoiExtractor(frac=roi_frac)
    return _extract_rgb(source, roi, t_start, t_end)


def extract_pure_rgb_face_opencv(
    seq_id: str,
    t_start: float,
    t_end: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    PURE RGB extraction using OpenCV face-box ROI.
    """
    source = PUREFrameSource(seq_id)
    roi = OpenCVFaceBoxRoi()
    return _extract_rgb(source, roi, t_start, t_end)


def extract_pure_rgb_face_mediapipe(
    seq_id: str,
    t_start: float,
    t_end: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    PURE RGB extraction using MediaPipe-based face regions (forehead + cheeks approx).
    """
    source = PUREFrameSource(seq_id)
    roi = MediaPipeFaceRegionsRoi()
    return _extract_rgb(source, roi, t_start, t_end)
