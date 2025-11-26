# roi_face_opencv.py
"""
Face ROI using OpenCV's CascadeClassifier (rectangular face box).

Good as a baseline face-based ROI strategy.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import cv2

from roi_base import RoiExtractor  # adjust import if using package


class OpenCVFaceBoxRoi(RoiExtractor):
    """
    Detects a face bounding box using a Haar cascade and returns that ROI.

    If no face is detected, falls back to the full image.
    """

    def __init__(
        self,
        cascade_path: Optional[str] = None,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
    ):
        """
        Parameters
        ----------
        cascade_path : str, optional
            Path to Haar cascade XML file. If None, tries to use
            OpenCV's default frontal face cascade.
        scale_factor : float
            How much the image size is reduced at each image scale.
        min_neighbors : int
            How many neighbors each candidate rectangle should have to retain it.
        """

        if cascade_path is None:
            default_path = (
                Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            )
            cascade_path = str(default_path)

        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")

        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def extract(self, img_bgr: np.ndarray) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
        )

        if len(faces) == 0:
            # Fallback: return full frame (you can also fallback to Central ROI)
            return img_bgr

        # Pick the largest detected face
        x, y, fw, fh = max(faces, key=lambda rect: rect[2] * rect[3])

        # Optional padding can be added here
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(w, x + fw)
        y1 = min(h, y + fh)

        return img_bgr[y0:y1, x0:x1, :]
