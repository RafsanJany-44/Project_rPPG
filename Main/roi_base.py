# roi_base.py
"""
ROI extractor abstraction.

Any ROI strategy (central crop, face box, MediaPipe regions, etc.)
should implement RoiExtractor.
"""

from typing import Protocol
import numpy as np


class RoiExtractor(Protocol):
    """
    Strategy interface for extracting a ROI from a single BGR frame.
    """

    def extract(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Extract ROI from a BGR frame.

        Parameters
        ----------
        img_bgr : np.ndarray
            Image of shape (H, W, 3), BGR, uint8.

        Returns
        -------
        roi_bgr : np.ndarray
            ROI image of shape (h_roi, w_roi, 3), BGR, uint8.
        """
        ...
