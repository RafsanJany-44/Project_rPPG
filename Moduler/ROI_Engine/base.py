# roi/base.py
from typing import Protocol
import numpy as np

class RoiExtractor(Protocol):
    def extract(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Extract ROI from a single BGR frame.

        Returns:
            roi_bgr: np.ndarray of shape (h_roi, w_roi, 3)
        """
        ...
