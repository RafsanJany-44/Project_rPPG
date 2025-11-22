# roi/central.py
import numpy as np
from .base import RoiExtractor

class CentralRoiExtractor(RoiExtractor):
    def __init__(self, frac: float = 0.5):
        assert 0 < frac <= 1.0
        self.frac = frac

    def extract(self, img_bgr: np.ndarray) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        roi_h = int(h * self.frac)
        roi_w = int(w * self.frac)

        y0 = (h - roi_h) // 2
        x0 = (w - roi_w) // 2
        y1 = y0 + roi_h
        x1 = x0 + roi_w

        return img_bgr[y0:y1, x0:x1, :]
