# roi/face_mediapipe.py
import numpy as np
from .base import RoiExtractor

class MediaPipeFaceRegionsRoi(RoiExtractor):
    def __init__(self, regions=("forehead", "left_cheek", "right_cheek")):
        # init mediapipe face mesh etc.
        self.regions = regions
        # self.mp_face_mesh = ...

    def extract(self, img_bgr: np.ndarray) -> np.ndarray:
        # 1) run Mediapipe
        # 2) get landmarks
        # 3) build masks/polygons for requested regions
        # 4) either crop bounding box around combined regions
        #    or return a masked image
        #
        # For now, think of this as:
        # roi_bgr = some_function_of(img_bgr, self.regions)
        # return roi_bgr
        raise NotImplementedError("Implement MediaPipe ROI extraction here.")
