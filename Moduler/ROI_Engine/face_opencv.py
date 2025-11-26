# roi/face_opencv.py
import numpy as np
import cv2
from .base import RoiExtractor

class OpenCVFaceBoxRoi(RoiExtractor):
    def __init__(self, face_cascade_path: str, scale_factor=1.1, min_neighbors=5):
        self.cascade = cv2.CascadeClassifier(face_cascade_path)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def extract(self, img_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
        )
        if len(faces) == 0:
            # Fallback: central ROI or whole frame
            return img_bgr

        x, y, w, h = faces[0]  # for now, first face
        return img_bgr[y:y+h, x:x+w, :]
