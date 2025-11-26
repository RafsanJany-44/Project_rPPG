# roi_face_mediapipe.py
"""
Face ROI using MediaPipe Face Mesh.

We approximate a combined region that covers forehead + cheeks by:
- Detecting face mesh,
- Computing face bounding box,
- Selecting an upper-to-mid portion of that box as the ROI.

You can later refine this to be more anatomically precise.
"""

from typing import Tuple

import numpy as np

from roi_base import RoiExtractor  # adjust import if using package

try:
    import mediapipe as mp
except ImportError as e:
    mp = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


class MediaPipeFaceRegionsRoi(RoiExtractor):
    """
    ROI based on MediaPipe face mesh.

    Currently:
      - detects face mesh,
      - builds bounding box of all landmarks,
      - returns a sub-rectangle covering approx. forehead + cheeks.

    This is primarily to give you the architecture; you can later
    replace the bbox heuristic with precise region masks.
    """

    def __init__(
        self,
        refine_landmarks: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "mediapipe is required for MediaPipeFaceRegionsRoi but "
                "could not be imported. Install it with `pip install mediapipe`."
            ) from _IMPORT_ERROR

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,  # we process frame-by-frame here
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def _get_face_bbox_from_landmarks(
        self, landmarks, img_shape: Tuple[int, int, int]
    ) -> Tuple[int, int, int, int]:
        h, w, _ = img_shape
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]

        x_min = max(0, int(np.floor(min(xs))))
        x_max = min(w, int(np.ceil(max(xs))))
        y_min = max(0, int(np.floor(min(ys))))
        y_max = min(h, int(np.ceil(max(ys))))

        return x_min, y_min, x_max, y_max

    def extract(self, img_bgr: np.ndarray) -> np.ndarray:
        h, w = img_bgr.shape[:2]

        # MediaPipe expects RGB
        img_rgb = img_bgr[:, :, ::-1]

        results = self.face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            # Fallback: original image
            return img_bgr

        # For now, use the first face
        face_landmarks = results.multi_face_landmarks[0].landmark

        x_min, y_min, x_max, y_max = self._get_face_bbox_from_landmarks(
            face_landmarks, img_bgr.shape
        )

        # Heuristic: define ROI as upper-to-mid portion (forehead + cheeks)
        box_width = x_max - x_min
        box_height = y_max - y_min

        # y: from slightly above the center of the eyes to mid-lower face
        roi_y0 = int(y_min + 0.15 * box_height)   # below hairline / top
        roi_y1 = int(y_min + 0.70 * box_height)   # around cheeks

        # x: take most of the width (cheek-to-cheek)
        roi_x0 = int(x_min + 0.10 * box_width)
        roi_x1 = int(x_min + 0.90 * box_width)

        roi_y0 = max(0, min(h, roi_y0))
        roi_y1 = max(0, min(h, roi_y1))
        roi_x0 = max(0, min(w, roi_x0))
        roi_x1 = max(0, min(w, roi_x1))

        if roi_y1 <= roi_y0 or roi_x1 <= roi_x0:
            # Fallback if something goes wrong
            return img_bgr[y_min:y_max, x_min:x_max, :]

        return img_bgr[roi_y0:roi_y1, roi_x0:roi_x1, :]
