# roi_face_mediapipe.py
"""
Face ROI using MediaPipe Face Mesh.

We build a more precise region based on facial landmarks:
- Forehead region (around landmark 10 and upper face)
- Left cheek
- Right cheek

The user can choose which subregions to include via flags.
"""

from typing import Tuple, List
import numpy as np

from roi_base import RoiExtractor  # adjust import path if needed

try:
    import mediapipe as mp
except ImportError as e:
    mp = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


class MediaPipeFaceRegionsRoi(RoiExtractor):
    """
    ROI based on MediaPipe FaceMesh.

    Parameters
    ----------
    use_forehead : bool
        If True, include a forehead ROI.
    use_left_cheek : bool
        If True, include a left-cheek ROI.
    use_right_cheek : bool
        If True, include a right-cheek ROI.
    refine_landmarks, min_detection_confidence, min_tracking_confidence :
        Standard MediaPipe FaceMesh options.

    Behavior
    --------
    - Detect face mesh on each frame.
    - From the 468 landmarks, select subsets for forehead / cheeks.
    - Compute a bounding box for each selected subset.
    - Take the union (min x, min y, max x, max y) of all selected boxes.
    - Crop that rectangle and return it as the ROI image.

    If detection fails, fall back to returning the full image.
    """

    def __init__(
        self,
        use_forehead: bool = True,
        use_left_cheek: bool = True,
        use_right_cheek: bool = True,
        refine_landmarks: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "mediapipe is required for MediaPipeFaceRegionsRoi but "
                "could not be imported. Install it with `pip install mediapipe`."
            ) from _IMPORT_ERROR

        self.use_forehead = use_forehead
        self.use_left_cheek = use_left_cheek
        self.use_right_cheek = use_right_cheek

        self.mp_face_mesh = mp.solutions.face_mesh
        # static_image_mode=True → treat each frame independently
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # Pre-defined landmark index sets
        # These come from the MediaPipe Face Mesh topology.
        # They are not perfect anatomically, but are reasonable approximations.
        # You can tweak / add indices as needed.
        self.forehead_indices = [
            10,   # mid-upper forehead
            338,  # right-upper forehead
            297,  # right-upper
            332,  # right-upper
            284,  # right-upper
            109,  # left-upper forehead
            67,   # left-upper
            103,  # left-upper
            54,   # left-upper
        ]

        # Left cheek (viewer’s left; subject’s right)
        self.left_cheek_indices = [
            234, 93, 132, 58, 172, 136, 150, 176
        ]

        # Right cheek (viewer’s right; subject’s left)
        self.right_cheek_indices = [
            454, 323, 361, 288, 397, 365, 379, 400
        ]

    # -----------------------
    # Internal helpers
    # -----------------------
    def _landmarks_to_xy(
        self, landmarks, img_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """Convert all 468 landmarks to (x, y) pixel coordinates."""
        h, w, _ = img_shape
        xs = []
        ys = []
        for lm in landmarks:
            xs.append(lm.x * w)
            ys.append(lm.y * h)
        xs = np.array(xs, dtype=np.float32)
        ys = np.array(ys, dtype=np.float32)
        pts = np.stack([xs, ys], axis=1)  # shape (468, 2)
        return pts

    def _bbox_from_indices(
        self, pts: np.ndarray, indices: List[int]
    ) -> Tuple[int, int, int, int]:
        """Get bounding box (x_min, y_min, x_max, y_max) for a subset of points."""
        if len(indices) == 0:
            return None

        sub = pts[indices, :]  # (N, 2)
        x_min = int(np.floor(np.min(sub[:, 0])))
        x_max = int(np.ceil(np.max(sub[:, 0])))
        y_min = int(np.floor(np.min(sub[:, 1])))
        y_max = int(np.ceil(np.max(sub[:, 1])))

        return x_min, y_min, x_max, y_max

    def _union_bboxes(
        self, boxes: List[Tuple[int, int, int, int]]
    ) -> Tuple[int, int, int, int] | None:
        """Take union of multiple bounding boxes."""
        valid = [b for b in boxes if b is not None]
        if not valid:
            return None
        xs_min = [b[0] for b in valid]
        ys_min = [b[1] for b in valid]
        xs_max = [b[2] for b in valid]
        ys_max = [b[3] for b in valid]
        return min(xs_min), min(ys_min), max(xs_max), max(ys_max)

    # -----------------------
    # Main API
    # -----------------------
    def extract(self, img_bgr: np.ndarray) -> np.ndarray:
        h, w = img_bgr.shape[:2]

        # MediaPipe expects RGB
        img_rgb = img_bgr[:, :, ::-1]

        results = self.face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            # Fallback: if no face detected, just return full image
            return img_bgr

        face_landmarks = results.multi_face_landmarks[0].landmark
        pts = self._landmarks_to_xy(face_landmarks, img_bgr.shape)

        boxes = []

        if self.use_forehead:
            boxes.append(self._bbox_from_indices(pts, self.forehead_indices))

        if self.use_left_cheek:
            boxes.append(self._bbox_from_indices(pts, self.left_cheek_indices))

        if self.use_right_cheek:
            boxes.append(self._bbox_from_indices(pts, self.right_cheek_indices))

        # If user disabled everything, fallback to full face bounding box
        if not boxes:
            x_min = int(np.floor(np.min(pts[:, 0])))
            x_max = int(np.ceil(np.max(pts[:, 0])))
            y_min = int(np.floor(np.min(pts[:, 1])))
            y_max = int(np.ceil(np.max(pts[:, 1])))
            boxes = [(x_min, y_min, x_max, y_max)]

        # Combine selected regions into one ROI
        union_box = self._union_bboxes(boxes)
        if union_box is None:
            return img_bgr  # fallback

        x_min, y_min, x_max, y_max = union_box

        # Clamp to image boundaries
        x_min = max(0, min(w, x_min))
        x_max = max(0, min(w, x_max))
        y_min = max(0, min(h, y_min))
        y_max = max(0, min(h, y_max))

        if x_max <= x_min or y_max <= y_min:
            # Something degenerate – fallback to full image
            return img_bgr

        roi = img_bgr[y_min:y_max, x_min:x_max, :]
        return roi
