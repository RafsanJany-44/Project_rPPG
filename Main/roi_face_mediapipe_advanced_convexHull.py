# roi_face_mediapipe_advanced.py
"""
Advanced face ROI using MediaPipe Face Mesh.

We now build a clean full-face skin mask using convex hulls:

- Outer hull: face oval.
- Inner hulls (holes): left eye, right eye, lips, nose.

Everything inside the outer hull but outside the inner hulls
is kept as ROI (forehead + cheeks + jaw, etc.).
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import cv2

from roi_base import RoiExtractor  # adjust import path if needed

try:
    import mediapipe as mp
except ImportError as e:
    mp = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


# -------------------------------------------------------------------
# 1. (Optional) ROI SHAPE CONFIGURATION
#    Kept for future extension but NOT used in the new hull-based mask.
# -------------------------------------------------------------------

ROI_SHAPES: List[Dict[str, Any]] = [
    {
        "name": "forehead_strip",
        "type": "polygon",
        "indices": [
            2, 3, 4, 5, 6, 8, 9, 10, 18, 21, 32, 35, 36, 43, 46, 47, 48, 50,
            54, 58, 67, 68, 69, 71, 92, 93, 101, 103, 104, 108, 109, 116, 117,
            118, 123, 132, 134, 135, 138, 139, 142, 148, 149, 150, 151, 152,
            182, 187, 188, 193, 197, 201, 205, 206, 207, 210, 211, 212, 216,
            234, 248, 251, 262, 265, 266, 273, 277, 278, 280, 284, 288, 297,
            299, 322, 323, 330, 332, 333, 337, 338, 345, 346, 361, 363, 364,
            367, 368, 371, 377, 379, 411, 412, 417, 421, 425, 426, 427, 430,
            432, 436,
        ],
    },
]


class MediaPipeFaceMeshRoi(RoiExtractor):
    """
    Advanced ROI based on MediaPipe FaceMesh.

    New behavior:
      - Build a full-face skin mask using convex hulls.
      - Remove eyes, mouth, and nose regions.

    Public API:
      - extract(img_bgr) -> roi_img
      - extract_with_mask(img_bgr) -> (roi_img, mask, debug_shapes)
    """

    def __init__(
        self,
        roi_shapes: List[Dict[str, Any]] = ROI_SHAPES,
        refine_landmarks: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "mediapipe is required for MediaPipeFaceMeshRoi but "
                "could not be imported. Install it with `pip install mediapipe`."
            ) from _IMPORT_ERROR

        # roi_shapes kept only for future extension / debugging
        self.roi_shapes = roi_shapes

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    # --------------------------------------------------------------
    # Internal helpers: landmarks
    # --------------------------------------------------------------
    def _landmarks_to_xy(
        self,
        landmarks,
        img_shape: Tuple[int, int, int],
    ) -> np.ndarray:
        """
        Convert 3D normalized landmarks to 2D pixel coordinates.

        Returns
        -------
        pts : np.ndarray of shape (468, 2)
        """
        h, w, _ = img_shape
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        pts = np.stack([np.array(xs, dtype=np.float32),
                        np.array(ys, dtype=np.float32)], axis=1)
        return pts

    # --------------------------------------------------------------
    # New core: full-face skin mask using convex hulls
    # --------------------------------------------------------------
    def _build_full_face_skin_mask(
        self,
        img_bgr: np.ndarray,
        pts: np.ndarray,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Build a mask that keeps the whole facial skin but removes
        eyes, mouth, and nose.

        Returns
        -------
        mask : (H, W) uint8 with {0,1}
        debug_shapes : list of hull polygons (for visualization)
        """
        h, w = img_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        debug_shapes: List[np.ndarray] = []

        mp_fm = self.mp_face_mesh

        # ---------- 1) Outer face hull (FACE_OVAL) ----------
        face_oval_pairs = getattr(mp_fm, "FACEMESH_FACE_OVAL")
        face_oval_idx = np.unique(np.array(list(face_oval_pairs)).flatten())
        face_oval_pts = pts[face_oval_idx]
        face_hull = cv2.convexHull(face_oval_pts.astype(np.int32))

        cv2.fillConvexPoly(mask, face_hull, 1)
        debug_shapes.append(face_hull.reshape(-1, 2).astype(np.float32))

        # ---------- 2) Helper to build inner hulls ----------
        # ---------- 2) Helper to build inner hulls ----------
        def hull_from_group(group) -> np.ndarray:
            """Build convex hull from a MediaPipe FACEMESH_* group.

            Safely ignores landmark indices that are out of range for pts
            (e.g., iris indices when refine_landmarks=False).
            """
            if not group:
                return None

            idx = np.unique(np.array(list(group)).flatten())

            
            idx = idx[(idx >= 0) & (idx < pts.shape[0])]

            if idx.size < 3:
                return None

            g_pts = pts[idx]
            return cv2.convexHull(g_pts.astype(np.int32))


        # Eyes
        left_eye_group = getattr(mp_fm, "FACEMESH_LEFT_EYE")
        right_eye_group = getattr(mp_fm, "FACEMESH_RIGHT_EYE")

        left_eye_hull = hull_from_group(left_eye_group)
        right_eye_hull = hull_from_group(right_eye_group)

        # Iris (if present in this MediaPipe version)
        left_iris_group = getattr(mp_fm, "FACEMESH_LEFT_IRIS", ())
        right_iris_group = getattr(mp_fm, "FACEMESH_RIGHT_IRIS", ())
        left_iris_hull = hull_from_group(left_iris_group)
        right_iris_hull = hull_from_group(right_iris_group)

        # Lips / mouth
        lips_group = getattr(mp_fm, "FACEMESH_LIPS")
        lips_hull = hull_from_group(lips_group)

        # Nose: MediaPipe has no ready-made group, so we pick
        # a set of well-known nose landmarks and hull them.
        nose_idx = np.array(
             #[1, 2, 4, 5, 6, 168, 98, 327, 197, 195],
             [],
            dtype=np.int32,
        )
        nose_idx = nose_idx[(nose_idx >= 0) & (nose_idx < pts.shape[0])]
        nose_hull = None
        if nose_idx.size >= 3:
            nose_hull = cv2.convexHull(pts[nose_idx].astype(np.int32))

        inner_hulls = [
            left_eye_hull,
            right_eye_hull,
            left_iris_hull,
            right_iris_hull,
            lips_hull,
            nose_hull,
        ]

        # ---------- 3) Carve inner holes ----------
        for hull in inner_hulls:
            if hull is None or hull.shape[0] < 3:
                continue
            cv2.fillConvexPoly(mask, hull, 0)
            debug_shapes.append(hull.reshape(-1, 2).astype(np.float32))

        return mask, debug_shapes

    # --------------------------------------------------------------
    # Fallback: full bounding box if mask empty
    # --------------------------------------------------------------
    def _fallback_box_mask(
        self,
        img_bgr: np.ndarray,
        pts: np.ndarray,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        h, w = img_bgr.shape[:2]
        x_min = int(np.floor(np.min(pts[:, 0])))
        x_max = int(np.ceil(np.max(pts[:, 0])))
        y_min = int(np.floor(np.min(pts[:, 1])))
        y_max = int(np.ceil(np.max(pts[:, 1])))

        x_min = max(0, min(w, x_min))
        x_max = max(0, min(w, x_max))
        y_min = max(0, min(h, y_min))
        y_max = max(0, min(h, y_max))

        mask = np.zeros((h, w), dtype=np.uint8)
        debug_shapes: List[np.ndarray] = []

        if x_max > x_min and y_max > y_min:
            mask[y_min:y_max, x_min:x_max] = 1
            rect_poly = np.array(
                [
                    [x_min, y_min],
                    [x_max, y_min],
                    [x_max, y_max],
                    [x_min, y_max],
                ],
                dtype=np.float32,
            )
            debug_shapes.append(rect_poly)

        return mask, debug_shapes

    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------
    def extract_with_mask(
        self,
        img_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Compute the ROI for one frame and also return the mask and shapes.

        Returns
        -------
        roi_img : (H, W, 3) uint8   - masked image
        mask    : (H, W)   uint8    - 0 outside ROI, 1 inside
        debug_shapes : list of np.ndarray with polygon points (x, y)
        """
        h, w = img_bgr.shape[:2]

        img_rgb = img_bgr[:, :, ::-1]
        results = self.face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            # If detection fails, just return whole image.
            mask = np.ones((h, w), dtype=np.uint8)
            roi_img = img_bgr.copy()
            return roi_img, mask, []

        face_landmarks = results.multi_face_landmarks[0].landmark
        pts = self._landmarks_to_xy(face_landmarks, img_bgr.shape)

        mask, debug_shapes = self._build_full_face_skin_mask(img_bgr, pts)

        # Fallback if for some reason mask is empty
        if np.count_nonzero(mask) == 0:
            mask, box_shapes = self._fallback_box_mask(img_bgr, pts)
            debug_shapes.extend(box_shapes)

        roi_img = img_bgr.copy()
        roi_img[mask == 0] = 0

        return roi_img, mask, debug_shapes

    def extract(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Standard RoiExtractor interface: only return the masked image.
        """
        roi_img, _, _ = self.extract_with_mask(img_bgr)
        return roi_img
