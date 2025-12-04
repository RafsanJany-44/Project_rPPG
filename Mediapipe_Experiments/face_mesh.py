# Copyright 2020 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone Face Mesh using local SolutionBase and resources."""

from typing import NamedTuple

import numpy as np

from solution_base import SolutionBase
from face_mesh_connections import (
    FACEMESH_CONTOURS,
    FACEMESH_FACE_OVAL,
    FACEMESH_IRISES,
    FACEMESH_LEFT_EYE,
    FACEMESH_LEFT_EYEBROW,
    FACEMESH_LEFT_IRIS,
    FACEMESH_LIPS,
    FACEMESH_NOSE,
    FACEMESH_RIGHT_EYE,
    FACEMESH_RIGHT_EYEBROW,
    FACEMESH_RIGHT_IRIS,
    FACEMESH_TESSELATION,
)

FACEMESH_NUM_LANDMARKS = 468
FACEMESH_NUM_LANDMARKS_WITH_IRISES = 478

# In this minimal setup, we expect the graph file to be in the same folder.
_BINARYPB_FILE_PATH = "face_landmark_front_cpu.binarypb"


class FaceMesh(SolutionBase):
  """Face Mesh wrapper based on local SolutionBase and graph.

  This class processes an RGB image and returns face landmarks on each
  detected face, using the copied MediaPipe graph and TFLite models in
  the same folder (no 'pip install mediapipe' required).
  """

  def __init__(
      self,
      static_image_mode: bool = False,
      max_num_faces: int = 1,
      refine_landmarks: bool = False,
      min_detection_confidence: float = 0.5,
      min_tracking_confidence: float = 0.5,
  ):
    """Initializes a FaceMesh object (minimal configuration).

    Note:
      In this standalone version, min_detection_confidence and
      min_tracking_confidence are not forwarded as calculator_params.
      The thresholds defined inside the graph are used as-is.
    """
    super().__init__(
        binary_graph_path=_BINARYPB_FILE_PATH,
        side_inputs={
            "num_faces": max_num_faces,
            "with_attention": refine_landmarks,
            "use_prev_landmarks": not static_image_mode,
        },
        # calculator_params are not supported in our minimal SolutionBase.
        calculator_params=None,
        outputs=["multi_face_landmarks"],
    )

  def process(self, image: np.ndarray) -> NamedTuple:
    """Processes an RGB image and returns face landmarks.

    Args:
      image: An RGB image (H, W, 3) as a numpy ndarray.

    Returns:
      A NamedTuple with a "multi_face_landmarks" field containing
      the landmarks for each detected face (or None if none).
    """
    return super().process(input_data={"image": image})
