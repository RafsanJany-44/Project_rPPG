# pure_image_io.py
"""
Utilities for handling PURE image frames and extracting timestamps
from filenames of the form: 'Image<TIMESTAMP>.png'.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np

from .config import PURE_ROOT


def get_image_dir(seq_id: str) -> Path:
    """
    Given a sequence ID like '01-01',
    return the directory path that contains:
      - 01-01.avi
      - Image*.png frames

    This function assumes the directory structure:
      PURE_IMAGE_ROOT / seq_id / (files...)
    """
    img_dir = PURE_ROOT / seq_id
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    return img_dir


def load_pure_image_files_and_timestamps(seq_id: str) -> Tuple[List[Path], np.ndarray]:
    """
    For a given sequence, list all PNG frames whose names follow
    the pattern 'Image<TIMESTAMP>.png', and extract the timestamps.

    Returns
    -------
    image_files : List[Path]
        List of image file paths, sorted by their parsed timestamp.
    t_img_ns : np.ndarray
        Array of timestamps (int64, nanoseconds) parsed from filenames,
        sorted in the same order as image_files.
    """
    img_dir = get_image_dir(seq_id)

    # Collect all files starting with 'Image' and ending with '.png'
    image_files = sorted(img_dir.glob("Image*.png"))

    if len(image_files) == 0:
        raise RuntimeError(f"No Image*.png files found in {img_dir}")

    # Parse timestamps from filenames:
    # 'Image1392643993642815000.png' -> 1392643993642815000 (int)
    t_list = []
    for f in image_files:
        stem = f.stem  # , 'Image1392643993642815000'
        if not stem.startswith("Image"):
            raise ValueError(f"Unexpected image filename format: {f.name}")
        ts_str = stem[len("Image"):]  # '1392643993642815000'
        try:
            ts_int = int(ts_str)
        except ValueError as e:
            raise ValueError(f"Cannot parse timestamp from: {f.name}") from e
        t_list.append(ts_int)

    t_img_ns = np.array(t_list, dtype=np.int64)

    return image_files, t_img_ns
