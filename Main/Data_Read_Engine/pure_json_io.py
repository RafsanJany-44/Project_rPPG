# pure_json_io.py
import json
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np

from .config import PURE_ROOT


def get_pure_json_path(seq_id: str) -> Path:
    """
    Given a sequence ID like '01-01',
    return the path to the corresponding JSON file.
    """
    json_name = f"{seq_id}.json"
    json_path = PURE_ROOT / json_name
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    return json_path


def load_pure_json(seq_id: str) -> Dict[str, Any]:
    """
    Load the PURE JSON file for a given sequence ID.
    Return the parsed JSON as a Python dict.
    """
    json_path = get_pure_json_path(seq_id)
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def extract_streams_from_pure_json(
    data: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    From a loaded PURE JSON dict, extract:
      - t_ppg_ns : np.ndarray of int64 (PPG timestamps in ns)
      - wave     : np.ndarray of float32 (PPG waveform values)
      - t_vid_ns : np.ndarray of int64 (video frame timestamps in ns)
      - hr_dev   : np.ndarray of float32 (device pulseRate in BPM)

    This function only uses /FullPackage and /Image entries.
    """
    full_pkg = data["/FullPackage"]
    img_pkgs = data["/Image"]

    # PPG stream
    t_ppg_ns = np.array([p["Timestamp"] for p in full_pkg], dtype=np.int64)
    wave = np.array(
        [p["Value"]["waveform"] for p in full_pkg],
        dtype=np.float32,
    )
    hr_dev = np.array(
        [p["Value"]["pulseRate"] for p in full_pkg],
        dtype=np.float32,
    )

    # Image (video) stream
    t_vid_ns = np.array([p["Timestamp"] for p in img_pkgs], dtype=np.int64)

    return t_ppg_ns, wave, t_vid_ns, hr_dev
