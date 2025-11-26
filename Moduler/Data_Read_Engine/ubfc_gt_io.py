# ubfc_gt_io.py


import numpy as np
from pathlib import Path


def load_ubfc_ground_truth(path):
    """
    Robust loader for UBFC ground_truth_X.txt.
    Handles both well-formatted (3 paragraphs) and compact (1-block) files.
    Returns timestamps (s) and PPG signal.
    """
    with open(path, "r") as f:
        content = f.read().strip()

    # Split by any whitespace
    numbers = np.array(content.split(), dtype=float)
    n = len(numbers)

    # UBFC files usually contain roughly equal-sized segments for:
    # [PPG | HR | timestamps]
    # We can use the value magnitude to detect transitions:
    # - PPG values are normalized (~0–1)
    # - HR values are 60–150
    # - timestamps are large and increasing (~0–60)
    #
    # Find approximate segment boundaries:
    # 1. Detect where values start increasing monotonically (timestamps)
    diff = np.diff(numbers)
    monotonic_ratio = np.sum(diff > 0) / len(diff)
    if monotonic_ratio > 0.9 and numbers[-1] > 10:
        # all timestamps, fallback
        raise RuntimeError(
            "File seems to contain only timestamps — please check formatting."
        )

    # Detect timestamps by large value range
    range_ratio = numbers / np.max(numbers)
    idx_transition = np.argmax(range_ratio > 0.5 * np.max(range_ratio))
    # Approx split into three equal parts if not detected
    split_n = n // 3

    # --- heuristic segmentation ---
    ppg_wave = numbers[:split_n]
    t_ppg_s = numbers[-split_n:]

    # Ensure same length
    N = min(len(ppg_wave), len(t_ppg_s))
    ppg_wave = ppg_wave[:N]
    t_ppg_s = t_ppg_s[:N]

    return t_ppg_s, ppg_wave