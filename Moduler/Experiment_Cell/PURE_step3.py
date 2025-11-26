# step3_check_image_filenames_vs_json.py
"""
STEP 3: Verify that PURE image filenames match JSON /Image timestamps.

Goals:
- Ensure that for each JSON /Image entry, there is exactly one
  'Image<TIMESTAMP>.png' file with the same timestamp.
- Confirm that no frames are missing or mis-ordered between JSON
  and the extracted PNG frames.
"""
import sys
from pathlib import Path
# Add project root (Project_rPPG) to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import numpy as np

from Main.Data_Read_Engine import DEFAULT_SEQ_ID
from Main.Data_Read_Engine import load_pure_json, extract_streams_from_pure_json
from Main.Data_Read_Engine import load_pure_image_files_and_timestamps


def main():
    seq_id = DEFAULT_SEQ_ID
    print(f"=== STEP 3: Check image filenames vs JSON timestamps for {seq_id} ===")

    # 1) Load JSON and extract video timestamps
    data = load_pure_json(seq_id)
    _, _, t_vid_ns, _ = extract_streams_from_pure_json(data)

    print(f"JSON /Image entries (video frames): {len(t_vid_ns)}")

    # 2) Load image files and timestamps from filenames
    image_files, t_img_ns = load_pure_image_files_and_timestamps(seq_id)
    print(f"PNG frame files (Image*.png):        {len(image_files)}")

    # 3) Check that counts match
    if len(t_vid_ns) != len(t_img_ns):
        print("\n[WARNING] Frame count mismatch:")
        print(f"  JSON frames:  {len(t_vid_ns)}")
        print(f"  PNG frames:   {len(t_img_ns)}")
    else:
        print("\nFrame counts match.")

    # 4) Compare timestamps element-wise
    #    JSON /Image entries should already be sorted by time,
    #    and image_files were sorted by parsed timestamp.
    same_shape = t_vid_ns.shape == t_img_ns.shape
    same_values = np.array_equal(t_vid_ns, t_img_ns)

    print("\n--- Timestamp comparison ---")
    print(f"Same shape:   {same_shape}")
    print(f"Same values:  {same_values}")

    # 5) Show first few pairs for manual inspection
    N_SHOW = min(10, len(t_vid_ns), len(t_img_ns))
    print(f"\nFirst {N_SHOW} frame timestamps (JSON vs filename):")
    for i in range(N_SHOW):
        json_ts = t_vid_ns[i]
        img_ts = t_img_ns[i]
        fname = image_files[i].name
        diff = img_ts - json_ts
        print(
            f"[{i:3d}]  JSON: {json_ts}   "
            f"IMG: {img_ts} ({fname})   "
            f"diff = {diff}"
        )

    # 6) Final interpretation
    if same_shape and same_values:
        print("\nResult: JSON /Image timestamps and Image*.png filenames match exactly.")
        print("This confirms the 1:1 mapping between JSON frames and PNG frames.")
    else:
        print("\nResult: There is a mismatch between JSON /Image timestamps and PNG filenames.")
        print("We should inspect the printed differences and directory structure carefully.")

    print("\nStep 3 completed.")


if __name__ == "__main__":
    main()
