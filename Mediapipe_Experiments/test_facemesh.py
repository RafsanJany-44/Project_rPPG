import os
import sys
import traceback

import cv2
import numpy as np

import face_mesh  # local file in the same folder


def landmarks_to_xy_array(face_landmarks, width, height):
    pts = []
    for lm in face_landmarks.landmark:
        x_px = int(lm.x * width)
        y_px = int(lm.y * height)
        pts.append((x_px, y_px))
    return np.array(pts, dtype=np.int32)


def draw_selected_landmarks(frame, pts, indices, color, radius=3):
    h, w, _ = frame.shape
    for idx in indices:
        if 0 <= idx < len(pts):
            x, y = pts[idx]
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame, (x, y), radius, color, -1)




def main():
    print("=== test_facemesh.py START ===")
    print("CWD:", os.getcwd())
    print("Python:", sys.version)

    img_path = "test.jpg"
    print("Trying to read image:", img_path)
    frame = cv2.imread(img_path)

    if frame is None:
        print(f"[ERROR] Could not read image: {img_path}")
        return

    h, w, _ = frame.shape
    print(f"Image loaded: {w}x{h}")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    print("Creating FaceMesh() ...")
    mesh = face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print("Running mesh.process(...) ...")
    results = mesh.process(frame_rgb)
    print("process() returned.")

    if not results or not hasattr(results, "multi_face_landmarks"):
        print("[ERROR] results has no 'multi_face_landmarks' field")
        return

    if not results.multi_face_landmarks:
        print("No landmarks detected.")
        return

    print("Landmarks detected!")
    print("Number of faces:", len(results.multi_face_landmarks))
    face_landmarks = results.multi_face_landmarks[0]
    print("Points in first face:", len(face_landmarks.landmark))

    # Convert to numpy array (N, 2)
    # Convert to numpy array (N, 2)
    pts = landmarks_to_xy_array(face_landmarks, w, h)
    print("Landmark array shape:", pts.shape)
    print("First 5 points (x, y):")
    print(pts[:5])

    # Draw all landmarks as small green dots
    for (x_px, y_px) in pts:
        if 0 <= x_px < w and 0 <= y_px < h:
            cv2.circle(frame, (x_px, y_px), 1, (0, 255, 0), -1)

    # Highlight a few specific indices (roughly forehead/cheeks/nose)
    key_indices = {
        "forehead-ish": [10],
        "chin": [152],
        "left_cheek": [234],
        "right_cheek": [454],
    }

    print("Selected key landmarks (index: (x, y)):")
    for name, idx_list in key_indices.items():
        for idx in idx_list:
            if 0 <= idx < len(pts):
                print(f"{name} {idx}: {tuple(pts[idx])}")

    # Draw them with bigger circles in different colors (BGR)
    draw_selected_landmarks(frame, pts, key_indices["forehead-ish"], (0, 0, 255), radius=4)   # red
    draw_selected_landmarks(frame, pts, key_indices["chin"],        (255, 0, 0), radius=4)   # blue
    draw_selected_landmarks(frame, pts, key_indices["left_cheek"],  (0, 255, 255), radius=4) # yellow
    draw_selected_landmarks(frame, pts, key_indices["right_cheek"], (255, 0, 255), radius=4) # magenta


    out_path = "face_with_landmarks.png"
    ok = cv2.imwrite(out_path, frame)
    print("Saved:", out_path, "ok:", ok)
    print("=== test_facemesh.py END ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("UNCAUGHT EXCEPTION:")
        print(e)
        traceback.print_exc()
