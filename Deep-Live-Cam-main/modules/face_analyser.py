import os
import shutil
from typing import Any, List, Dict, Optional

import cv2
import numpy as np
import insightface

import modules.globals as globals_
from modules.typing import Frame
from modules.cluster_analysis import find_cluster_centroids, find_closest_centroid
from modules.utilities import (
    get_temp_directory_path,
    create_temp,
    extract_frames,
    clean_temp,
    get_temp_frame_paths,
)
from pathlib import Path

# Optional: fast camera (not required here, but keep import safe)
try:
    from modules.camera_fast import get_frame as get_cam_frame, get_fps as get_cam_fps
except Exception:
    get_cam_frame = lambda: None
    get_cam_fps = lambda: 0.0

# Cache for the analyser
FACE_ANALYSER: Optional[Any] = None

def get_face_analyser() -> Any:
    """
    Initialize insightface FaceAnalysis once.
    For your AMD iGPU, we keep it on CPU with smaller det_size
    so live mode is less laggy.
    """
    global FACE_ANALYSER
    if FACE_ANALYSER is None:
        # Force CPU – AMD iGPU + ONNX on Windows often stutters
        providers = ["CPUExecutionProvider"]

        FACE_ANALYSER = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=providers,
        )
        # Smaller detection input = faster
        FACE_ANALYSER.prepare(ctx_id=0, det_size=(320, 320))
    return FACE_ANALYSER


# ------------------------
#  Detection helpers
# ------------------------
def get_one_face(frame: Frame) -> Any:
    """
    Detect a single face (the one closest to the left side).
    Completely wrapped in try/except so detector errors do NOT crash UI.
    """
    if frame is None:
        return None
    try:
        faces = get_face_analyser().get(frame)
    except Exception as e:
        print(f"[face_analyser] get_one_face detector error: {e}")
        return None

    if not faces:
        return None
    try:
        # Return the face closest to left edge
        return min(faces, key=lambda x: x.bbox[0])
    except Exception as e:
        print(f"[face_analyser] get_one_face post-processing error: {e}")
        return None


def get_many_faces(frame: Frame) -> Any:
    """
    Detect and return all faces.
    Returns None on any error, so UI won't crash when user adds images.
    """
    if frame is None:
        return None
    try:
        faces = get_face_analyser().get(frame)
    except Exception as e:
        print(f"[face_analyser] get_many_faces detector error: {e}")
        return None

    return faces if faces else None


def create_convex_hull_for_face_mask(face_landmark: Any) -> Any:
    """
    Safely create convex hull for face mask. Returns None if anything fails.
    """
    hull = None
    try:
        if face_landmark is not None and len(face_landmark) > 0:
            hull = cv2.convexHull(np.array(face_landmark, dtype=np.int32))
    except Exception as e:
        print(f"[face_analyser] Warning: Convex hull error ignored: {e}")
    return hull


# ------------------------
#  Mapping helpers
# ------------------------
def has_valid_map() -> bool:
    """Check if there is at least one mapping having both source and target."""
    for m in globals_.source_target_map:
        if "source" in m and "target" in m:
            return True
    return False


def default_source_face() -> Any:
    """Return any stored default source face."""
    for m in globals_.source_target_map:
        if "source" in m and "face" in m["source"]:
            return m["source"]["face"]
    return None


def simplify_maps() -> None:
    """
    Build a simplified map structure which holds:
    - source_faces: list of face objects
    - target_embeddings: corresponding embeddings
    """
    centroids = []
    faces = []
    for m in globals_.source_target_map:
        if "source" in m and "target" in m:
            tgt_face = m["target"].get("face")
            if tgt_face is not None and hasattr(tgt_face, "normed_embedding"):
                centroids.append(tgt_face.normed_embedding)
            src_face = m["source"].get("face") if "source" in m else None
            if src_face is not None:
                faces.append(src_face)

    globals_.simple_map = {"source_faces": faces, "target_embeddings": centroids}


def add_blank_map() -> None:
    """Append a blank mapping entry with auto-incremented id."""
    max_id = -1
    try:
        if globals_.source_target_map:
            max_id = max(
                (x.get("id", -1) for x in globals_.source_target_map), default=-1
            )
    except Exception:
        max_id = len(globals_.source_target_map) - 1

    globals_.source_target_map.append({"id": max_id + 1})


# ------------------------
#  Unique faces from IMAGE
# ------------------------
def get_unique_faces_from_target_image() -> None:
    """
    Detect unique faces from a single target image and populate source_target_map.
    Uses safe clamping of bbox to avoid invalid crops.
    """
    globals_.source_target_map = []
    target_path = getattr(globals_, "target_path", None)
    if not target_path or not os.path.exists(target_path):
        print("⚠ Target image not readable (path missing)!")
        return

    img = cv2.imread(target_path)
    if img is None:
        print("⚠ Target image not readable (cv2.imread failed)!")
        return

    faces = get_many_faces(img)
    if not faces:
        print("⚠ No faces detected in target image!")
        return

    h, w = img.shape[:2]

    for i, face in enumerate(faces):
        try:
            x1, y1, x2, y2 = map(int, face.bbox[:4])
        except Exception:
            continue

        # clamp coordinates within image
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        globals_.source_target_map.append(
            {
                "id": i,
                "target": {
                    "cv2": crop,
                    "face": face,
                },
            }
        )


# ------------------------
#  Unique faces from VIDEO
# ------------------------
def get_unique_faces_from_target_video() -> None:
    """
    Detect unique faces from target video:
      - extract frames
      - run detection + embeddings
      - cluster embeddings -> centroids
      - build source_target_map with target_faces_in_frame
    """
    globals_.source_target_map = []
    embeddings: List[np.ndarray] = []
    frame_face_entries: List[Dict[str, Any]] = []

    target_path = getattr(globals_, "target_path", None)
    if not target_path or not os.path.exists(target_path):
        print("⚠ Target video path not set or not found!")
        return

    print("Creating temp resources...")
    clean_temp(target_path)
    create_temp(target_path)
    print("Extracting frames...")
    extract_frames(target_path)

    frame_paths = get_temp_frame_paths(target_path)
    if not frame_paths:
        print("⚠ No frames extracted from target video!")
        return

    # Detect faces and collect embeddings
    for i, p in enumerate(frame_paths):
        img = cv2.imread(p)
        if img is None:
            continue

        faces = get_many_faces(img)
        if not faces:
            continue

        for face in faces:
            # Only use faces with embeddings
            if hasattr(face, "normed_embedding"):
                face.normed_embedding = np.array(face.normed_embedding)
                embeddings.append(face.normed_embedding)
                face.frame_index = i
                face.source_path = p
                frame_face_entries.append({"frame": i, "face": face, "location": p})

    if not embeddings:
        print("⚠ No valid embeddings found in target video!")
        return

    # Cluster embeddings
    centroids = find_cluster_centroids(embeddings)

    # Assign each detected face to nearest centroid
    for entry in frame_face_entries:
        face = entry["face"]
        idx, _ = find_closest_centroid(centroids, face.normed_embedding)
        face.target_centroid = idx

    # Initialize map entries
    for i in range(len(centroids)):
        globals_.source_target_map.append({"id": i, "target_faces_in_frame": []})

    # Group faces by centroid and record frame info
    for entry in frame_face_entries:
        face = entry["face"]
        cid = getattr(face, "target_centroid", None)
        if cid is None or cid < 0 or cid >= len(globals_.source_target_map):
            continue

        globals_.source_target_map[cid]["target_faces_in_frame"].append(
            {
                "frame": entry["frame"],
                "faces": [face],
                "location": entry["location"],
            }
        )

    # Choose default representative face per cluster
    default_target_face()


def default_target_face() -> None:
    """
    For each entry in source_target_map, choose best representative face
    (highest det_score) from target_faces_in_frame and store as 'target'.
    """
    for m in globals_.source_target_map:
        best_face = None
        best_entry = None

        for entry in m.get("target_faces_in_frame", []):
            faces = entry.get("faces", [])
            for face in faces:
                try:
                    if best_face is None or getattr(face, "det_score", 0.0) > getattr(
                        best_face, "det_score", 0.0
                    ):
                        best_face = face
                        best_entry = entry
                except Exception:
                    continue

        if best_face is None or best_entry is None:
            continue

        img_src = cv2.imread(best_entry["location"])
        if img_src is None:
            continue

        h, w = img_src.shape[:2]
        try:
            x1, y1, x2, y2 = map(int, best_face.bbox[:4])
        except Exception:
            continue

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = img_src[y1:y2, x1:x2]
        if crop.size > 0:
            m["target"] = {"cv2": crop, "face": best_face}


# ------------------------
#  Dump faces (optional debug)
# ------------------------
def dump_faces(centroids: Any, frame_face_embeddings: list):
    """
    Debug helper: dump faces into temp directories grouped by centroid.
    Keep bboxes clamped to avoid crashes.
    """
    temp_dir = get_temp_directory_path(globals_.target_path)

    for i in range(len(centroids)):
        sub = f"{temp_dir}/{i}"
        if os.path.exists(sub):
            shutil.rmtree(sub)
        Path(sub).mkdir(parents=True, exist_ok=True)

        for item in frame_face_embeddings:
            img = cv2.imread(item["location"])
            if img is None:
                continue
            h, w = img.shape[:2]
            for j, face in enumerate(item.get("faces", [])):
                if hasattr(face, "target_centroid") and face.target_centroid == i:
                    try:
                        x1, y1, x2, y2 = map(int, face.bbox[:4])
                    except Exception:
                        continue

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = img[y1:y2, x1:x2]
                    if crop.size > 0:
                        cv2.imwrite(f"{sub}/{item['frame']}_{j}.png", crop)

    return None
